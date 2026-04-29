# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup


class GateComputeMode(Enum):
    """Controls which parts of the gate computation fall back to host (torch).

    The gate has two stages: matmul (x @ W_gate) and grouped_gate (topk routing).
    Each can independently run on device (TTNN) or host (torch).
    The device grouped-gate has two variants: bf16 (default) and fp32.
    """

    DEVICE = "device"  # matmul device, grouped gate device (bf16)
    DEVICE_FP32 = "device_fp32"  # matmul device, grouped gate device (fp32)
    HOST_GROUPED_GATE = "host_grouped_gate"  # matmul device, grouped gate host
    HOST_MATMUL = "host_matmul"  # matmul host, grouped gate device (bf16)
    HOST_ALL = "host_all"  # matmul host, grouped gate host


@dataclass
class TtMoEGateConfig:
    # gate_params

    ccl_config: dict = field(default_factory=lambda: {"DISPATCH_AXIS": 0, "TP_AXIS": 1, "NUM_LINKS": 2})
    mm_configs: dict = field(
        default_factory=lambda: {
            # Keyed by (sp_dim, per_device_emb_dim); forward() looks up the tuple.
            # Missing key → TTNN auto-picks program config.
            (4096, DeepSeekV3Config.EMB_SIZE // 4): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                in0_block_w=56,
                out_subblock_h=2,
                out_subblock_w=4,
                out_block_h=2,
                out_block_w=4,
                per_core_M=2,
                per_core_N=8,
                fuse_batch=True,
                mcast_in0=False,
            ),
            "COMPUTE_CONFIG": ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
        }
    )

    dim: int = DeepSeekV3Config.EMB_SIZE
    sp_dim: int = 4096  # ISL per chip
    n_routed_experts: int = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    n_shared_experts: int = DeepSeekV3Config.NUM_SHARED_EXPERTS  # PREVIOUS VALUE: 2 @ddjekic to check
    n_activated_experts: int = DeepSeekV3Config.NUM_EXPERTS_PER_TOKEN
    n_expert_groups: int = DeepSeekV3Config.NUM_EXPERT_GROUPS
    n_limited_groups: int = DeepSeekV3Config.NUM_LIMITED_GROUPS  # = topk_groups
    route_scale: float = DeepSeekV3Config.ROUTE_SCALE  # PREVIOUS VALUE: 1.0 @ddjekic to check
    score_func: str = "sigmoid"

    # grid_config
    core_grid: ttnn.CoreRangeSet = field(
        default_factory=lambda: (
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
            if is_blackhole()
            else ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        )
    )

    @property
    def num_cores(self):
        return self.core_grid.num_cores()


class TtMoEGatePrefill(LightweightModule):
    """MoE gate module from DeepSeek-R1."""

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str) -> bool:
        """Check if gate weight and bias cache files exist."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        if not pattern_exists(f"{cache_name_prefix}.weight*.tensorbin", "MoEGate"):
            logger.debug(f"TTNN cache missing: {cache_name_prefix}.weight")
            return False
        if not pattern_exists(f"{cache_name_prefix}.e_score_correction_bias*.tensorbin", "MoEGate"):
            logger.debug(f"TTNN cache missing: {cache_name_prefix}.e_score_correction_bias")
            return False
        return True

    @staticmethod
    def _convert_and_cache_gate_weights(
        torch_weight: torch.Tensor,  # (n_experts, dim) - HF format
        torch_bias: torch.Tensor,  # (n_experts,)
        config: TtMoEGateConfig,
        mesh_device: ttnn.MeshDevice,
        cache_path: Path | None,
        cache_name_prefix: str | None,
        device: ttnn.MeshDevice | None = None,  # None=cache, mesh_device=load
    ) -> dict | None:
        """
        Shared logic for converting gate weights to TTNN with caching.

        Bias handling: Cache stores unbroadcasted (n_experts,) format. On load,
        returns unbroadcasted bias for caller to broadcast to (sp_dim, n_experts).
        This is required by ttnn.experimental.deepseek_grouped_gate kernel.

        Returns:
            If device=None (cache mode): None
            If device=mesh_device (load mode): Dict with:
                - "weight": ttnn.Tensor on device
                - "bias_unbroadcasted": ttnn.Tensor on host (needs broadcasting)
                - "torch_weight": torch.Tensor in HF format (for fallback)
                - "torch_bias": torch.Tensor (for fallback)
        """

        def _cache_name(name):
            if cache_path is None or cache_name_prefix is None:
                return None
            return str(cache_path / f"{cache_name_prefix}.{name}")

        # Transpose weight from HF (n_experts, dim) to TTNN (dim, n_experts)
        if torch_weight is not None:
            weight_for_ttnn = torch_weight.T
        else:
            weight_for_ttnn = torch.empty(config.dim, config.n_routed_experts)

        if torch_bias is None:
            torch_bias = torch.empty(config.n_routed_experts)

        # Convert weight
        weight_tt = ttnn.as_tensor(
            weight_for_ttnn,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0),
                mesh_shape=mesh_device.shape,
            ),
            cache_file_name=_cache_name("weight"),
        )

        # Cache bias unbroadcasted (required by deepseek_grouped_gate)
        bias_tt = ttnn.as_tensor(
            torch_bias,
            device=None,  # Always load to host first
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=_cache_name("e_score_correction_bias"),
        )

        if device is None:
            # Cache-only mode
            del weight_tt, bias_tt
            return None
        else:
            # Load mode - return tensors for __init__
            # For host fallback modes, we need actual torch tensors (not dummy zeros)
            # Convert loaded TTNN tensors back to torch format
            weight_torch_loaded = ttnn.to_torch(
                weight_tt,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device,
                    config=ttnn.MeshComposerConfig(
                        dims=(-1, 0),  # Work on tensor dims 0 and -1 (both dims of 2D tensor)
                        mesh_shape_override=ttnn.MeshShape(
                            1,  # SP replicated
                            mesh_device.shape[1],  # TP fractured
                        ),
                    ),
                ),
            )
            torch_weight_hf = weight_torch_loaded.T  # Transpose to HF format: (n_experts, dim)

            # Convert bias: already on host as unbroadcasted (n_experts,)
            bias_torch_loaded = ttnn.to_torch(bias_tt)

            return {
                "weight": weight_tt,
                "bias_unbroadcasted": bias_tt,
                "torch_weight": torch_weight_hf,  # Converted from cache (not dummy!)
                "torch_bias": bias_torch_loaded,  # Converted from cache (not dummy!)
            }

    @staticmethod
    def build_ttnn_cache(
        torch_weight: torch.Tensor,
        torch_bias: torch.Tensor,
        config: TtMoEGateConfig,
        mesh_device: ttnn.MeshDevice,
        cache_path: Path,
        cache_name_prefix: str,
    ):
        """Build TTNN cache for gate weights without device copy."""
        TtMoEGatePrefill._convert_and_cache_gate_weights(
            torch_weight, torch_bias, config, mesh_device, cache_path, cache_name_prefix, device=None
        )

    def __init__(
        self,
        config,
        mesh_device,
        dispatch_table: torch.Tensor,
        experts_per_chip: int,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        fallback_mode: GateComputeMode = GateComputeMode.DEVICE,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
    ):
        """
        Args:
            weight: Gate weight in HF convention: (n_routed_experts, dim).
                    Transposed internally to (dim, n_routed_experts) for the TTNN matmul path.
            experts_per_chip: Number of experts per chip (for expert region offset grouping in offset_cumsum).
        """
        self.config = config
        self.mesh_device = mesh_device
        self.fallback_mode = fallback_mode

        if weight is not None and bias is not None:
            weights = self._convert_and_cache_gate_weights(
                weight, bias, config, mesh_device, weight_cache_path, cache_name_prefix, device=mesh_device
            )
        elif weight_cache_path is not None:
            weights = self._convert_and_cache_gate_weights(
                None, None, config, mesh_device, weight_cache_path, cache_name_prefix, device=mesh_device
            )
        else:
            weights = self._convert_and_cache_gate_weights(
                torch.zeros([config.n_routed_experts, config.dim]),
                torch.zeros([config.n_routed_experts]),
                config,
                mesh_device,
                None,
                None,
                device=mesh_device,
            )

        self.weight = weights["weight"]
        bias_tt = weights["bias_unbroadcasted"]
        torch_weight_fallback = weights["torch_weight"]
        torch_bias_fallback = weights["torch_bias"]

        # Broadcast bias for deepseek_grouped_gate kernel
        bias_torch = ttnn.to_torch(bias_tt)
        del bias_tt
        bias_broadcasted = bias_torch.repeat(config.sp_dim).view(config.sp_dim, -1)
        self.bias = ttnn.from_torch(
            bias_broadcasted,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.routing_setup = TtMoERoutingSetup(
            mesh_device, dispatch_table, num_links=config.ccl_config["NUM_LINKS"], experts_per_chip=experts_per_chip
        )

        # Torch copies for host fallback paths — keep in HF convention (n_experts, dim)
        if fallback_mode not in (GateComputeMode.DEVICE, GateComputeMode.DEVICE_FP32):
            self.torch_weight = torch_weight_fallback  # (n_experts, dim) - HF format
            self.torch_bias = torch_bias_fallback  # (n_experts,)

        # Reference model for host grouped-gate paths
        if fallback_mode in (GateComputeMode.HOST_GROUPED_GATE, GateComputeMode.HOST_ALL):
            from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate

            self.ref_config = SimpleNamespace(
                num_experts_per_tok=config.n_activated_experts,
                n_routed_experts=config.n_routed_experts,
                routed_scaling_factor=config.route_scale,
                scoring_func=config.score_func,
                topk_method="noaux_tc",
                n_group=config.n_expert_groups,
                topk_group=config.n_limited_groups,
                norm_topk_prob=True,
                hidden_size=config.dim,
            )
            self.reference_model = ReferenceMoEGate(self.ref_config, use_bitonic_sort=True)
            self.reference_model.weight.data = self.torch_weight  # (n_experts, dim)
            self.reference_model.e_score_correction_bias.data = self.torch_bias  # (n_experts,)

    # ------------------------------------------------------------------
    # Helpers: compose / shard patterns reused across fallback modes
    # ------------------------------------------------------------------

    def _compose_x_to_host(self, x: ttnn.Tensor) -> torch.Tensor:
        """Compose TP+SP sharded x back to a single host tensor (tokens, dim)."""
        return ttnn.to_torch(
            x,
            mesh_composer=ttnn.create_mesh_composer(
                self.mesh_device,
                config=ttnn.MeshComposerConfig(
                    dims=(0, -1),  # SP on axis 0, TP on axis 1
                ),
            ),
        )

    def _compose_logits_to_host(self, logits: ttnn.Tensor) -> torch.Tensor:
        """Compose SP-sharded logits (TP-replicated after all_reduce) to host."""
        return ttnn.to_torch(
            logits,
            mesh_composer=ttnn.create_mesh_composer(
                self.mesh_device,
                config=ttnn.MeshComposerConfig(
                    dims=(0, -1),
                    mesh_shape_override=ttnn.MeshShape(
                        self.mesh_device.shape[0],  # concat SP shards
                        1,  # collapse TP replicas
                    ),
                ),
            ),
        )

    def _host_logits_to_device(self, host_logits: torch.Tensor) -> ttnn.Tensor:
        """Shard host logits (tokens, n_experts) back to device: SP-sharded, TP-replicated."""
        return ttnn.from_torch(
            host_logits,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(0, None),  # shard SP, replicate TP
                mesh_shape=self.mesh_device.shape,
            ),
        )

    def _host_scores_to_device(self, host_scores: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            host_scores,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(0, None),
                mesh_shape=self.mesh_device.shape,
            ),
        )

    def _host_indices_to_device(self, host_indices: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            host_indices.to(torch.int16),
            device=self.mesh_device,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(0, None),
                mesh_shape=self.mesh_device.shape,
            ),
        )

    def _device_matmul(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Gate matmul + TP all-reduce on device."""
        per_device_dim = x.shape[-1]
        n_tp_devices = self.mesh_device.shape[1]
        assert (
            per_device_dim * n_tp_devices == self.config.dim
        ), f"Expected per-device dim {self.config.dim // n_tp_devices}, got {per_device_dim}"
        config_key = (self.config.sp_dim, per_device_dim)
        program_config = self.config.mm_configs.get(config_key)
        if program_config is None:
            logger.warning(f"[MoeGate] No matmul program config for {config_key}, using TTNN default")

        logits = ttnn.matmul(
            x,
            self.weight,
            compute_kernel_config=self.config.mm_configs["COMPUTE_CONFIG"],
            program_config=program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if self.mesh_device.shape[self.config.ccl_config["TP_AXIS"]] > 1:
            logits = ttnn.experimental.all_reduce_async(
                logits,
                cluster_axis=self.config.ccl_config["TP_AXIS"],
                mesh_device=self.mesh_device,
                num_links=self.config.ccl_config["NUM_LINKS"],
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )
        return logits

    def _host_matmul(self, x: ttnn.Tensor) -> torch.Tensor:
        """Compose x to host, run gate matmul in torch, return host logits."""
        host_x = self._compose_x_to_host(x)
        return F.linear(host_x.float(), self.torch_weight.float())

    def _device_grouped_gate(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run deepseek_grouped_gate on device."""
        logger.debug(f"[MoeGate] _device_grouped_gate: logits.shape={logits.shape}, bias.shape={self.bias.shape}")
        return ttnn.experimental.deepseek_grouped_gate(
            logits,
            self.bias,
            n_groups=self.config.n_expert_groups,
            summed_experts_per_group=self.config.n_expert_groups // self.config.n_limited_groups,
            topk_groups=self.config.n_limited_groups,
            n_activated_experts=self.config.n_activated_experts,
            route_scale=self.config.route_scale,
            epsilon=1e-20,
        )

    def _device_grouped_gate_fp32(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run moe_grouped_topk on device with fp32 typecast."""
        logits_f32 = ttnn.typecast(logits, ttnn.float32)
        bias_f32 = ttnn.typecast(self.bias, ttnn.float32)
        ttnn_scores, ttnn_top_k_experts_indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits_f32,
            bias_f32,
            n_groups=self.config.n_expert_groups,
            summed_experts_per_group=self.config.n_expert_groups // self.config.n_limited_groups,
            topk_groups=self.config.n_limited_groups,
            n_activated_experts=self.config.n_activated_experts,
            route_scale=self.config.route_scale,
            stable_sort=True,
            epsilon=1e-20,
        )
        ttnn.deallocate(logits_f32)
        ttnn.deallocate(bias_f32)
        return ttnn_scores, ttnn_top_k_experts_indices

    def _host_grouped_gate(self, host_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run grouped_gate_golden on host. Returns (indices, scores)."""
        return self.reference_model.grouped_gate_golden(
            host_logits,
            self.torch_bias,
            self.config.route_scale,
            1e-20,
            self.config.n_expert_groups,
            self.config.n_expert_groups // self.config.n_limited_groups,
            self.config.n_limited_groups,
            self.config.n_activated_experts,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        mode = self.fallback_mode
        logger.debug(f"[MoeGate] fallback_mode={mode.value}")

        # ---- Phase 1: Logits (matmul) ----
        signpost(header="moe_gate_linear")
        if mode in (GateComputeMode.DEVICE, GateComputeMode.DEVICE_FP32, GateComputeMode.HOST_GROUPED_GATE):
            logits = self._device_matmul(x)
        else:  # HOST_MATMUL, HOST_ALL
            host_logits = self._host_matmul(x)
        signpost(header="moe_gate_linear")

        # ---- Phase 2: Grouped gate ----
        signpost(header="moe_gate_grouped_gate")
        if mode == GateComputeMode.DEVICE:
            ttnn_scores, ttnn_top_k_experts_indices = self._device_grouped_gate(logits)

        elif mode == GateComputeMode.DEVICE_FP32:
            ttnn_scores, ttnn_top_k_experts_indices = self._device_grouped_gate_fp32(logits)

        elif mode == GateComputeMode.HOST_GROUPED_GATE:
            host_logits = self._compose_logits_to_host(logits)
            host_indices, host_scores = self._host_grouped_gate(host_logits)
            ttnn_scores = self._host_scores_to_device(host_scores)
            ttnn_top_k_experts_indices = self._host_indices_to_device(host_indices)

        elif mode == GateComputeMode.HOST_MATMUL:
            logits = self._host_logits_to_device(host_logits)
            ttnn_scores, ttnn_top_k_experts_indices = self._device_grouped_gate(logits)

        elif mode == GateComputeMode.HOST_ALL:
            host_indices, host_scores = self._host_grouped_gate(host_logits)
            ttnn_scores = self._host_scores_to_device(host_scores)
            ttnn_top_k_experts_indices = self._host_indices_to_device(host_indices)
            logits = self._host_logits_to_device(host_logits)
        signpost(header="moe_gate_grouped_gate")

        # ---- Phase 3: Routing setup ----
        signpost(header="moe_gate_calculate_dispatch_offsets")
        ttnn_top_k_experts_indices = ttnn.to_layout(ttnn_top_k_experts_indices, ttnn.ROW_MAJOR_LAYOUT)

        dispatch_offsets, total_counts_per_expert, expert_region_offsets, _ = self.routing_setup(
            ttnn_top_k_experts_indices=ttnn_top_k_experts_indices,
            num_routed_experts=self.config.n_routed_experts,
            seq_len_per_chip=self.config.sp_dim,
            num_experts_per_tok=self.config.n_activated_experts,
        )
        signpost(header="moe_gate_calculate_dispatch_offsets")

        return (
            ttnn_scores,
            ttnn_top_k_experts_indices,
            logits,
            dispatch_offsets,
            total_counts_per_expert,
            expert_region_offsets,
        )
