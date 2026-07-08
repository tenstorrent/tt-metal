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
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl


class GateComputeMode(Enum):
    """Controls which parts of the gate computation fall back to host (torch).

    The gate has two stages: matmul (x @ W_gate) and grouped_gate (topk routing).
    Each can independently run on device (TTNN) or host (torch).
    The device gate has two precision variants: bf16 (default) and fp32.

    The device gate routing rule is selected from the model config: the grouped-topk
    op handles both cases, collapsing to a plain top-k when there is a single expert
    group (n_expert_groups == 1, e.g. Kimi) and using grouped routing otherwise.
    """

    DEVICE = "device"  # matmul device, gate device (bf16)
    DEVICE_FP32 = "device_fp32"  # matmul device, gate device (fp32)
    HOST_GROUPED_GATE = "host_grouped_gate"  # matmul device, grouped gate host
    HOST_MATMUL = "host_matmul"  # matmul host, grouped gate device (bf16)
    HOST_ALL = "host_all"  # matmul host, grouped gate host
    # DeepSeek-V4 hash routing: expert indices come from a static tid2eid[input_ids] table
    # (not top-k); weights are still score_func(x@W) gathered at those indices, normalized, scaled.
    # Host-first implementation reusing the V4 reference HashRouter.
    HASH_HOST = "hash_host"
    # DeepSeek-V4 hash routing fully on device: matmul device, moe_hash_gate device. The tid2eid[input_ids]
    # lookup is fused into the op's reader kernel; weights reuse the shared activation/normalize/scale path.
    HASH_DEVICE = "hash_device"


@dataclass
class TtMoEGateConfig:
    # gate_params

    ccl_config: dict = field(default_factory=lambda: {"DISPATCH_AXIS": 0, "TP_AXIS": 1, "NUM_LINKS": 2})
    mm_configs: dict = field(
        default_factory=lambda: {
            # Keyed by (sp_dim, per_device_emb_dim, n_routed_experts); forward() looks up the tuple.
            # The seq-len element below is a placeholder — __post_init__ rewrites it to the actual
            # per-chip sequence length (self.sp_dim) so the lookup tracks the real workload.
            # per_core_N = n_routed_experts / 32 (tile width). Missing key → TTNN auto-picks.
            (4096, DeepSeekV3Config.EMB_SIZE // 4, DeepSeekV3Config.NUM_ROUTED_EXPERTS): (
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                    in0_block_w=56,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    out_block_h=2,
                    out_block_w=4,
                    per_core_M=2,
                    per_core_N=8,
                    fuse_batch=True,
                    mcast_in0=False,
                )
            ),
            (4096, KimiK26Config.EMB_SIZE // 4, KimiK26Config.NUM_ROUTED_EXPERTS): (
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                    in0_block_w=56,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    out_block_h=2,
                    out_block_w=4,
                    per_core_M=2,
                    per_core_N=12,
                    fuse_batch=True,
                    mcast_in0=False,
                )
            ),
            "COMPUTE_CONFIG": ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
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

    def __post_init__(self):
        # The mm_configs tuple keys are authored with a placeholder seq-len. Re-key them to the
        # actual per-chip sequence length (sp_dim) so _device_matmul's lookup
        # (sp_dim, per_device_emb_dim, n_routed_experts) hits the tuned program config instead of
        # silently falling back to TTNN's default tiling.
        self.mm_configs = {
            ((self.sp_dim, *key[1:]) if isinstance(key, tuple) else key): value
            for key, value in self.mm_configs.items()
        }

    @property
    def num_cores(self):
        return self.core_grid.num_cores()

    @classmethod
    def from_model_cfg(cls, model_cfg: type, **overrides) -> "TtMoEGateConfig":
        """Build from a TestVariant.model_config class. Extra kwargs override per-instance."""
        return cls(
            dim=model_cfg.EMB_SIZE,
            n_routed_experts=model_cfg.NUM_ROUTED_EXPERTS,
            n_shared_experts=model_cfg.NUM_SHARED_EXPERTS,
            n_activated_experts=model_cfg.NUM_EXPERTS_PER_TOKEN,
            n_expert_groups=model_cfg.NUM_EXPERT_GROUPS,
            n_limited_groups=model_cfg.NUM_LIMITED_GROUPS,
            route_scale=model_cfg.ROUTE_SCALE,
            # V4 variants ship SCORE_FUNC="sqrtsoftplus"; V3/Kimi omit it and keep the sigmoid default.
            score_func=getattr(model_cfg, "SCORE_FUNC", cls.score_func),
            **overrides,
        )


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
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        fallback_mode: GateComputeMode = GateComputeMode.DEVICE,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        is_balanced: bool = False,
        hash_table: torch.Tensor = None,
    ):
        """
        Args:
            weight: Gate weight in HF convention: (n_routed_experts, dim).
                    Transposed internally to (dim, n_routed_experts) for the TTNN matmul path.
            is_balanced: If True, uses zigzag (balanced) sequence placement across SP devices.
                Affects per-device real token count computation for padding awareness.
            hash_table: DeepSeek-V4 hash routing tid2eid table, shape (vocab_size, n_activated_experts).
                Required for GateComputeMode.HASH_HOST and GateComputeMode.HASH_DEVICE; ignored otherwise.
        """
        self.config = config
        self.mesh_device = mesh_device
        # Shared per-mesh CCL singleton: provides persistent global semaphores for the gate's TP
        # all-reduce so the op reuses them instead of allocating fresh L1 semaphores every layer
        # (those leaked, pinning the L1 floor and clashing with the next layer's ring_mla CBs).
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.fallback_mode = fallback_mode
        self.is_balanced = is_balanced

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

        # DeepSeek-V4 hash routing: reuse the V4 reference HashRouter on host (indices via tid2eid
        # lookup; weights via score_func(x@W) gathered/normalized/scaled). No bespoke gate math here.
        if fallback_mode == GateComputeMode.HASH_HOST:
            from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HashRouter

            if hash_table is None:
                raise ValueError("GateComputeMode.HASH_HOST requires a hash_table (tid2eid).")
            self.hash_ref_config = SimpleNamespace(
                num_experts_per_tok=config.n_activated_experts,
                num_local_experts=config.n_routed_experts,
                hidden_size=config.dim,
                scoring_func=config.score_func,
                routed_scaling_factor=config.route_scale,
                vocab_size=hash_table.shape[0],
            )
            self.hash_router = DeepseekV4HashRouter(self.hash_ref_config)
            self.hash_router.eval()
            self.hash_router.weight.data = self.torch_weight  # (n_experts, dim), HF convention
            self.hash_router.tid2eid.data = hash_table.to(torch.long)

        # DeepSeek-V4 hash routing on device: ship the tid2eid table (replicated across the mesh) so the
        # moe_hash_gate reader can fuse tid2eid[input_ids]. Rows are padded to 16 uint16 (32-byte aligned)
        # for NoC page reads; the first n_activated_experts columns hold the expert ids.
        if fallback_mode == GateComputeMode.HASH_DEVICE:
            if hash_table is None:
                raise ValueError("GateComputeMode.HASH_DEVICE requires a hash_table (tid2eid).")
            self.tid2eid_dev = self._prepare_tid2eid_device(hash_table)

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

    def _prepare_tid2eid_device(self, hash_table: torch.Tensor) -> ttnn.Tensor:
        """Ship the tid2eid table replicated to all devices for the device hash gate.

        hash_table is (vocab_size, n_activated_experts). Rows are padded to 16 uint16 columns so each
        table row is a 32-byte NoC page (the moe_hash_gate reader indexes it by token id). Expert ids
        fit exactly in uint16 (unlike bf16, which cannot represent ids > 256 -> matters for dsv4_pro's
        384 experts), so this stays exact for any expert count.
        """
        vocab_size, n_act = hash_table.shape
        padded_cols = 16
        # int16 (not int32) host source: ttnn.from_torch does not narrow int32 -> uint16, so an int32
        # source leaves each expert id as 4 bytes (id, 0) and the uint16 reader reads every other slot
        # as 0. int16 matches the on-device uint16 element width (expert ids fit in int16).
        padded = torch.zeros((vocab_size, padded_cols), dtype=torch.int16)
        padded[:, :n_act] = hash_table.to(torch.int16)
        return ttnn.from_torch(
            padded,
            device=self.mesh_device,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _input_ids_to_device(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """Shard host input_ids to device to match x's SP token order.

        input_ids (any shape, flattened to [total_tokens]) is reshaped to [total_tokens/32, 32] and
        SP-sharded on axis 0 / TP-replicated, so device d holds the same tokens as its logits rows.
        One ROW_MAJOR page = 32 uint32 token ids = one score height tile. (Assumes sequential SP
        placement; balanced/zigzag would need a matching token permutation.)
        """
        ids = input_ids.reshape(-1)
        total_tokens = ids.shape[0]
        assert total_tokens % 32 == 0, f"input_ids length ({total_tokens}) must be a multiple of 32"
        ids = ids.reshape(total_tokens // 32, 32)
        return ttnn.from_torch(
            ids.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(0, None),  # shard tokens across SP, replicate across TP
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
        config_key = (self.config.sp_dim, per_device_dim, self.config.n_routed_experts)
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
        tp_axis = self.config.ccl_config["TP_AXIS"]
        if self.mesh_device.shape[tp_axis] > 1:
            # Pass persistent CCL semaphores (created once in TT_CCL) so all_reduce_async reuses them
            # instead of internally allocating+leaking global semaphores in main L1 every call. The
            # composite all-reduce needs barrier_semaphores of size 2 ([0]=reduce-scatter, [1]=all-gather),
            # plus the reduce-scatter (3) and all-gather (2) semaphore vectors.
            logits = ttnn.experimental.all_reduce_async(
                logits,
                cluster_axis=tp_axis,
                mesh_device=self.mesh_device,
                barrier_semaphores=self.tt_ccl.barrier_semaphore_handles[tp_axis],
                rs_global_semaphores=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=tp_axis),
                ag_global_semaphores=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=tp_axis),
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

    def build_padding_config(self, actual_isl: int, padding_side: str = "right") -> ttnn.Tensor:
        """Create the per-SP-shard [local_num_real_tokens, pad_side] config for moe_grouped_topk.

        Public so callers (TtMoe) can build the config once and share the same tensor between
        the gate topk and the dispatch op.

        When is_balanced=True, the sequence uses zigzag placement: the original sequence
        is split into 2*sp_factor chunks and device d holds chunks d and (2*sp_factor-1-d),
        early chunk first. Padding remains contiguous on the expected side within each
        device's local buffer because early chunks always precede late chunks locally.
        Only the per-device real token count changes relative to the sequential case.
        """
        if padding_side not in ("right", "left"):
            raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}")

        sp_factor = self.mesh_device.shape[0]
        seq_len_per_chip = self.config.sp_dim
        total_tokens = sp_factor * seq_len_per_chip
        pad_side = 0 if padding_side == "right" else 1

        padding_config = []

        if self.is_balanced:
            num_chunks = 2 * sp_factor
            chunk_size = total_tokens // num_chunks

            for sp_idx in range(sp_factor):
                chunk_a = sp_idx
                chunk_b = num_chunks - 1 - sp_idx

                if padding_side == "right":
                    real_a = min(chunk_size, max(0, actual_isl - chunk_a * chunk_size))
                    real_b = min(chunk_size, max(0, actual_isl - chunk_b * chunk_size))
                else:
                    total_padded = max(0, total_tokens - actual_isl)
                    pad_a = min(chunk_size, max(0, total_padded - chunk_a * chunk_size))
                    pad_b = min(chunk_size, max(0, total_padded - chunk_b * chunk_size))
                    real_a = chunk_size - pad_a
                    real_b = chunk_size - pad_b

                padding_config.append([real_a + real_b, pad_side])
        else:
            for sp_idx in range(sp_factor):
                if padding_side == "right":
                    local_real_tokens = min(seq_len_per_chip, max(0, actual_isl - sp_idx * seq_len_per_chip))
                else:
                    total_padded_tokens = max(0, total_tokens - actual_isl)
                    local_padded_tokens = min(seq_len_per_chip, max(0, total_padded_tokens - sp_idx * seq_len_per_chip))
                    local_real_tokens = seq_len_per_chip - local_padded_tokens

                padding_config.append([local_real_tokens, pad_side])

        return ttnn.from_torch(
            torch.tensor(padding_config, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(0, None),
                mesh_shape=self.mesh_device.shape,
            ),
        )

    def _device_grouped_gate_fp32(
        self,
        logits: ttnn.Tensor,
        actual_isl: int = None,
        padding_side: str = "right",
        padding_config: ttnn.Tensor = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run moe_grouped_topk on device with fp32 typecast.

        When actual_isl is set, padded token rows get sentinel expert
        indices (= n_routed_experts) so downstream masked_bincount/dispatch/
        combine skip them.  For SP > 1, the padding config tensor carries
        per-device local real-token counts.

        If a caller-owned ``padding_config`` is provided it is used as-is (and the
        caller is responsible for deallocating it, since it may be shared with the
        dispatch op). Otherwise one is built locally and freed here.
        """
        owns_padding_config = padding_config is None
        if owns_padding_config:
            padding_config = self.build_padding_config(actual_isl, padding_side) if actual_isl is not None else None

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
            score_func=self.config.score_func,
            padding_config=padding_config,
        )
        ttnn.deallocate(logits_f32)
        ttnn.deallocate(bias_f32)
        if owns_padding_config and padding_config is not None:
            ttnn.deallocate(padding_config)
        return ttnn_scores, ttnn_top_k_experts_indices

    def _device_hash_gate(
        self,
        logits: ttnn.Tensor,
        input_ids: torch.Tensor,
        actual_isl: int = None,
        padding_side: str = "right",
        padding_config: ttnn.Tensor = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run moe_hash_gate on device: fused tid2eid[input_ids] routing + score_func/normalize/scale.

        Mirrors _device_grouped_gate_fp32's fp32 typecast and padding-config ownership, but expert
        selection comes from the hash table instead of top-k.
        """
        if input_ids is None:
            raise ValueError("GateComputeMode.HASH_DEVICE forward requires input_ids for the tid2eid lookup.")

        owns_padding_config = padding_config is None
        if owns_padding_config:
            padding_config = self.build_padding_config(actual_isl, padding_side) if actual_isl is not None else None

        logits_f32 = ttnn.typecast(logits, ttnn.float32)
        input_ids_dev = self._input_ids_to_device(input_ids)
        ttnn_scores, ttnn_top_k_experts_indices = ttnn.experimental.deepseek_prefill.moe_hash_gate(
            logits_f32,
            input_ids_dev,
            self.tid2eid_dev,
            n_activated_experts=self.config.n_activated_experts,
            route_scale=self.config.route_scale,
            epsilon=1e-20,
            score_func=self.config.score_func,
            padding_config=padding_config,
        )
        ttnn.deallocate(logits_f32)
        ttnn.deallocate(input_ids_dev)
        if owns_padding_config and padding_config is not None:
            ttnn.deallocate(padding_config)
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

    def forward(
        self,
        x: ttnn.Tensor,
        actual_isl: int = None,
        padding_side: str = "right",
        padding_config: ttnn.Tensor = None,
        input_ids: torch.Tensor = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        mode = self.fallback_mode
        logger.debug(f"[MoeGate] fallback_mode={mode.value}")

        # ---- Phase 1: Logits (matmul) ----
        signpost(header="moe_gate_linear")
        if mode in (
            GateComputeMode.DEVICE,
            GateComputeMode.DEVICE_FP32,
            GateComputeMode.HOST_GROUPED_GATE,
            GateComputeMode.HASH_DEVICE,
        ):
            logits = self._device_matmul(x)
        elif mode == GateComputeMode.HASH_HOST:
            pass  # the reference HashRouter computes logits from composed host x in Phase 2
        else:  # HOST_MATMUL, HOST_ALL
            host_logits = self._host_matmul(x)
        signpost(header="moe_gate_linear")

        # ---- Phase 2: Grouped gate ----
        signpost(header="moe_gate_grouped_gate")
        # The device gate kernels select the routing rule from n_expert_groups: with a single expert
        # group (n_expert_groups == 1, e.g. Kimi) the grouped-topk op collapses to a plain top-k.
        single_group = self.config.n_expert_groups == 1
        if mode == GateComputeMode.DEVICE:
            # The bf16 grouped gate (deepseek_grouped_gate) only supports the multi-group DeepSeek
            # shape; single-group models route through moe_grouped_topk (fp32), which handles n_groups == 1.
            ttnn_scores, ttnn_top_k_experts_indices = (
                self._device_grouped_gate_fp32(logits) if single_group else self._device_grouped_gate(logits)
            )

        elif mode == GateComputeMode.DEVICE_FP32:
            ttnn_scores, ttnn_top_k_experts_indices = self._device_grouped_gate_fp32(
                logits,
                actual_isl=actual_isl,
                padding_side=padding_side,
                padding_config=padding_config,
            )

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

        elif mode == GateComputeMode.HASH_HOST:
            if input_ids is None:
                raise ValueError("GateComputeMode.HASH_HOST forward requires input_ids for the tid2eid lookup.")
            host_x = self._compose_x_to_host(x)
            # Reference HashRouter returns (raw logits, weights * route_scale, expert indices).
            host_logits, host_scores, host_indices = self.hash_router(host_x, input_ids)
            ttnn_scores = self._host_scores_to_device(host_scores)
            ttnn_top_k_experts_indices = self._host_indices_to_device(host_indices)
            logits = self._host_logits_to_device(host_logits)

        elif mode == GateComputeMode.HASH_DEVICE:
            ttnn_scores, ttnn_top_k_experts_indices = self._device_hash_gate(
                logits,
                input_ids,
                actual_isl=actual_isl,
                padding_side=padding_side,
                padding_config=padding_config,
            )
        signpost(header="moe_gate_grouped_gate")

        return (
            ttnn_scores,
            ttnn_top_k_experts_indices,
            logits,
        )
