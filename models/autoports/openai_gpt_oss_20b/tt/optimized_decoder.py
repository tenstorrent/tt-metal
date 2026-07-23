# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device GPT-OSS 20B decoder layer.

The implementation keeps the fused decoder's packed attention and cache
semantics, but replaces the dense all-expert MoE hot path with the repo-native
routed GPT-OSS sparse-expert module.  Both public forwards are owned by this
class so an optimized test cannot silently execute a functional forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import (
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    _config_value,
    _expert_tensor,
    _state_tensor,
    _to_device_tensor,
)
from models.autoports.openai_gpt_oss_20b.tt.fused_decoder import FusedDecoder
from models.demos.deepseek_v3.utils.config_helpers import (
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)
from models.demos.gpt_oss.config import MeshConfig, Mode, ModeConfig
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.tt.experts import ExpertConfig, Experts


@dataclass
class OptimizedGPTOSSProgramConfig(GPTOSSProgramConfig):
    """Sparse-matmul program builder used by this decoder's search matrix.

    The shared builder's one-tile ``out_block_w`` is valid for the default
    subblock, but not for wider output-subblock experiments.  Keep the
    validation and adaptation local to this autoport.
    """

    def _build_matmul_config(
        self,
        cores: tuple[int, int],
        m: int,
        n: int,
        in0_block_w: int = 1,
        out_subblock_w: int = 1,
        k: int | None = None,
    ):
        core_x, core_y = cores
        core_count = core_x * core_y
        per_core_n = math.ceil(math.ceil(n / ttnn.TILE_SIZE) / core_count)
        if k is not None:
            k_tiles = math.ceil(k / ttnn.TILE_SIZE)
            if k_tiles % in0_block_w != 0:
                divisors = [width for width in range(2, in0_block_w + 1) if k_tiles % width == 0]
                in0_block_w = max(divisors) if divisors else k_tiles
        if out_subblock_w <= 0 or per_core_n % out_subblock_w != 0:
            raise ValueError(
                f"out_subblock_w={out_subblock_w} must divide per_core_N={per_core_n} " f"for cores={cores}, n={n}"
            )
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=out_subblock_w,
            per_core_M=max(ttnn.TILE_SIZE, m) // ttnn.TILE_SIZE,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )


@dataclass(frozen=True)
class OptimizationConfig:
    """Cumulative optimized policy plus focused A/B controls."""

    use_shard_advisor_attention_layouts: bool = True
    use_shard_advisor_router_layouts: bool = True
    use_shard_advisor_dense_moe_layouts: bool = False
    use_dram_sharded_attention: bool = False
    dram_attention_weight_dtype: str = "bfloat8_b"
    dram_attention_core_limit: int = 110
    attention_math_fidelity: str = "auto"
    use_sparse_experts: bool = True
    expert_weight_dtype: str = "bfloat8_b"
    expert_math_fidelity: str = "lofi"
    prefill_expert_math_fidelity: str = "hifi2"
    prefill_expert_output_dtype: str = "bfloat16"
    use_precise_sparse_prefill: bool = True
    use_manual_prefill_attention: bool = True
    use_dense_long_prefill: bool = True
    expert_gate_up_cores: tuple[int, int] = (9, 10)
    expert_down_cores: tuple[int, int] = (9, 10)
    expert_gate_up_in0_block_w: int = 45
    expert_down_in0_block_w: int = 45
    expert_gate_up_subblock_w: int = 1
    expert_down_subblock_w: int = 1
    expert_input_l1: bool = True
    use_packed_sparse_gate_up: bool = False
    kv_cache_dtype: str = "bfloat8_b"
    prefill_matmul_config: str = "auto"
    prefill_sdpa_chunk_size: int | None = None
    explicit_sdpa_program_config: bool = True
    sdpa_grid: tuple[int, int] = (8, 8)
    sdpa_k_chunk_size: int = ttnn.TILE_SIZE

    def with_changes(self, **changes) -> "OptimizationConfig":
        return replace(self, **changes)


class OptimizedDecoder(FusedDecoder):
    """Fused-attention decoder with routed active-expert execution."""

    def __init__(self, *, optimization_config: OptimizationConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.optimization_config = optimization_config or OptimizationConfig()
        self.experts = None
        self._configure_shard_advisor_candidate()
        self._configure_dram_attention_candidate()
        self._configure_attention_program_candidates()

    def _width_sharded_config(self, width: int, cores: int):
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        core_grid = ttnn.num_cores_to_corerangeset(
            cores,
            ttnn.CoreCoord(device_grid.x, device_grid.y),
            row_wise=True,
        )
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width // cores),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _configure_shard_advisor_candidate(self) -> None:
        """Materialize the exact decode configs from the fresh final_ir.mlir."""

        self.advisor_norm_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.hidden_size),
            core_grid=ttnn.CoreGrid(x=10, y=1),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.advisor_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[10, 1],
            subblock_w=3,
            block_h=1,
            block_w=9,
            inplace=False,
        )
        self.advisor_qkv_input_config = self._width_sharded_config(self.hidden_size, 45)
        self.advisor_qkv_output_config = self._width_sharded_config(
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            80,
        )
        self.advisor_residual_config = self._width_sharded_config(self.hidden_size, 90)
        self.advisor_qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=2,
            out_block_h=1,
            out_block_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.advisor_o_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        one_core = ttnn.CoreGrid(x=1, y=1)
        self.advisor_router_input_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.hidden_size),
            core_grid=one_core,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.advisor_router_output_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.num_experts),
            core_grid=one_core,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.advisor_routing_weights_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.num_experts),
            core_grid=one_core,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.advisor_router_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
            in0_block_w=90,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        advisor_90_grid = ttnn.num_cores_to_corerangeset(
            90,
            ttnn.CoreCoord(
                self.mesh_device.compute_with_storage_grid_size().x,
                self.mesh_device.compute_with_storage_grid_size().y,
            ),
            row_wise=True,
        )
        self.advisor_expert_hidden_config = ttnn.create_sharded_memory_config(
            shape=(self.num_experts * ttnn.TILE_SIZE, self.hidden_size // 90),
            core_grid=advisor_90_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.advisor_expert_gate_up_config = ttnn.create_sharded_memory_config(
            shape=(self.num_experts * ttnn.TILE_SIZE, 2 * self.intermediate_size // 90),
            core_grid=advisor_90_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _configure_dram_attention_candidate(self) -> None:
        """Build OPT-004 decode policies; weights are materialized on demand."""

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        max_cores = min(
            self.optimization_config.dram_attention_core_limit,
            device_grid.x * device_grid.y,
        )

        def policy(k: int, n: int):
            input_cores = max(get_activation_sharding_core_counts_for_dram_matmul(k, max_cores))
            output_cores = max(get_activation_sharding_core_counts_for_dram_matmul(n, max_cores))
            return (
                get_dram_sharded_matmul_config(ttnn.TILE_SIZE, k, n, input_cores, output_cores),
                self._width_sharded_config(k, input_cores),
                self._width_sharded_config(n, output_cores),
            )

        qkv_width = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.dram_qkv_program_config, self.dram_qkv_input_config, self.dram_qkv_output_config = policy(
            self.hidden_size,
            qkv_width,
        )
        self.dram_o_program_config, self.dram_o_input_config, self.dram_o_output_config = policy(
            self.num_heads * self.head_dim,
            self.hidden_size,
        )
        fidelity = self.optimization_config.attention_math_fidelity
        self.decode_compute_kernel_config = (
            None
            if fidelity == "auto"
            else ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity={
                    "lofi": ttnn.MathFidelity.LoFi,
                    "hifi2": ttnn.MathFidelity.HiFi2,
                    "hifi4": ttnn.MathFidelity.HiFi4,
                }[fidelity],
                math_approx_mode=fidelity == "lofi",
                fp32_dest_acc_en=fidelity == "hifi4",
                packer_l1_acc=True,
            )
        )
        self.decode_qkv_weight = None
        self.decode_o_weight = None

    def _configure_attention_program_candidates(self) -> None:
        """Build isolated SDPA and large-prefill A/B candidates."""

        self.decode_sdpa_program_config = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.optimization_config.sdpa_grid,
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=self.optimization_config.sdpa_k_chunk_size,
            )
            if self.optimization_config.explicit_sdpa_program_config
            else None
        )
        prefill_configs = {
            "auto": (None, None),
            "2d_8x4": (
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 4),
                    in0_block_w=5,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=1,
                    per_core_N=20,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(6, 4),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    per_core_M=1,
                    per_core_N=15,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
            ),
            "2d_10x4": (
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 4),
                    in0_block_w=5,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=1,
                    per_core_N=16,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 4),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    per_core_M=1,
                    per_core_N=9,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
            ),
        }
        try:
            self.prefill_qkv_program_config, self.prefill_o_program_config = prefill_configs[
                self.optimization_config.prefill_matmul_config
            ]
        except KeyError as error:
            raise ValueError(
                f"unknown prefill_matmul_config={self.optimization_config.prefill_matmul_config!r}; "
                f"expected one of {tuple(prefill_configs)}"
            ) from error

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = 1,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        optimization_config: OptimizationConfig | None = None,
        **kwargs,
    ) -> "OptimizedDecoder":
        if max_cache_len < EMITTED_PREFILL_SEQUENCE:
            raise ValueError(f"max_cache_len must be at least {EMITTED_PREFILL_SEQUENCE}, got {max_cache_len}")
        decoder = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            # The functional stage intentionally validates only its emitted
            # extent. Build that graph first, then extend the optimized-only
            # metadata below when the caller requests a larger cache.
            max_cache_len=min(max_cache_len, EMITTED_CACHE_LENGTH),
            **kwargs,
        )
        decoder.optimization_config = optimization_config or OptimizationConfig()
        layer_types = _config_value(hf_config, "layer_types")
        if layer_types is None or layer_idx >= len(layer_types):
            raise ValueError(f"HF config does not define layer_types[{layer_idx}]")
        layer_type = layer_types[layer_idx]
        if layer_type == "sliding_attention":
            decoder.attention_window = decoder.sliding_window
        elif layer_type == "full_attention":
            decoder.attention_window = None
        else:
            raise ValueError(f"unsupported layer_types[{layer_idx}]={layer_type!r}")

        # Optimized SDPA consumes no dense eager mask. Releasing it also lets
        # this stage support cache extents beyond the captured 128-token graph
        # without allocating an O(sequence^2) host/device mask.
        retain_manual_mask = (
            decoder.optimization_config.use_manual_prefill_attention and decoder.attention_window is None
        )
        if not retain_manual_mask:
            decoder.attention_mask.deallocate(True)
            decoder.attention_mask = None
        if max_cache_len > EMITTED_CACHE_LENGTH:
            rotary = GptOssRotaryEmbedding(hf_config)
            positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
            probe = torch.zeros((1, 1, max_cache_len, decoder.head_dim), dtype=torch.bfloat16)
            cos, sin = rotary(probe, positions)
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(1).to(torch.bfloat16)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(1).to(torch.bfloat16)
            decoder.rotary_cos.deallocate(True)
            decoder.rotary_sin.deallocate(True)
            decoder.position_indices.deallocate(True)
            decoder.rotary_cos = _to_device_tensor(cos, mesh_device)
            decoder.rotary_sin = _to_device_tensor(sin, mesh_device)
            decoder.position_indices = _to_device_tensor(
                torch.arange(max_cache_len, dtype=torch.int32),
                mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            )
            decoder.max_cache_len = max_cache_len
            decoder.prefill_rotary_views.clear()
            decoder.decode_position_views.clear()
        decoder._configure_dram_attention_candidate()
        decoder._configure_attention_program_candidates()
        decoder.advisor_norm_weights = {}
        for weight_name in ("input_norm", "post_attention_norm"):
            tiled_weight = ttnn.to_layout(getattr(decoder, weight_name), ttnn.TILE_LAYOUT)
            decoder.advisor_norm_weights[weight_name] = ttnn.to_memory_config(
                ttnn.reshape(tiled_weight, [decoder.hidden_size]),
                decoder.advisor_residual_config,
            )
        if decoder.optimization_config.use_dram_sharded_attention:
            weight_dtype = {
                "bfloat4_b": ttnn.bfloat4_b,
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat16": ttnn.bfloat16,
            }[decoder.optimization_config.dram_attention_weight_dtype]
            dram_grid = decoder.mesh_device.dram_grid_size()
            qkv_width = (decoder.num_heads + 2 * decoder.num_kv_heads) * decoder.head_dim
            decoder.decode_qkv_weight = ttnn.to_memory_config(
                ttnn.typecast(decoder.qkv_weight, weight_dtype),
                dram_sharded_weight_config(decoder.hidden_size, qkv_width, dram_grid),
            )
            decoder.decode_o_weight = ttnn.to_memory_config(
                ttnn.typecast(decoder.output_weight, weight_dtype),
                dram_sharded_weight_config(decoder.num_heads * decoder.head_dim, decoder.hidden_size, dram_grid),
            )
        if decoder.optimization_config.use_sparse_experts and decoder.batch == 1:
            decoder._load_sparse_experts(state_dict)
        return decoder

    def _load_sparse_experts(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        weight_dtype = {
            "bfloat4_b": ttnn.bfloat4_b,
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[self.optimization_config.expert_weight_dtype]
        expert_state = {
            "gate_up_proj": _expert_tensor(state_dict, self.layer_idx, "gate_up_proj"),
            "gate_up_proj_bias": _state_tensor(state_dict, self.layer_idx, "mlp.experts.gate_up_proj_bias").to(
                torch.bfloat16
            ),
            "down_proj": _expert_tensor(state_dict, self.layer_idx, "down_proj"),
            "down_proj_bias": _state_tensor(state_dict, self.layer_idx, "mlp.experts.down_proj_bias").to(
                torch.bfloat16
            ),
        }
        mesh_config = MeshConfig(
            tuple(self.mesh_device.shape),
            decode=ModeConfig(tp=1, ep=1, sp=1),
            prefill=ModeConfig(tp=1, ep=1, sp=1),
        )
        expert_config = ExpertConfig(
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            num_experts_per_tok=self.experts_per_token,
            swiglu_limit=self.swiglu_limit,
            alpha=1.703125,
        )
        program_config = OptimizedGPTOSSProgramConfig(
            decode_gate_up_cores=self.optimization_config.expert_gate_up_cores,
            decode_down_cores=self.optimization_config.expert_down_cores,
            prefill_gate_up_cores=self.optimization_config.expert_gate_up_cores,
            prefill_down_cores=self.optimization_config.expert_down_cores,
            decode_gate_up_in0_block_w=self.optimization_config.expert_gate_up_in0_block_w,
            decode_down_in0_block_w=self.optimization_config.expert_down_in0_block_w,
            prefill_gate_up_in0_block_w=self.optimization_config.expert_gate_up_in0_block_w,
            prefill_down_in0_block_w=self.optimization_config.expert_down_in0_block_w,
            decode_gate_up_subblock_w=self.optimization_config.expert_gate_up_subblock_w,
            decode_down_subblock_w=self.optimization_config.expert_down_subblock_w,
            prefill_gate_up_subblock_w=self.optimization_config.expert_gate_up_subblock_w,
            prefill_down_subblock_w=self.optimization_config.expert_down_subblock_w,
        )
        self.experts = Experts(
            mesh_device=self.mesh_device,
            config=expert_config,
            state_dict=expert_state,
            ccl_manager=None,
            mesh_config=mesh_config,
            program_config=program_config,
            weight_dtype=weight_dtype,
        )
        self.packed_gate_up_weight = None
        self.packed_gate_up_bias = None
        if self.optimization_config.use_packed_sparse_gate_up:
            column_mapper = mesh_config.column_parallel(self.mesh_device)
            self.packed_gate_up_weight = ttnn.as_tensor(
                expert_state["gate_up_proj"]
                .reshape(1, self.num_experts, self.hidden_size, 2 * self.intermediate_size)
                .contiguous(),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=column_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.packed_gate_up_bias = ttnn.as_tensor(
                expert_state["gate_up_proj_bias"].reshape(1, self.num_experts, 2 * self.intermediate_size).contiguous(),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        fidelity = self.optimization_config.expert_math_fidelity
        self.expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity={
                "lofi": ttnn.MathFidelity.LoFi,
                "hifi2": ttnn.MathFidelity.HiFi2,
            }[fidelity],
            math_approx_mode=fidelity == "lofi",
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        prefill_fidelity = self.optimization_config.prefill_expert_math_fidelity
        self.prefill_expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity={
                "lofi": ttnn.MathFidelity.LoFi,
                "hifi2": ttnn.MathFidelity.HiFi2,
            }[prefill_fidelity],
            math_approx_mode=prefill_fidelity == "lofi",
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # The sparse module owns separate gate/up/down tensors.  Releasing the
        # inherited dense copies both restores capacity and guarantees that the
        # selected runtime cannot silently execute the dense expert path.
        retain_dense_long_prefill = self.optimization_config.use_dense_long_prefill and self.attention_window is None
        released_dense_names = (
            (
                "gate_weight",
                "up_weight",
                "gate_bias",
                "up_bias",
                "gate_up_weight",
                "gate_up_bias",
                "down_weight",
                "down_bias",
            )
            if not retain_dense_long_prefill
            else ("gate_up_weight", "gate_up_bias")
        )
        for name in released_dense_names:
            tensor = getattr(self, name)
            tensor.deallocate(True)
            setattr(self, name, None)

    def _apply_fused_swiglu(self, gate, up):
        """Apply the exact clipped GPT-OSS SwiGLU with fused sigmoid dispatch."""

        gate = ttnn.clamp(gate, min=None, max=self.swiglu_limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit, output_tensor=up)
        sigmoid = ttnn.multiply(
            gate,
            1.703125,
            activations=[ttnn.UnaryOpType.SIGMOID],
        )
        gated = ttnn.multiply(gate, sigmoid, output_tensor=gate)
        if hasattr(sigmoid, "deallocate"):
            sigmoid.deallocate(True)
        up = ttnn.add(up, 1.0, output_tensor=up)
        result = ttnn.multiply(up, gated, output_tensor=up)
        if hasattr(gated, "deallocate"):
            gated.deallocate(True)
        return result

    def _sparse_decode_experts(self, hidden_states, routing_weights):
        """Run the single-device routed decode path with explicit fidelity.

        The shared GPT-OSS expert module does not expose sparse-matmul's
        ``compute_kernel_config`` even though the underlying TTNN op does.
        Keep this decoder-local copy of its batch-one/TP1/EP1 decode graph so
        LoFi versus HiFi2 can be measured under an otherwise identical policy.
        """

        experts = self.experts
        mode_config = experts.mesh_config.get_config(Mode.DECODE)
        if mode_config.tp != 1 or mode_config.ep != 1:
            raise ValueError("optimized single-device sparse decode requires TP=1 and EP=1")

        weights = experts.weights
        program_config = experts.program_config
        sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([ttnn.TILE_SIZE, ttnn.TILE_SIZE])

        def project(input_tensor, weight, config, *, input_a_sparse=False):
            return ttnn.sparse_matmul(
                input_tensor,
                weight,
                sparsity=sparsity,
                nnz=None,
                is_input_a_sparse=input_a_sparse,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=config,
                compute_kernel_config=self.expert_compute_kernel_config,
                dtype=ttnn.bfloat8_b,
            )

        if self.optimization_config.use_packed_sparse_gate_up:
            packed_width = 2 * weights.intermediate_size_per_device
            gate_up_config = program_config.get_decode_gate_up_config(
                hidden_states.shape[2],
                packed_width,
                k=hidden_states.shape[-1],
            )
            gate_up = project(hidden_states, self.packed_gate_up_weight, gate_up_config)
            hidden_states.deallocate(True)
            gate_up = ttnn.reshape(gate_up, (self.batch, self.num_experts, 1, packed_width))
            gate_up = ttnn.transpose(gate_up, 1, 2)
            gate_up = ttnn.reshape(gate_up, (self.batch, self.num_experts, packed_width))
            gate_up = ttnn.add(gate_up, self.packed_gate_up_bias, output_tensor=gate_up)
            # TTNN strided slice does not accept BFP8. Promote once after the
            # packed projection, then split gate/up in L1.
            gate_up = ttnn.typecast(gate_up, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0],
                [self.batch, self.num_experts, packed_width],
                [1, 1, 2],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 1],
                [self.batch, self.num_experts, packed_width],
                [1, 1, 2],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gate_up.deallocate(True)
        else:
            gate_up_config = program_config.get_decode_gate_up_config(
                hidden_states.shape[2],
                weights.gate_proj.shape[3],
                k=hidden_states.shape[-1],
            )
            gate = project(hidden_states, weights.gate_proj, gate_up_config)
            gate = ttnn.reshape(gate, (self.batch, self.num_experts, 1, weights.intermediate_size_per_device))
            gate = ttnn.transpose(gate, 1, 2)
            gate = ttnn.reshape(gate, (self.batch, self.num_experts, weights.intermediate_size_per_device))
            gate = ttnn.add(gate, weights.gate_proj_bias, output_tensor=gate)

            up = project(hidden_states, weights.up_proj, gate_up_config)
            hidden_states.deallocate(True)
            up = ttnn.reshape(up, (self.batch, self.num_experts, 1, weights.intermediate_size_per_device))
            up = ttnn.transpose(up, 1, 2)
            up = ttnn.reshape(up, (self.batch, self.num_experts, weights.intermediate_size_per_device))
            up = ttnn.add(up, weights.up_proj_bias, output_tensor=up)

        down_input = self._apply_fused_swiglu(gate, up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(
            down_input,
            (1, self.num_experts, 1, weights.intermediate_size_per_device),
        )
        down = project(
            down_input,
            weights.down_proj,
            program_config.get_decode_down_config(
                down_input.shape[2],
                weights.down_proj.shape[-1],
                k=down_input.shape[-1],
            ),
            input_a_sparse=True,
        )
        down_input.deallocate(True)
        sparsity.deallocate(True)

        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (self.batch, self.num_experts, self.hidden_size))
        next_states = ttnn.add(next_states, weights.down_proj_bias, output_tensor=next_states)
        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (self.batch, self.num_experts, 1))
        next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
        routing_weights.deallocate(True)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, self.batch, 1, self.hidden_size),
            (1, self.batch, ttnn.TILE_SIZE, self.hidden_size),
        )

    def _sparse_prefill_chunk(self, hidden_states, routing_weights):
        """Run one tile-aligned sparse-prefill chunk at the selected precision.

        The shared expert prefill hardcodes BFP8 outputs and leaves sparse
        matmul fidelity implicit.  That is fast, but loses the decoder's 0.99
        real-weight bar at S=128.  This local TP1/EP1 graph preserves the same
        routing and expert algebra while making BF16/HiFi2 explicit.
        """

        experts = self.experts
        weights = experts.weights
        program_config = experts.program_config
        _, batch_size, seq_len, _ = hidden_states.shape
        if batch_size != 1 or seq_len % ttnn.TILE_SIZE != 0:
            raise ValueError("precise sparse prefill requires batch one and an internally tile-aligned sequence")

        activation_dtype = {
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[self.optimization_config.prefill_expert_output_dtype]
        group_size = seq_len // ttnn.TILE_SIZE
        hidden_groups = ttnn.reshape(hidden_states, (1, group_size, ttnn.TILE_SIZE, self.hidden_size))
        sparsity = ttnn.repeat(experts.prefill_sparsity, (1, 1, group_size, 1))
        gate_up_nnz = self.num_experts * group_size
        output_tile = ttnn.Tile([ttnn.TILE_SIZE, ttnn.TILE_SIZE])

        def project(input_tensor, weight, config, *, input_a_sparse=False, projection_sparsity=sparsity, nnz=None):
            return ttnn.sparse_matmul(
                input_tensor,
                weight,
                sparsity=projection_sparsity,
                nnz=nnz,
                is_input_a_sparse=input_a_sparse,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=config,
                compute_kernel_config=self.prefill_expert_compute_kernel_config,
                dtype=activation_dtype,
            )

        gate_up_config = program_config.get_prefill_gate_up_config(
            hidden_groups.shape[2],
            weights.gate_proj.shape[3],
            k=hidden_groups.shape[-1],
        )
        gate = project(hidden_groups, weights.gate_proj, gate_up_config, nnz=gate_up_nnz)
        gate = ttnn.transpose(gate, 1, 3)
        gate = ttnn.reshape(gate, (batch_size, self.num_experts, seq_len, weights.intermediate_size_per_device))
        gate_bias = ttnn.transpose(weights.gate_proj_bias, 1, 0)
        gate = ttnn.add(gate, gate_bias, output_tensor=gate)

        up = project(hidden_groups, weights.up_proj, gate_up_config, nnz=gate_up_nnz)
        hidden_groups.deallocate(True)
        up = ttnn.transpose(up, 1, 3)
        up = ttnn.reshape(up, (batch_size, self.num_experts, seq_len, weights.intermediate_size_per_device))
        up_bias = ttnn.transpose(weights.up_proj_bias, 1, 0)
        up = ttnn.add(up, up_bias, output_tensor=up)
        down_input = self._apply_fused_swiglu(gate, up)
        down_input = ttnn.reshape(
            down_input,
            (1, self.num_experts, seq_len, weights.intermediate_size_per_device),
        )

        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (batch_size, self.num_experts, seq_len, 1))
        split_size = program_config.get_down_split_size(seq_len)
        if seq_len > split_size:
            down_inputs = ttnn.split(down_input, split_size, dim=2)
            down_input.deallocate(True)
            routing_splits = ttnn.split(routing_weights, split_size, dim=2)
            routing_weights.deallocate(True)
        else:
            down_inputs = [down_input]
            routing_splits = [routing_weights]

        reduced_chunks = []
        for down_split, routing_split in zip(down_inputs, routing_splits):
            split_len = down_split.shape[2]
            down = project(
                down_split,
                weights.down_proj,
                program_config.get_prefill_down_config(
                    down_split.shape[2],
                    weights.down_proj.shape[-1],
                    k=down_split.shape[-1],
                ),
                input_a_sparse=True,
                projection_sparsity=experts.prefill_sparsity,
                nnz=self.num_experts,
            )
            down_split.deallocate(True)
            next_states = ttnn.reshape(down, (batch_size, self.num_experts, split_len, self.hidden_size))
            down_bias = ttnn.transpose(weights.down_proj_bias, 1, 0)
            next_states = ttnn.add(next_states, down_bias, output_tensor=next_states)
            next_states = ttnn.multiply(next_states, routing_split, output_tensor=next_states)
            routing_split.deallocate(True)
            reduced = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
            down.deallocate(True)
            reduced_chunks.append(reduced)

        output = reduced_chunks[0]
        for next_chunk in reduced_chunks[1:]:
            concatenated = ttnn.concat([output, next_chunk], dim=2)
            output.deallocate(True)
            next_chunk.deallocate(True)
            output = concatenated
        return ttnn.reshape(
            output,
            (1, batch_size, seq_len, self.hidden_size),
            (1, batch_size, max(ttnn.TILE_SIZE, seq_len), self.hidden_size),
        )

    def _sparse_prefill_experts(self, hidden_states, routing_weights):
        """Process arbitrarily long padded prefill through bounded chunks."""

        chunk_size = self.experts.program_config.sequence_chunk_size
        if hidden_states.shape[2] <= chunk_size:
            return self._sparse_prefill_chunk(hidden_states, routing_weights)
        hidden_chunks = ttnn.split(hidden_states, chunk_size, dim=2)
        routing_chunks = ttnn.split(routing_weights, chunk_size, dim=0)
        hidden_states.deallocate(True)
        routing_weights.deallocate(True)
        output = None
        for hidden_chunk, routing_chunk in zip(hidden_chunks, routing_chunks):
            next_chunk = self._sparse_prefill_chunk(hidden_chunk, routing_chunk)
            if output is None:
                output = next_chunk
            else:
                concatenated = ttnn.concat([output, next_chunk], dim=2)
                output.deallocate(True)
                next_chunk.deallocate(True)
                output = concatenated
        return output

    def _dense_reference_experts(self, hidden_states, seq_len: int):
        """Run the decoder-local BF16 all-expert reference graph.

        The advisor's dense attention+MLP control and the selected
        full-attention S=128 accuracy path are both optimized-owned.  Decode
        uses one packed gate/up projection; prefill uses two narrower
        projections to avoid the packed output's strided extraction.
        """

        tokens = self.batch * seq_len
        token_states = ttnn.reshape(hidden_states, [tokens, self.hidden_size])
        routing_weights = self._route(hidden_states, seq_len)
        expert_input = ttnn.reshape(token_states, [1, tokens, self.hidden_size])
        expert_input = ttnn.repeat(
            expert_input,
            ttnn.Shape([self.num_experts, 1, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if seq_len == 1:
            gate_up = ttnn.linear(
                expert_input,
                self.gate_up_weight,
                bias=self.gate_up_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 1],
                [self.num_experts, tokens, 2 * self.intermediate_size],
                [1, 1, 2],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0],
                [self.num_experts, tokens, 2 * self.intermediate_size],
                [1, 1, 2],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = ttnn.linear(
                expert_input,
                self.gate_weight,
                bias=self.gate_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            up = ttnn.linear(
                expert_input,
                self.up_weight,
                bias=self.up_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
        if hasattr(expert_input, "deallocate"):
            expert_input.deallocate(True)
        activated = self._apply_fused_swiglu(gate, up)
        expert_output = ttnn.linear(
            activated,
            self.down_weight,
            bias=self.down_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        if hasattr(activated, "deallocate"):
            activated.deallocate(True)
        routing_weights = ttnn.permute(routing_weights, (1, 0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing_weights = ttnn.reshape(routing_weights, [self.num_experts, tokens, 1])
        expert_output = ttnn.multiply(
            expert_output,
            routing_weights,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=expert_output,
        )
        if hasattr(routing_weights, "deallocate"):
            routing_weights.deallocate(True)
        reduced = ttnn.sum(expert_output, [0], False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(reduced, [1, self.batch, seq_len, self.hidden_size])

    def _route(self, hidden_states, seq_len: int):
        tokens = self.batch * seq_len
        advisor_layouts = self.optimization_config.use_shard_advisor_router_layouts and seq_len == 1
        advisor_dense = self.optimization_config.use_shard_advisor_dense_moe_layouts and seq_len == 1
        advisor_l1_norm = advisor_dense or (
            self.experts is not None and self.optimization_config.expert_input_l1 and seq_len == 1
        )
        token_states = ttnn.reshape(hidden_states, [tokens, self.hidden_size])
        router_input = ttnn.typecast(
            token_states,
            ttnn.float32,
            memory_config=self.advisor_norm_memory_config if advisor_l1_norm else ttnn.DRAM_MEMORY_CONFIG,
        )
        if advisor_layouts:
            router_input = ttnn.to_memory_config(router_input, self.advisor_router_input_config)
        router_logits = ttnn.linear(
            router_input,
            self.router_weight,
            bias=self.router_bias,
            dtype=ttnn.float32,
            memory_config=self.advisor_router_output_config if advisor_layouts else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.advisor_router_program_config if advisor_layouts else None,
            compute_kernel_config=self.compute_kernel_config,
        )
        router_logits = ttnn.typecast(
            router_logits,
            ttnn.bfloat16,
            memory_config=self.advisor_router_output_config if advisor_layouts else ttnn.DRAM_MEMORY_CONFIG,
        )
        if advisor_layouts:
            router_logits = ttnn.to_memory_config(router_logits, ttnn.L1_MEMORY_CONFIG)
        top_values, top_indices = ttnn.topk(
            router_logits,
            self.experts_per_token,
            1,
            True,
            True,
            memory_config=ttnn.L1_MEMORY_CONFIG if advisor_layouts else ttnn.DRAM_MEMORY_CONFIG,
        )
        top_weights = ttnn.softmax(
            top_values,
            1,
            memory_config=self.advisor_routing_weights_config if advisor_layouts else ttnn.DRAM_MEMORY_CONFIG,
            numeric_stable=True,
        )
        if advisor_layouts:
            top_weights = ttnn.to_memory_config(top_weights, ttnn.L1_MEMORY_CONFIG)
        routing_weights = ttnn.scatter(
            input=ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_weights,
            memory_config=self.advisor_routing_weights_config if advisor_layouts else ttnn.DRAM_MEMORY_CONFIG,
        )
        if advisor_layouts:
            # final_ir changes the scatter result back to interleaved at the
            # following reshape.  Make that boundary explicit because the
            # shared sparse-expert module starts with TILE -> ROW_MAJOR.
            routing_weights = ttnn.to_memory_config(routing_weights, ttnn.L1_MEMORY_CONFIG)
        return routing_weights

    def _optimized_moe_forward(self, hidden_states, seq_len: int):
        residual = hidden_states
        advisor_dense = (
            self.experts is None and self.optimization_config.use_shard_advisor_dense_moe_layouts and seq_len == 1
        )
        if advisor_dense:
            return self._advisor_dense_moe_forward(residual)
        sparse_l1_chain = self.experts is not None and self.optimization_config.expert_input_l1 and seq_len == 1
        norm_input = (
            ttnn.to_memory_config(hidden_states, self.advisor_norm_memory_config) if sparse_l1_chain else hidden_states
        )
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=self.rms_norm_eps,
            weight=self.advisor_norm_weights["post_attention_norm"] if sparse_l1_chain else self.post_attention_norm,
            memory_config=self.advisor_norm_memory_config if sparse_l1_chain else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.advisor_norm_program_config if sparse_l1_chain else None,
            compute_kernel_config=self.compute_kernel_config,
        )

        # The shared sparse expert module currently serves batch one. Preserve
        # larger-batch compatibility with this class's dense graph; the
        # primary measured batch-one decode path can only use the sparse
        # module because its dense weights were released above.
        if self.experts is None:
            expert_output = self._dense_reference_experts(normalized, seq_len)
            return ttnn.add(
                residual,
                expert_output,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tensor=expert_output,
            )

        if (
            self.optimization_config.use_dense_long_prefill
            and self.attention_window is None
            and seq_len >= EMITTED_CACHE_LENGTH
        ):
            expert_output = self._dense_reference_experts(normalized, seq_len)
            return ttnn.add(
                residual,
                expert_output,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tensor=expert_output,
            )

        if seq_len == 1:
            routing_weights = self._route(normalized, seq_len)
            expert_input = normalized
            if sparse_l1_chain:
                expert_input = ttnn.to_memory_config(expert_input, ttnn.L1_MEMORY_CONFIG)
                routing_weights = ttnn.to_memory_config(routing_weights, ttnn.L1_MEMORY_CONFIG)
            expert_output = self._sparse_decode_experts(expert_input, routing_weights)
        else:
            routing_weights = self._route(normalized, seq_len)
            padded_len = math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            expert_input = normalized
            if padded_len != seq_len:
                expert_input = ttnn.pad(
                    expert_input,
                    [(0, 0), (0, 0), (0, padded_len - seq_len), (0, 0)],
                    value=0.0,
                )
                routing_weights = ttnn.pad(
                    routing_weights,
                    [(0, padded_len - seq_len), (0, 0)],
                    value=0.0,
                )
            if self.optimization_config.use_precise_sparse_prefill:
                expert_output = self._sparse_prefill_experts(expert_input, routing_weights)
            else:
                expert_output = self.experts(
                    expert_input,
                    topk_expert_weights=routing_weights,
                    is_decode=False,
                )
            if padded_len != seq_len:
                expert_output = ttnn.slice(
                    expert_output,
                    [0, 0, 0, 0],
                    [1, self.batch, seq_len, self.hidden_size],
                    [1, 1, 1, 1],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        if sparse_l1_chain:
            residual = ttnn.to_memory_config(residual, self.advisor_residual_config)
            expert_output = ttnn.to_memory_config(expert_output, self.advisor_residual_config)
            output = ttnn.add(
                residual,
                expert_output,
                dtype=ttnn.bfloat16,
                memory_config=self.advisor_residual_config,
            )
            return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(residual, expert_output, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _advisor_dense_moe_forward(self, hidden_states):
        """Evaluate the complete dense-MoE L1 layout chain from final_ir.mlir."""

        norm_input = ttnn.to_memory_config(hidden_states, self.advisor_norm_memory_config)
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=self.rms_norm_eps,
            weight=self.advisor_norm_weights["post_attention_norm"],
            memory_config=self.advisor_norm_memory_config,
            program_config=self.advisor_norm_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        routing_weights = self._route(normalized, 1)
        flat = ttnn.reshape(normalized, [1, self.hidden_size])
        expert_input = ttnn.reshape(flat, [1, 1, self.hidden_size])
        expert_input = ttnn.repeat(expert_input, ttnn.Shape([self.num_experts, 1, 1]))
        expert_input = ttnn.to_memory_config(expert_input, self.advisor_expert_hidden_config)
        expert_input = ttnn.to_memory_config(expert_input, ttnn.L1_MEMORY_CONFIG)
        gate_up = ttnn.linear(
            expert_input,
            self.gate_up_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        gate_up = ttnn.to_memory_config(gate_up, self.advisor_expert_gate_up_config)
        gate_up = ttnn.add(
            gate_up,
            self.gate_up_bias,
            memory_config=self.advisor_expert_gate_up_config,
        )
        gate_up = ttnn.to_memory_config(gate_up, ttnn.L1_MEMORY_CONFIG)
        up = ttnn.slice(
            gate_up,
            [0, 0, 1],
            [self.num_experts, 1, 2 * self.intermediate_size],
            [1, 1, 2],
        )
        gate = ttnn.slice(
            gate_up,
            [0, 0, 0],
            [self.num_experts, 1, 2 * self.intermediate_size],
            [1, 1, 2],
        )
        up = ttnn.clamp(
            up,
            -self.swiglu_limit,
            self.swiglu_limit,
            memory_config=self.advisor_expert_hidden_config,
        )
        gate = ttnn.clamp(
            gate,
            float("-inf"),
            self.swiglu_limit,
            memory_config=self.advisor_expert_hidden_config,
        )
        sigmoid = ttnn.multiply(
            gate,
            1.703125,
            dtype=ttnn.bfloat16,
            memory_config=self.advisor_expert_hidden_config,
            activations=[ttnn.UnaryOpType.SIGMOID],
        )
        gated = ttnn.multiply(
            gate,
            sigmoid,
            dtype=ttnn.bfloat16,
            memory_config=self.advisor_expert_hidden_config,
        )
        activated = ttnn.multiply(
            ttnn.add(up, 1.0, memory_config=self.advisor_expert_hidden_config),
            gated,
            dtype=ttnn.bfloat16,
            memory_config=self.advisor_expert_hidden_config,
        )
        activated = ttnn.to_memory_config(activated, ttnn.L1_MEMORY_CONFIG)
        expert_output = ttnn.linear(
            activated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        expert_output = ttnn.to_memory_config(expert_output, self.advisor_expert_hidden_config)
        expert_output = ttnn.add(
            expert_output,
            self.down_bias,
            memory_config=self.advisor_expert_hidden_config,
        )
        expert_output = ttnn.to_memory_config(expert_output, ttnn.L1_MEMORY_CONFIG)
        routing_weights = ttnn.reshape(routing_weights, [self.num_experts, 1, 1])
        routing_weights = ttnn.to_memory_config(routing_weights, ttnn.L1_MEMORY_CONFIG)
        expert_output = ttnn.multiply(
            expert_output,
            routing_weights,
            memory_config=self.advisor_expert_hidden_config,
        )
        expert_output = ttnn.to_memory_config(expert_output, ttnn.L1_MEMORY_CONFIG)
        expert_output = ttnn.sum(
            expert_output,
            [0],
            False,
            memory_config=self.advisor_residual_config,
        )
        expert_output = ttnn.to_memory_config(expert_output, ttnn.L1_MEMORY_CONFIG)
        expert_output = ttnn.reshape(expert_output, [1, self.batch, 1, self.hidden_size])
        residual = ttnn.to_memory_config(hidden_states, self.advisor_residual_config)
        output = ttnn.add(residual, expert_output, memory_config=self.advisor_residual_config)
        return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)

    def _prefill_attention(self, hidden_states, key_cache, value_cache, seq_len: int):
        residual = hidden_states
        qkv_program_config = self.prefill_qkv_program_config if seq_len == 128 else None
        o_program_config = self.prefill_o_program_config if seq_len == 128 else None
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            bias=self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qkv_program_config,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos, sin = self._get_prefill_rotary_views(seq_len)
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cache_key = key
        cache_value = value
        if self.optimization_config.kv_cache_dtype == "bfloat8_b":
            # The current-token BF16 tensors remain inputs to prefill SDPA;
            # quantize only the values written to persistent cache.
            cache_key = ttnn.typecast(key, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cache_value = ttnn.typecast(value, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.batch == 1:
            ttnn.fill_cache(key_cache, cache_key, 0)
            ttnn.fill_cache(value_cache, cache_value, 0)
        else:
            for user_id in range(self.batch):
                user_key = ttnn.slice(
                    cache_key,
                    [user_id, 0, 0, 0],
                    [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                    [1, 1, 1, 1],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                user_value = ttnn.slice(
                    cache_value,
                    [user_id, 0, 0, 0],
                    [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                    [1, 1, 1, 1],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.fill_cache(key_cache, user_key, user_id)
                ttnn.fill_cache(value_cache, user_value, user_id)
        use_manual_accuracy_path = (
            self.optimization_config.use_manual_prefill_attention
            and self.attention_window is None
            and seq_len == EMITTED_CACHE_LENGTH
        )
        if use_manual_accuracy_path:
            if self.attention_mask is None:
                raise ValueError("manual prefill attention requires the retained S=128 causal mask")
            repeated_key = ttnn.repeat_interleave(key, self.num_heads // self.num_kv_heads, 1)
            transposed_key = ttnn.permute(repeated_key, (0, 1, 3, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scores = ttnn.matmul(query, transposed_key, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scores = ttnn.multiply(scores, self.scale, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mask = ttnn.slice(
                self.attention_mask,
                [0, 0, 0, 0],
                [1, 1, seq_len, seq_len],
                [1, 1, 1, 1],
            )
            mask = ttnn.typecast(mask, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scores = ttnn.add(scores, mask, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sinks = ttnn.slice(
                self.attention_sinks,
                [0, 0, 0, 0],
                [self.batch, self.num_heads, seq_len, 1],
                [1, 1, 1, 1],
            )
            sinks = ttnn.typecast(sinks, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            probabilities = ttnn.softmax(
                ttnn.concat([scores, sinks], 3, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                numeric_stable=True,
            )
            probabilities = ttnn.slice(
                probabilities,
                [0, 0, 0, 0],
                [self.batch, self.num_heads, seq_len, seq_len],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            repeated_value = ttnn.repeat_interleave(value, self.num_heads // self.num_kv_heads, 1)
            attention = ttnn.matmul(
                probabilities,
                repeated_value,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            attention = ttnn.typecast(attention, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            attention = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                is_causal=True,
                scale=self.scale,
                sliding_window_size=self.attention_window,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=(
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                        q_chunk_size=self.optimization_config.prefill_sdpa_chunk_size,
                        k_chunk_size=self.optimization_config.prefill_sdpa_chunk_size,
                        exp_approx_mode=False,
                    )
                    if self.optimization_config.prefill_sdpa_chunk_size is not None
                    else None
                ),
                attention_sink=self.prefill_sdpa_sink,
            )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.num_heads * self.head_dim])
        attention = ttnn.linear(
            attention,
            self.output_weight,
            bias=self.output_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=o_program_config,
        )
        return ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=attention,
        )

    def _decode_attention(self, hidden_states, key_cache, value_cache, current_pos: int):
        dram_attention = self.optimization_config.use_dram_sharded_attention
        # The captured advisor graph is the emitted batch-one decode graph.
        # Its one-row norm/residual shard is invalid once decode height grows
        # with batch, so retain the fused interleaved attention contract there.
        advisor_layouts = (
            self.optimization_config.use_shard_advisor_attention_layouts and not dram_attention and self.batch == 1
        )
        residual = hidden_states
        norm_input = (
            ttnn.to_memory_config(hidden_states, self.advisor_norm_memory_config) if advisor_layouts else hidden_states
        )
        normed = ttnn.rms_norm(
            norm_input,
            epsilon=self.rms_norm_eps,
            weight=self.advisor_norm_weights["input_norm"] if advisor_layouts else self.input_norm,
            memory_config=self.advisor_norm_memory_config if advisor_layouts else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.advisor_norm_program_config if advisor_layouts else None,
            compute_kernel_config=self.compute_kernel_config,
        )
        if advisor_layouts:
            normed = ttnn.to_memory_config(normed, self.advisor_qkv_input_config)
        elif dram_attention:
            normed = ttnn.to_memory_config(normed, self.dram_qkv_input_config)
        fused_qkv = ttnn.linear(
            normed,
            self.decode_qkv_weight if dram_attention else self.qkv_weight,
            # DRAM-sharded matmul cannot fuse an interleaved bias into its
            # width-sharded output; follow Attention1D and add it separately.
            bias=None if dram_attention else self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.dram_qkv_output_config
                if dram_attention
                else self.advisor_qkv_output_config
                if advisor_layouts
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            program_config=(
                self.dram_qkv_program_config
                if dram_attention
                else self.advisor_qkv_program_config
                if advisor_layouts
                else None
            ),
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        if dram_attention:
            fused_qkv = ttnn.add(fused_qkv, self.qkv_bias, memory_config=self.dram_qkv_output_config)
            fused_qkv = ttnn.to_memory_config(fused_qkv, ttnn.L1_MEMORY_CONFIG)
        elif advisor_layouts:
            # The head-split kernel uses global CBs and consumes interleaved L1.
            fused_qkv = ttnn.to_memory_config(fused_qkv, ttnn.L1_MEMORY_CONFIG)
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [1, 1, self.batch, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=self.decode_heads_mem_config,
        )
        cos, sin, update_indices = self._get_decode_position_views(current_pos)
        query = ttnn.experimental.rotary_embedding(
            query,
            cos,
            sin,
            0,
            memory_config=self.decode_heads_mem_config,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            0,
            memory_config=self.decode_heads_mem_config,
        )
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=True,
            attn_mask=None,
            cur_pos_tensor=update_indices,
            attention_sink=self.decode_attention_sinks,
            scale=self.scale,
            sliding_window_size=self.attention_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_sdpa_program_config,
        )
        attention = ttnn.to_memory_config(attention, self.decode_heads_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=self.num_heads)
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.num_heads * self.head_dim],
            [1, 1, 1, 1],
        )
        attention = ttnn.reshape(attention, [1, self.batch, 1, self.num_heads * self.head_dim])
        if advisor_layouts:
            attention = ttnn.to_memory_config(attention, ttnn.L1_MEMORY_CONFIG)
        elif dram_attention:
            attention = ttnn.to_memory_config(attention, self.dram_o_input_config)
        attention = ttnn.linear(
            attention,
            self.decode_o_weight if dram_attention else self.output_weight,
            bias=None if dram_attention else self.output_bias,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.dram_o_output_config
                if dram_attention
                else self.advisor_residual_config
                if advisor_layouts
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            program_config=(
                self.dram_o_program_config
                if dram_attention
                else self.advisor_o_program_config
                if advisor_layouts
                else None
            ),
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        if dram_attention:
            attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
            attention = ttnn.add(attention, self.output_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if advisor_layouts:
            residual = ttnn.to_memory_config(residual, self.advisor_residual_config)
            attention = ttnn.add(
                residual,
                attention,
                dtype=ttnn.bfloat16,
                memory_config=self.advisor_residual_config,
            )
            # Keep the advisor's residual layout through post-attention norm
            # for the selected sparse L1 chain.  The opt-out remains useful
            # for the measured DRAM-boundary comparison.
            if self.optimization_config.expert_input_l1:
                return attention
            return ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=attention,
        )

    def create_kv_cache(self):
        """Allocate a cache matching the selected persistent-cache policy."""

        cache_dtype = {
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[self.optimization_config.kv_cache_dtype]
        shape = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
        return (
            ttnn.zeros(
                shape,
                dtype=cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.zeros(
                shape,
                dtype=cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )

    def prefill_forward(self, hidden_states, key_cache, value_cache):
        """Run optimized prefill for any valid logical sequence length."""

        seq_len = self._validate_hidden_states(hidden_states)
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1; use decode_forward for one token")
        self._validate_caches(key_cache, value_cache)
        hidden_states = self._prefill_attention(hidden_states, key_cache, value_cache, seq_len)
        return self._optimized_moe_forward(hidden_states, seq_len)

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int):
        """Run optimized paged decode for one logical token."""

        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")
        hidden_states = self._decode_attention(hidden_states, key_cache, value_cache, current_pos)
        return self._optimized_moe_forward(hidden_states, 1)


__all__ = ["OptimizationConfig", "OptimizedDecoder", "OptimizedGPTOSSProgramConfig"]
