# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized GPT-OSS-20B decoder layer.

The public contract is inherited from :class:`FunctionalDecoder`, while every
optimized runtime component is overridden here.  The selected path combines
the dedicated decode concat-heads operation, device-indexed RoPE, routed sparse
experts, and a boundary-correct sliding-attention mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    SUPPORTED_CONTEXT,
    FunctionalDecoder,
    _dense_expert_weight,
    _require_tensor,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.tt.experts import ExpertConfig, Experts


@dataclass
class OptimizedGPTOSSProgramConfig(GPTOSSProgramConfig):
    """Stage-local sparse-matmul builder with valid output blocking.

    The shared GPT-OSS builder always emits ``out_block_w=1``.  That is valid
    for its default one-tile subblocks, but not for the wider subblocks swept
    by this stage.  Keeping the correction local avoids changing another
    model's runtime while still making the experiment well-formed.
    """

    def _build_matmul_config(self, cores, m, n, in0_block_w=1, out_subblock_w=1, k=None):
        core_x, core_y = cores
        num_cores = core_x * core_y
        per_core_n = math.ceil(math.ceil(n / ttnn.TILE_SIZE) / num_cores)
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
    """Cumulative optimized-decoder policy and experiment controls."""

    use_decode_concat_heads: bool = True
    use_shard_advisor_layouts: bool = True
    use_shard_advisor_attention_layouts: bool = True
    use_shard_advisor_moe_layouts: bool = False
    use_sparse_experts: bool = True
    use_explicit_sliding_mask: bool = True
    explicit_sliding_sdpa_program_config: bool = True
    use_dram_sharded_attention: bool = False
    dram_attention_query_dram: bool = False
    dram_attention_weight_dtype: str = "bfloat4_b"
    expert_weight_dtype: str = "bfloat8_b"
    expert_gate_up_cores: tuple[int, int] = (9, 10)
    expert_down_cores: tuple[int, int] = (9, 10)
    expert_gate_up_in0_block_w: int = 45
    expert_down_in0_block_w: int = 45
    expert_gate_up_subblock_w: int = 1
    expert_down_subblock_w: int = 1
    expert_input_l1: bool = False
    kv_cache_dtype: str = "bfloat16"
    math_fidelity: str = "hifi4"
    prefill_matmul_config: str = "auto"
    explicit_sdpa_program_config: bool = False
    sdpa_grid: tuple[int, int] = (8, 8)
    sdpa_k_chunk_size: int = ttnn.TILE_SIZE

    def with_changes(self, **changes) -> "OptimizationConfig":
        return replace(self, **changes)


class OptimizedDecoder(FunctionalDecoder):
    """Optimized runtime with the functional decoder's public/cache contract."""

    def __init__(self, *, optimization_config: OptimizationConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self._configure_optimization(optimization_config or OptimizationConfig())

    def _configure_optimization(self, optimization_config):
        self.optimization_config = optimization_config
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        batch_grid = ttnn.num_cores_to_corerangeset(
            self.batch,
            ttnn.CoreCoord(device_grid.x, device_grid.y),
            row_wise=True,
        )
        padded_heads = ((self.num_heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self.decode_heads_memory_config = ttnn.create_sharded_memory_config(
            shape=(padded_heads, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_rope_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if not hasattr(self, "decode_cos_matrix"):
            # Decode gathers the current row on device so changing the position
            # tensor remains correct across cached programs and trace replay.
            self.decode_cos_matrix = ttnn.to_layout(
                ttnn.reshape(self.cos_cache, [self.max_cache_len, self.head_dim]),
                ttnn.ROW_MAJOR_LAYOUT,
            )
            self.decode_sin_matrix = ttnn.to_layout(
                ttnn.reshape(self.sin_cache, [self.max_cache_len, self.head_dim]),
                ttnn.ROW_MAJOR_LAYOUT,
            )
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
        if (
            self.sliding_window is not None
            and self.optimization_config.use_explicit_sliding_mask
            and not hasattr(self, "decode_key_positions")
        ):
            self.decode_key_positions = ttnn.reshape(
                ttnn.arange(
                    0,
                    self.max_cache_len,
                    1,
                    device=self.mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                [1, 1, 1, self.max_cache_len],
            )
            self.sliding_sdpa_program_config = (
                ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.optimization_config.sdpa_grid,
                    exp_approx_mode=False,
                    # Explicit-mask decode treats the query sequence as the
                    # padded head dimension.  Match the supported non-causal
                    # SDPA-decode geometry exercised by the kernel unit tests.
                    q_chunk_size=padded_heads,
                    k_chunk_size=self.optimization_config.sdpa_k_chunk_size,
                )
                if self.optimization_config.explicit_sliding_sdpa_program_config
                else None
            )
            self.sliding_compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        fidelity = {
            "lofi": ttnn.MathFidelity.LoFi,
            "hifi2": ttnn.MathFidelity.HiFi2,
            "hifi4": ttnn.MathFidelity.HiFi4,
        }[self.optimization_config.math_fidelity]
        self.decode_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=fidelity,
            math_approx_mode=self.optimization_config.math_fidelity == "lofi",
            fp32_dest_acc_en=self.optimization_config.math_fidelity == "hifi4",
            packer_l1_acc=True,
        )
        max_cores = device_grid.x * device_grid.y

        def dram_matmul_policy(k, n):
            input_cores = max(get_activation_sharding_core_counts_for_dram_matmul(k, max_cores))
            output_cores = max(get_activation_sharding_core_counts_for_dram_matmul(n, max_cores))

            def l1_width_config(width, cores):
                grid = ttnn.num_cores_to_corerangeset(
                    cores,
                    ttnn.CoreCoord(device_grid.x, device_grid.y),
                    row_wise=True,
                )
                return ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, width // cores),
                    core_grid=grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

            return (
                get_dram_sharded_matmul_config(ttnn.TILE_SIZE, k, n, input_cores, output_cores),
                l1_width_config(k, input_cores),
                l1_width_config(n, output_cores),
            )

        self.dram_qkv_program_config, self.dram_qkv_input_config, self.dram_qkv_output_config = dram_matmul_policy(
            self.hidden_size, self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
        )
        self.dram_o_program_config, self.dram_o_input_config, self.dram_o_output_config = dram_matmul_policy(
            self.num_heads * self.head_dim, self.hidden_size
        )

        # Exact decode matmul configs emitted by shard-advise.  They are an
        # experiment toggle because the final sparse-expert policy is tuned
        # independently from the advisor's dense MoE graph.
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

        def advisor_width_config(width, cores):
            grid = ttnn.num_cores_to_corerangeset(
                cores,
                ttnn.CoreCoord(device_grid.x, device_grid.y),
                row_wise=True,
            )
            return ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, width // cores),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        # Full decode layout/revert chain emitted in shard_advise/final_ir.mlir.
        # These configurations intentionally remain behind the advisor A/B
        # switch: the advisor optimized the captured dense graph, while the
        # selected runtime uses sparse experts.
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
        self.advisor_qkv_input_config = advisor_width_config(self.hidden_size, 45)
        self.advisor_qkv_output_config = advisor_width_config(
            self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim,
            80,
        )
        self.advisor_residual_config = advisor_width_config(self.hidden_size, 90)
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
        advisor_90_grid = ttnn.num_cores_to_corerangeset(
            90,
            ttnn.CoreCoord(device_grid.x, device_grid.y),
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

    @classmethod
    def from_state_dict(cls, state_dict, *, optimization_config: OptimizationConfig | None = None, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder._configure_optimization(optimization_config or OptimizationConfig())
        decoder.advisor_norm_weights = {}
        if decoder.optimization_config.use_shard_advisor_layouts:
            # final_ir.mlir reshapes both RMSNorm weights to one logical row
            # and width-shards them over 90 cores before either sharded norm.
            # Keep these as dedicated device tensors so the advisor A/B does
            # not rely on an implicit weight-layout conversion.
            for weight_name in ("input_norm", "post_attention_norm"):
                tiled_weight = ttnn.to_layout(decoder.weights[weight_name], ttnn.TILE_LAYOUT)
                decoder.advisor_norm_weights[weight_name] = ttnn.to_memory_config(
                    ttnn.reshape(tiled_weight, [decoder.hidden_size]),
                    decoder.advisor_residual_config,
                )
        if decoder.optimization_config.use_dram_sharded_attention:
            decode_dtype = {
                "bfloat4_b": ttnn.bfloat4_b,
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat16": ttnn.bfloat16,
            }[decoder.optimization_config.dram_attention_weight_dtype]
            dram_grid = decoder.mesh_device.dram_grid_size()
            qkv_n = decoder.num_heads * decoder.head_dim + 2 * decoder.num_kv_heads * decoder.head_dim
            decoder.decode_qkv_weight = ttnn.to_memory_config(
                ttnn.typecast(decoder.weights["qkv_weight"], decode_dtype),
                dram_sharded_weight_config(decoder.hidden_size, qkv_n, dram_grid),
            )
            decoder.decode_o_weight = ttnn.to_memory_config(
                ttnn.typecast(decoder.weights["o_weight"], decode_dtype),
                dram_sharded_weight_config(decoder.num_heads * decoder.head_dim, decoder.hidden_size, dram_grid),
            )
        if decoder.optimization_config.use_sparse_experts:
            weight_dtype = {
                "bfloat4_b": ttnn.bfloat4_b,
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat16": ttnn.bfloat16,
            }[decoder.optimization_config.expert_weight_dtype]
            layer_idx = decoder.layer_idx
            expert_state = {
                "gate_up_proj": _dense_expert_weight(state_dict, layer_idx, "gate_up_proj"),
                "gate_up_proj_bias": _require_tensor(state_dict, layer_idx, "mlp.experts.gate_up_proj_bias"),
                "down_proj": _dense_expert_weight(state_dict, layer_idx, "down_proj"),
                "down_proj_bias": _require_tensor(state_dict, layer_idx, "mlp.experts.down_proj_bias"),
            }
            mesh_config = MeshConfig(
                tuple(decoder.mesh_device.shape),
                decode=ModeConfig(tp=1, ep=1, sp=1),
                prefill=ModeConfig(tp=1, ep=1, sp=1),
            )
            expert_config = ExpertConfig(
                intermediate_size=decoder.intermediate_size,
                num_experts=decoder.num_experts,
                hidden_size=decoder.hidden_size,
                num_experts_per_tok=decoder.top_k,
                swiglu_limit=float(getattr(decoder.hf_config, "swiglu_limit", 7.0)),
                alpha=1.703125,
            )
            program_config = OptimizedGPTOSSProgramConfig(
                decode_gate_up_cores=decoder.optimization_config.expert_gate_up_cores,
                decode_down_cores=decoder.optimization_config.expert_down_cores,
                prefill_gate_up_cores=decoder.optimization_config.expert_gate_up_cores,
                prefill_down_cores=decoder.optimization_config.expert_down_cores,
                decode_gate_up_in0_block_w=decoder.optimization_config.expert_gate_up_in0_block_w,
                decode_down_in0_block_w=decoder.optimization_config.expert_down_in0_block_w,
                prefill_gate_up_in0_block_w=decoder.optimization_config.expert_gate_up_in0_block_w,
                prefill_down_in0_block_w=decoder.optimization_config.expert_down_in0_block_w,
                decode_gate_up_subblock_w=decoder.optimization_config.expert_gate_up_subblock_w,
                decode_down_subblock_w=decoder.optimization_config.expert_down_subblock_w,
                prefill_gate_up_subblock_w=decoder.optimization_config.expert_gate_up_subblock_w,
                prefill_down_subblock_w=decoder.optimization_config.expert_down_subblock_w,
            )
            decoder.experts = Experts(
                mesh_device=decoder.mesh_device,
                config=expert_config,
                state_dict=expert_state,
                ccl_manager=None,
                mesh_config=mesh_config,
                program_config=program_config,
                weight_dtype=weight_dtype,
            )
            # The sparse path owns separate gate/up/down tensors.  Release the
            # functional all-expert copies so the optimized context capacity is
            # never lower than the advertised dense baseline.
            for name in ("gate_up_weight", "gate_up_bias", "down_weight", "down_bias"):
                decoder.weights.pop(name).deallocate(True)
        return decoder

    def _prefill_attention(self, hidden_states, key_cache, value_cache, seq_len):
        qkv_program_config = self.prefill_qkv_program_config if seq_len == 128 else None
        o_program_config = self.prefill_o_program_config if seq_len == 128 else None
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["input_norm"],
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = ttnn.linear(
            normalized,
            self.weights["qkv_weight"],
            bias=self.weights["qkv_bias"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qkv_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = ttnn.reshape(fused, [self.batch, seq_len, -1])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.cos_cache, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        sin = ttnn.slice(self.sin_cache, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(query, [0, 0, 0, 0], [1, self.num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [1, self.num_kv_heads, seq_len, self.head_dim])
        cache_key = key
        cache_value = value
        if self.optimization_config.kv_cache_dtype == "bfloat8_b":
            # Cache update operations require source and destination dtypes to
            # match.  Keep the BF16 K/V tensors for the current prefill SDPA,
            # and quantize only the operands written to persistent cache.
            cache_key = ttnn.typecast(key, ttnn.bfloat8_b)
            cache_value = ttnn.typecast(value, ttnn.bfloat8_b)
        ttnn.fill_cache(key_cache, cache_key, batch_idx=0)
        ttnn.fill_cache(value_cache, cache_value, batch_idx=0)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.weights["prefill_sinks"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.linear(
            attended,
            self.weights["o_weight"],
            bias=self.weights["o_bias"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=o_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.add(hidden_states, ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size]))

    def create_kv_cache(self):
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

    def _decode_attention(
        self,
        hidden_states,
        key_cache,
        value_cache,
        cache_position,
        cache_position_tensor,
    ):
        advisor_layouts = (
            self.optimization_config.use_shard_advisor_layouts
            and self.optimization_config.use_shard_advisor_attention_layouts
        )
        norm_input = (
            ttnn.to_memory_config(hidden_states, self.advisor_norm_memory_config) if advisor_layouts else hidden_states
        )
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=self.eps,
            weight=(self.advisor_norm_weights["input_norm"] if advisor_layouts else self.weights["input_norm"]),
            program_config=(self.advisor_norm_program_config if advisor_layouts else None),
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        if advisor_layouts:
            normalized = ttnn.to_memory_config(normalized, self.advisor_qkv_input_config)
        elif self.optimization_config.use_dram_sharded_attention:
            normalized = ttnn.to_memory_config(normalized, self.dram_qkv_input_config)
        fused = ttnn.linear(
            normalized,
            (
                self.decode_qkv_weight
                if self.optimization_config.use_dram_sharded_attention
                else self.weights["qkv_weight"]
            ),
            # The DRAM-sharded matmul contract produces a width-sharded L1
            # tensor.  Match Attention1D by applying bias as a separate
            # broadcast on that output instead of asking the matmul program to
            # fuse an interleaved bias into a sharded result.
            bias=None if self.optimization_config.use_dram_sharded_attention else self.weights["qkv_bias"],
            dtype=ttnn.bfloat16,
            memory_config=(
                self.dram_qkv_output_config
                if self.optimization_config.use_dram_sharded_attention
                else self.advisor_qkv_output_config
                if advisor_layouts
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            program_config=(
                self.dram_qkv_program_config
                if self.optimization_config.use_dram_sharded_attention
                else self.advisor_qkv_program_config
                if advisor_layouts
                else None
            ),
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        if self.optimization_config.use_dram_sharded_attention:
            fused = ttnn.add(fused, self.weights["qkv_bias"], memory_config=self.dram_qkv_output_config)
            # The head-split op uses globally allocated CBs and cannot consume
            # the 80-core width-sharded QKV output directly on Blackhole.
            fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        elif advisor_layouts:
            fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        fused = ttnn.reshape(fused, [1, 1, self.batch, -1])
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        rope_position = ttnn.typecast(cache_position_tensor, ttnn.uint32)
        rope_position = ttnn.reshape(rope_position, [1, self.batch])
        cos = ttnn.embedding(
            rope_position,
            self.decode_cos_matrix,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            rope_position,
            self.decode_sin_matrix,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
        cos = ttnn.interleaved_to_sharded(cos, self.decode_rope_memory_config)
        sin = ttnn.interleaved_to_sharded(sin, self.decode_rope_memory_config)
        query = ttnn.experimental.rotary_embedding_hf(
            query,
            cos,
            sin,
            is_decode_mode=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        key = ttnn.experimental.rotary_embedding_hf(
            key,
            cos,
            sin,
            is_decode_mode=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=cache_position_tensor,
            share_cache=False,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=cache_position_tensor,
            share_cache=False,
        )
        explicit_sliding_mask = self.sliding_window is not None and self.optimization_config.use_explicit_sliding_mask
        if self.optimization_config.dram_attention_query_dram:
            query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        attention_mask = self._decode_sliding_attention_mask(cache_position_tensor) if explicit_sliding_mask else None
        sdpa_kwargs = dict(
            is_causal=not explicit_sliding_mask,
            attn_mask=attention_mask,
            # cur_pos_tensor selects the causal path.  In explicit-mask mode
            # all visibility is carried by attn_mask, as in the SDPA unit
            # tests, so do not supply a second and conflicting selector.
            cur_pos_tensor=None if explicit_sliding_mask else cache_position_tensor,
            attention_sink=self.weights["decode_sinks"],
            scale=self.scale,
            sliding_window_size=None if explicit_sliding_mask else self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=(
                self.sliding_sdpa_program_config if explicit_sliding_mask else self.decode_sdpa_program_config
            ),
            compute_kernel_config=(
                self.sliding_compute_kernel_config if explicit_sliding_mask else self.decode_compute_kernel_config
            ),
        )
        attended = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            **sdpa_kwargs,
        )
        if self.optimization_config.use_decode_concat_heads:
            attended = ttnn.to_memory_config(attended, self.decode_heads_memory_config)
            attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.num_heads)
            attended = ttnn.slice(
                attended,
                [0, 0, 0, 0],
                [1, 1, self.batch, self.num_heads * self.head_dim],
                [1, 1, 1, 1],
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
        else:
            attended = ttnn.reshape(attended, [self.batch, self.num_heads * self.head_dim])
        if advisor_layouts:
            # final_ir.mlir feeds the O projection from L1 interleaved; the
            # concat-heads kernel otherwise leaves this runtime tensor width
            # sharded with a 64-element shard that is incompatible with the
            # advisor's in0_block_w=8.
            attended = ttnn.to_memory_config(attended, ttnn.L1_MEMORY_CONFIG)
        elif self.optimization_config.use_dram_sharded_attention:
            attended = ttnn.to_memory_config(attended, self.dram_o_input_config)
        projected = ttnn.linear(
            attended,
            self.decode_o_weight if self.optimization_config.use_dram_sharded_attention else self.weights["o_weight"],
            bias=None if self.optimization_config.use_dram_sharded_attention else self.weights["o_bias"],
            dtype=ttnn.bfloat16,
            memory_config=(
                self.dram_o_output_config
                if self.optimization_config.use_dram_sharded_attention
                else self.advisor_residual_config
                if advisor_layouts
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            program_config=(
                self.dram_o_program_config
                if self.optimization_config.use_dram_sharded_attention
                else self.advisor_o_program_config
                if advisor_layouts
                else None
            ),
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        if self.optimization_config.use_dram_sharded_attention:
            # The O program uses 90 output shards; return to the public
            # interleaved residual contract before the following RMSNorm,
            # which requires a rectangular sharded grid.
            projected = ttnn.to_memory_config(projected, ttnn.DRAM_MEMORY_CONFIG)
            projected = ttnn.add(projected, self.weights["o_bias"])
        projected = ttnn.reshape(projected, [1, self.batch, 1, self.hidden_size])
        if advisor_layouts:
            hidden_states = ttnn.to_memory_config(hidden_states, self.advisor_residual_config)
            output = ttnn.add(hidden_states, projected, memory_config=self.advisor_residual_config)
            if not self.optimization_config.use_shard_advisor_moe_layouts:
                output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
            return output
        return ttnn.add(hidden_states, projected)

    def _decode_sliding_attention_mask(self, cache_position_tensor):
        position = ttnn.typecast(cache_position_tensor, ttnn.float32)
        position = ttnn.reshape(ttnn.to_layout(position, ttnn.TILE_LAYOUT), [1, 1, 1, 1])
        window_start = ttnn.subtract(position, float(self.sliding_window - 1))
        in_window = ttnn.logical_and(
            ttnn.ge(self.decode_key_positions, window_start),
            ttnn.le(self.decode_key_positions, position),
        )
        # The decode kernel scales QK + mask together.  A finite BF16 value
        # avoids special-value propagation while still underflowing exp after
        # the model's attention scale is applied.
        mask = ttnn.where(in_window, 0.0, -10_000.0)
        mask = ttnn.typecast(mask, ttnn.bfloat16)
        mask = ttnn.repeat(mask, ttnn.Shape([1, 1, self.num_heads, 1]))
        return mask

    def _route(self, normalized, token_count):
        advisor_layouts = (
            self.optimization_config.use_shard_advisor_layouts
            and self.optimization_config.use_shard_advisor_moe_layouts
            and token_count == self.batch
        )
        flat = ttnn.reshape(normalized, [token_count, self.hidden_size])
        router_input = ttnn.typecast(flat, ttnn.float32)
        if advisor_layouts:
            router_input = ttnn.to_memory_config(router_input, self.advisor_router_input_config)
        router_logits = ttnn.linear(
            router_input,
            self.weights["router_weight"],
            bias=self.weights["router_bias"],
            dtype=ttnn.float32,
            memory_config=(self.advisor_router_output_config if advisor_layouts else ttnn.DRAM_MEMORY_CONFIG),
            program_config=(self.advisor_router_program_config if advisor_layouts else None),
            compute_kernel_config=self.compute_kernel_config,
        )
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16)
        if advisor_layouts:
            router_logits = ttnn.to_memory_config(router_logits, ttnn.L1_MEMORY_CONFIG)
        top_values, top_indices = ttnn.topk(router_logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.softmax(top_values, dim=-1, numeric_stable=True)
        if advisor_layouts:
            # final_ir.mlir converts the one-core block-sharded softmax result
            # to L1 interleaved before scatter.
            top_values = ttnn.to_memory_config(top_values, ttnn.L1_MEMORY_CONFIG)
        return ttnn.scatter(
            ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_values,
            memory_config=(self.advisor_routing_weights_config if advisor_layouts else None),
        )

    def _sparse_moe_forward(self, hidden_states, normalized, routing_weights, seq_len):
        if seq_len == 1 and self.optimization_config.expert_input_l1:
            normalized = ttnn.to_memory_config(normalized, ttnn.L1_MEMORY_CONFIG)
        padded_len = ((seq_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if seq_len > 1 and padded_len != seq_len:
            normalized = ttnn.pad(
                normalized,
                [(0, 0), (0, 0), (0, padded_len - seq_len), (0, 0)],
                value=0.0,
            )
            routing_weights = ttnn.pad(
                routing_weights,
                [(0, padded_len - seq_len), (0, 0)],
                value=0.0,
            )
        expert_output = self.experts(
            normalized,
            topk_expert_weights=routing_weights,
            is_decode=seq_len == 1,
        )
        if padded_len != seq_len:
            expert_output = ttnn.slice(
                expert_output,
                [0, 0, 0, 0],
                [1, self.batch, seq_len, self.hidden_size],
            )
        return ttnn.add(hidden_states, expert_output)

    def _dense_moe_forward(self, hidden_states, normalized, routing_weights, seq_len):
        advisor_layouts = (
            self.optimization_config.use_shard_advisor_layouts
            and self.optimization_config.use_shard_advisor_moe_layouts
            and seq_len == 1
        )
        token_count = self.batch * seq_len
        flat = ttnn.reshape(normalized, [token_count, self.hidden_size])
        expert_input = ttnn.reshape(flat, [1, token_count, self.hidden_size])
        expert_input = ttnn.repeat(expert_input, ttnn.Shape([self.num_experts, 1, 1]))
        if advisor_layouts:
            # The captured repeat produces width-90 output before reverting to
            # L1 interleaved for the dense gate/up matmul.
            expert_input = ttnn.to_memory_config(expert_input, self.advisor_expert_hidden_config)
            expert_input = ttnn.to_memory_config(expert_input, ttnn.L1_MEMORY_CONFIG)
        gate_up = ttnn.matmul(
            expert_input,
            self.weights["gate_up_weight"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        if advisor_layouts:
            gate_up = ttnn.to_memory_config(gate_up, self.advisor_expert_gate_up_config)
        gate_up = ttnn.add(
            gate_up,
            self.weights["gate_up_bias"],
            memory_config=self.advisor_expert_gate_up_config if advisor_layouts else None,
        )
        if advisor_layouts:
            gate_up = ttnn.to_memory_config(gate_up, ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(
            gate_up,
            [0, 0, 0],
            [self.num_experts, token_count, 2 * self.intermediate_size],
            [1, 1, 2],
        )
        up = ttnn.slice(
            gate_up,
            [0, 0, 1],
            [self.num_experts, token_count, 2 * self.intermediate_size],
            [1, 1, 2],
        )
        gate = ttnn.clamp(
            gate,
            min=None,
            max=7.0,
            memory_config=self.advisor_expert_hidden_config if advisor_layouts else None,
        )
        up = ttnn.clamp(
            up,
            min=-7.0,
            max=7.0,
            memory_config=self.advisor_expert_hidden_config if advisor_layouts else None,
        )
        gate = ttnn.multiply(gate, ttnn.sigmoid(ttnn.multiply(gate, 1.703125)))
        activated = ttnn.multiply(ttnn.add(up, 1.0), gate)
        if advisor_layouts:
            activated = ttnn.to_memory_config(activated, ttnn.L1_MEMORY_CONFIG)
        expert_output = ttnn.matmul(
            activated,
            self.weights["down_weight"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        if advisor_layouts:
            expert_output = ttnn.to_memory_config(expert_output, self.advisor_expert_hidden_config)
        expert_output = ttnn.add(
            expert_output,
            self.weights["down_bias"],
            memory_config=self.advisor_expert_hidden_config if advisor_layouts else None,
        )
        if advisor_layouts:
            expert_output = ttnn.to_memory_config(expert_output, ttnn.L1_MEMORY_CONFIG)
        if not advisor_layouts:
            routing_weights = ttnn.permute(routing_weights, [1, 0])
        routing_weights = ttnn.reshape(routing_weights, [self.num_experts, token_count, 1])
        if advisor_layouts:
            # Keep the broadcast operand interleaved, matching layout46 in the
            # advisor IR, before asking multiply to emit width-sharded output.
            routing_weights = ttnn.to_memory_config(routing_weights, ttnn.L1_MEMORY_CONFIG)
        expert_output = ttnn.multiply(
            expert_output,
            routing_weights,
            memory_config=self.advisor_expert_hidden_config if advisor_layouts else None,
        )
        if advisor_layouts:
            expert_output = ttnn.to_memory_config(expert_output, ttnn.L1_MEMORY_CONFIG)
        expert_output = ttnn.sum(
            expert_output,
            dim=0,
            memory_config=self.advisor_residual_config if advisor_layouts else None,
        )
        if advisor_layouts:
            expert_output = ttnn.to_memory_config(expert_output, ttnn.L1_MEMORY_CONFIG)
        expert_output = ttnn.reshape(expert_output, [1, self.batch, seq_len, self.hidden_size])
        output = ttnn.add(
            hidden_states,
            expert_output,
            memory_config=(self.advisor_residual_config if advisor_layouts else None),
        )
        if advisor_layouts:
            output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        return output

    def _moe_forward(self, hidden_states, seq_len):
        advisor_layouts = (
            self.optimization_config.use_shard_advisor_layouts
            and self.optimization_config.use_shard_advisor_moe_layouts
            and seq_len == 1
        )
        if advisor_layouts:
            # The captured MoE consumes the width-90 residual emitted by the
            # attention add. This conversion is a no-op in the full advisor
            # chain and makes the MoE-only control reproduce that boundary.
            hidden_states = ttnn.to_memory_config(hidden_states, self.advisor_residual_config)
        norm_input = (
            ttnn.to_memory_config(hidden_states, self.advisor_norm_memory_config) if advisor_layouts else hidden_states
        )
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=self.eps,
            weight=(
                self.advisor_norm_weights["post_attention_norm"]
                if advisor_layouts
                else self.weights["post_attention_norm"]
            ),
            program_config=(self.advisor_norm_program_config if advisor_layouts else None),
            compute_kernel_config=self.compute_kernel_config,
        )
        routing_weights = self._route(normalized, self.batch * seq_len)
        if self.optimization_config.use_sparse_experts:
            return self._sparse_moe_forward(hidden_states, normalized, routing_weights, seq_len)
        return self._dense_moe_forward(hidden_states, normalized, routing_weights, seq_len)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache):
        seq_len = self._validate_hidden_states(hidden_states)
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1")
        if seq_len > self.max_cache_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_cache_len={self.max_cache_len}")
        hidden_states = self._prefill_attention(hidden_states, key_cache, value_cache, seq_len)
        return self._moe_forward(hidden_states, seq_len)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        cache_position,
        cache_position_tensor,
    ):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        if not 0 <= cache_position < self.max_cache_len:
            raise ValueError(f"cache_position must be in [0, {self.max_cache_len}), got {cache_position}")
        hidden_states = self._decode_attention(
            hidden_states,
            key_cache,
            value_cache,
            cache_position,
            cache_position_tensor,
        )
        return self._moe_forward(hidden_states, 1)

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")


__all__ = [
    "EMITTED_BATCH",
    "EMITTED_CACHE_LENGTH",
    "EMITTED_PREFILL_SEQUENCE",
    "SUPPORTED_CONTEXT",
    "OptimizationConfig",
    "OptimizedDecoder",
]
