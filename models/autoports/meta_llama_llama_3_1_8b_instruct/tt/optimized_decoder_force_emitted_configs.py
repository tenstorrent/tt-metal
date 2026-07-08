# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decode variant that forces the *emitted* (`model_ttnn.py`) memory-config layout.

This is an experiment, not a tuned path. It starts from the production
``OptimizedDecoder`` and pushes the tt-forge codegen emit's decode layout
(``ttnn-models/.../graph_0/model_ttnn.py``) back onto the decode path: keep the whole
layer **L1 WIDTH_SHARDED** (input RMSNorm, matmul outputs, gate/up/multiply, residuals)
and run every matmul with a **1D-multicast** ``MatmulMultiCoreReuseMultiCast1DProgramConfig``
over interleaved weights (the emit's style, *not* the DRAM-sharded ``down`` recipe), with
SiLU fused into the gate matmul.

Two scopes are supported (``force_scope``):

* ``"all"`` — force the emitted layout on **every** decode compute op, including the ones
  the ``optimize`` stage deliberately tuned (this reverts the DRAM-sharded ``down`` and the
  tuned gate/up back to the emit's plain 1D style).
* ``"unchanged"`` (default) — force the emitted layout **only** on the ops the optimizer
  left at the functional/DRAM default (input RMSNorm, packed QKV matmul, O projection, and
  the head-glue reshapes), and **keep** the optimizer's tuned ops (sharded residual /
  post-attention RMSNorm, tuned gate/up geometry, DRAM-sharded ``down``). This is the
  "fill the gaps the optimizer left, don't overwrite its wins" experiment.

The emit was generated for a device with an 11-wide compute grid (``CoreCoord(11, 9)``
program grids), which does not fit the Wormhole 8x8 compute grid, so the emit's exact
``CoreRangeSet`` / grid / ``per_core_N`` are **legalized to 8x8** (full-grid width-sharding,
``per_core_N = N_tiles / 64``). All 8x8 divisions are exact for this model, so no config
numbers are invented beyond that legalization; the ``in0_block_w`` / subblock formulas reuse
the parent decoder's helpers. dtype (BFP4 weights) and math fidelity (LoFi) are kept
identical to ``OptimizedDecoder`` so any delta is attributable to memory layout alone.
"""

from __future__ import annotations

import math
import time

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderTimings,
    _decode_head_sub_core_grids,
    signpost,
)


class OptimizedDecoderForceEmittedConfigs(OptimizedDecoder):
    """``OptimizedDecoder`` decode path with the emitted L1-sharded layout forced on."""

    optimization_profile = {
        **OptimizedDecoder.optimization_profile,
        "name": "llama31_8b_instruct_optimized_decoder_force_emitted_configs_v1",
        "decode_layout": (
            "emitted model_ttnn layout forced: L1 WIDTH_SHARDED input norm + QKV/O matmul "
            "outputs, 1D-multicast matmuls over interleaved weights; force_scope='all' also "
            "reverts gate/up/down to emitted 1D, force_scope='unchanged' keeps the "
            "optimizer's tuned gate/up + DRAM-sharded down; legalized to 8x8"
        ),
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # "unchanged" = force emitted layout only where the optimizer kept the DRAM default;
        # "all" = force emitted layout on every decode compute op.
        self.force_scope = "unchanged"

    def _orig_grid(self) -> tuple[int, int]:
        compute_grid = self.mesh_device.compute_with_storage_grid_size()
        return min(8, compute_grid.x), min(8, compute_grid.y)

    def _orig_full_core_range(self) -> ttnn.CoreRangeSet:
        grid_x, grid_y = self._orig_grid()
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})

    def _orig_width_sharded_memcfg(self, width: int) -> ttnn.MemoryConfig:
        grid_x, grid_y = self._orig_grid()
        num_cores = grid_x * grid_y
        shard_w = self._pad_to(width, 32 * num_cores) // num_cores
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self._orig_full_core_range(), [32, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _orig_1d_program_config(
        self, k: int, n: int, *, fused_activation=None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        grid_x, grid_y = self._orig_grid()
        num_cores = grid_x * grid_y
        per_core_k_tiles = self._pad_to(k, 32 * num_cores) // (32 * num_cores)
        per_core_n = self._pad_to(n, 32 * num_cores) // (32 * num_cores)

        in0_block_w = per_core_k_tiles
        while in0_block_w > 1 and per_core_k_tiles % in0_block_w != 0:
            in0_block_w -= 1

        out_subblock_w = min(4, per_core_n)
        while out_subblock_w > 1 and per_core_n % out_subblock_w != 0:
            out_subblock_w -= 1

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=fused_activation,
            mcast_in0=True,
        )

    def _orig_matmul(self, activation, weight, k, n, *, transpose_b, fidelity_config, fused_activation=None):
        activation = ttnn.to_memory_config(activation, self._orig_width_sharded_memcfg(k))
        return ttnn.matmul(
            activation,
            weight,
            transpose_b=transpose_b,
            dtype=ttnn.bfloat16,
            memory_config=self._orig_width_sharded_memcfg(n),
            program_config=self._orig_1d_program_config(k, n, fused_activation=fused_activation),
            compute_kernel_config=fidelity_config,
        )

    def _decode_qkv_emitted(self, hidden_states, position_cos, position_sin, batch_size):
        """Attention front with the emitted layout forced: sharded input norm, 1D QKV, L1 head glue."""
        decode_head_memcfg = self._decode_head_memory_config(batch_size)
        residual_memcfg = self._decode_residual_memory_config(batch_size)

        # Emit: input RMSNorm is L1 WIDTH_SHARDED (optimizer left it DRAM 1-core).
        hidden_states = ttnn.to_memory_config(hidden_states, residual_memcfg)
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.input_layernorm_weight,
            memory_config=residual_memcfg,
            program_config=self._decode_residual_norm_program_config(batch_size),
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )

        # Emit: packed QKV matmul, L1 WIDTH_SHARDED output via 1D-multicast.
        qkv = self._orig_matmul(
            hidden_states,
            self.qkv_decode_weight,
            self.cfg.hidden_size,
            self.cfg.num_attention_heads * self.cfg.head_dim + 2 * self.cfg.num_key_value_heads * self.cfg.head_dim,
            transpose_b=False,
            fidelity_config=self.attention_compute_kernel_config,
        )
        # Emit moves the packed QKV to L1 interleaved before head reshapes.
        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)

        q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, batch_size, 4096], [1, 1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.slice(
            qkv, [0, 0, 0, 4096], [1, 1, batch_size, 5120], [1, 1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        v = ttnn.slice(
            qkv, [0, 0, 0, 5120], [1, 1, batch_size, 6144], [1, 1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        q = ttnn.reshape(q, [1, batch_size, self.cfg.num_attention_heads, self.cfg.head_dim])
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(q, position_cos, position_sin, None, memory_config=ttnn.L1_MEMORY_CONFIG)
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)
        q = ttnn.slice(q, [0, 0, 0, 0], [1, batch_size, self.cfg.num_attention_heads, self.cfg.head_dim], [1, 1, 1, 1])
        q = ttnn.to_memory_config(q, decode_head_memcfg)

        k = ttnn.reshape(k, [1, batch_size, self.cfg.num_key_value_heads, self.cfg.head_dim])
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(k, position_cos, position_sin, None, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.slice(k, [0, 0, 0, 0], [1, batch_size, self.cfg.num_key_value_heads, self.cfg.head_dim], [1, 1, 1, 1])
        k = ttnn.to_memory_config(k, decode_head_memcfg)

        v = ttnn.reshape(v, [1, batch_size, self.cfg.num_key_value_heads, self.cfg.head_dim])
        v = ttnn.to_memory_config(v, decode_head_memcfg)
        return q, k, v

    def _mlp_emitted(self, post_norm):
        """MLP with the emitted 1D-multicast layout (force_scope='all' only)."""
        gate = self._orig_matmul(
            post_norm,
            self.gate_proj_weight,
            self.cfg.hidden_size,
            self.cfg.intermediate_size,
            transpose_b=True,
            fidelity_config=self.mlp_compute_kernel_config,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        up = self._orig_matmul(
            post_norm,
            self.up_proj_weight,
            self.cfg.hidden_size,
            self.cfg.intermediate_size,
            transpose_b=True,
            fidelity_config=self.mlp_compute_kernel_config,
        )
        gated = ttnn.multiply(
            gate,
            up,
            dtype=ttnn.bfloat16,
            memory_config=self._orig_width_sharded_memcfg(self.cfg.intermediate_size),
        )
        return self._orig_matmul(
            gated,
            self.down_proj_weight,
            self.cfg.intermediate_size,
            self.cfg.hidden_size,
            transpose_b=True,
            fidelity_config=self.mlp_compute_kernel_config,
        )

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        use_persistent_ccl: bool = True,
    ) -> ttnn.Tensor:
        del use_persistent_ccl
        if self.force_scope not in ("all", "unchanged"):
            raise ValueError(f"force_scope must be 'all' or 'unchanged', got {self.force_scope!r}")
        signpost("PERF_DECODE")
        start = time.perf_counter()
        batch_size = hidden_states.shape[-2]
        if hidden_states.shape[-3] != 1:
            raise ValueError(f"decode expects one logical token per user, got shape {hidden_states.shape}")

        # Attention front: emitted layout forced in both scopes (input norm, QKV, head glue).
        q, k, v = self._decode_qkv_emitted(hidden_states, position_cos, position_sin, batch_size)
        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(
            k_cache,
            k,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self.paged_kv_config.block_size,
            num_kv_heads=self.cfg.num_key_value_heads,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self.paged_kv_config.block_size,
            num_kv_heads=self.cfg.num_key_value_heads,
        )

        sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            program_config=self.sdpa_decode_program_config,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            block_size=self.paged_kv_config.block_size,
            num_kv_heads=self.cfg.num_key_value_heads,
        )
        sdpa = ttnn.to_memory_config(sdpa, self._decode_head_memory_config(batch_size))
        attn = ttnn.experimental.nlp_concat_heads_decode(
            sdpa,
            num_heads=self.cfg.num_attention_heads,
            sub_core_grids=_decode_head_sub_core_grids(self.mesh_device, batch_size),
        )
        attn = ttnn.slice(attn, [0, 0, 0, 0], [1, 1, batch_size, 4096], [1, 1, 1, 1])

        # O projection: emitted layout forced in both scopes (optimizer left it DRAM default).
        attn_out = self._orig_matmul(
            attn,
            self.o_proj_weight,
            self.cfg.hidden_size,
            self.cfg.hidden_size,
            transpose_b=True,
            fidelity_config=self.attention_compute_kernel_config,
        )

        # Residual + post-attention norm: kept as the optimizer tuned them (already sharded).
        residual_memcfg = self._decode_residual_memory_config(batch_size)
        hidden_states = ttnn.to_memory_config(hidden_states, residual_memcfg)
        attn_out = ttnn.to_memory_config(attn_out, residual_memcfg)
        attn_residual = ttnn.add(attn_out, hidden_states, dtype=ttnn.bfloat16, memory_config=residual_memcfg)
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=residual_memcfg,
            program_config=self._decode_residual_norm_program_config(batch_size),
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )

        if self.force_scope == "all":
            # Revert the MLP to the emitted 1D-multicast layout too.
            mlp_out = self._mlp_emitted(post_norm)
        else:
            # Keep the optimizer's tuned gate/up geometry + DRAM-sharded down.
            mlp_out = self._mlp(
                post_norm,
                use_dram_sharded_down=True,
                input_memory_config=None,
                output_memory_config=None,
            )
        mlp_out = ttnn.to_memory_config(mlp_out, residual_memcfg)
        output = ttnn.add(mlp_out, attn_residual, dtype=ttnn.bfloat16, memory_config=residual_memcfg)
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_DECODE_END")
        self.timings = OptimizedDecoderTimings(
            prefill_ms=self.timings.prefill_ms,
            decode_ms=elapsed_ms,
            traced_decode_ms=self.timings.traced_decode_ms,
        )
        return output
