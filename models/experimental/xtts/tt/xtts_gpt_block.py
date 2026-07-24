# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of a single XTTS-v2 GPT decoder block.

Mirrors ``reference/xtts_gpt_block.py`` (a HuggingFace ``GPT2Block``):

    h = x + attn(ln_1(x))          # causal multi-head self-attention
    y = h + mlp(ln_2(h))           # c_fc -> gelu -> c_proj

Weight-layout notes:
  * GPT-2 uses ``Conv1D``, whose weight is stored ``[in, out]`` — already the
    layout ``ttnn.linear`` expects (y = x @ W + b), so NO transpose is needed.
  * Attention is causal with scale ``1/sqrt(head_dim)`` — matches the defaults
    of ``ttnn.transformer.scaled_dot_product_attention``.
"""

import math

import torch
import ttnn

from models.experimental.xtts.reference.xtts_gpt_block import (
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    NUM_HEADS,
)
from models.common.lightweightmodule import LightweightModule

NEG_INF = -1e30  # additive attention-mask fill for masked-out (future) positions

# Per-weight block-float width for the decode matmuls (memory-bound: fewer weight bytes = faster).
# bfloat4_b (4-bit) halves the DRAM stream vs bfloat8_b but is lower precision — only the weights
# that keep the accuracy gates go here. Determined empirically (see the bfp4 sweep); the rest stay
# bfloat8_b. Names are the c_attn / c_proj / mlp_c_fc / mlp_c_proj suffixes.
_BFP4_WEIGHTS = {"attn.c_attn.weight", "attn.c_proj.weight"}
L1 = ttnn.L1_MEMORY_CONFIG  # keep activations in L1 (weights stay in DRAM); the profiler flags the
# decode matmuls' input-0 as DRAM-resident — an L1 activation avoids that per-matmul DRAM read.


def _to_device(torch_tensor, device):
    """torch -> ttnn bf16 tile tensor on device."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


def _to_device_w8(torch_tensor, device, dtype=ttnn.bfloat8_b):
    """torch -> ttnn block-float tile weight. The decode matmuls are batch-1 (M=32, one token padded
    to a tile) so they are MEMORY-bound — the time is dominated by streaming the weight from DRAM,
    not the (tiny) M=32 math — so shrinking the weight bytes directly shrinks the dominant cost.
    ``dtype`` picks the block-float width per weight: bfloat8_b (8-bit) is the safe default; the
    larger, less sensitive weights use bfloat4_b (4-bit, half the bytes) where accuracy still holds
    (gated by the block-decode PCC/Frobenius test + the end-to-end exact-code-match test)."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=dtype,
    )


def _to_device_bias(torch_tensor, device):
    """Matmul bias -> bf16 tile [1, N]. The rank>=2 shape makes the tilized bias's padded
    penultimate dim == 32, which (with an explicit matmul program_config) is required for ttnn
    to FUSE the bias into the matmul epilogue instead of emitting a separate broadcast add."""
    return _to_device(torch_tensor.reshape(1, -1), device)


_LN_SHARD_CACHE = {}  # device-id -> (sharded memory_config, sharded LN program_config)


def _decode_ln_cfg(device):
    """Build (and cache per device) the width-sharded decode layer-norm config: hidden (1024)
    split over 8 cores, one tile row (decode M = 1 token)."""
    key = id(device)
    if key not in _LN_SHARD_CACHE:
        nc = 8
        bw = HIDDEN_SIZE // nc // 32  # width tiles per core
        mc = ttnn.create_sharded_memory_config(
            shape=(32, HIDDEN_SIZE // nc),
            core_grid=ttnn.CoreGrid(x=nc, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[nc, 1], subblock_w=bw, block_h=1, block_w=bw, inplace=False
        )
        _LN_SHARD_CACHE[key] = (mc, pc)
    return _LN_SHARD_CACHE[key]


def sharded_decode_ln(x, weight, bias, device):
    """Width-sharded DECODE layer-norm (single token, M padded to one tile): reshard the L1
    activation to width-sharded, run the sharded LN kernel, reshard the result back to interleaved
    L1. ~48% faster than the interleaved LN and BIT-IDENTICAL (isolated PCC 1.0) because the whole
    1024-wide reduction is parallelized over 8 cores instead of running on too few. Shared by the
    block (ln_1/ln_2), the stack (ln_f), and the model (final_norm). Consumes ``x``."""
    mc, pc = _decode_ln_cfg(device)
    xs = ttnn.to_memory_config(x, mc)
    h = ttnn.layer_norm(xs, weight=weight, bias=bias, epsilon=LAYER_NORM_EPS, program_config=pc, memory_config=mc)
    ttnn.deallocate(xs)
    out = ttnn.to_memory_config(h, L1)
    ttnn.deallocate(h)
    return out


def _mm_1d_config(device, m, k, n, fused_activation=None):
    """1D-multicast matmul program_config for the GPT linears (mcast the L1 activation, stream the
    DRAM weight per-core over N). Passing an explicit config is what lets ttnn fuse the bias (and,
    for c_fc, the GELU) into the epilogue for an L1 output — the auto path post-processes both as
    separate ops. Built per-forward because prefill M (= seq len) varies; decode M = 1."""
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = math.ceil(m / 32), math.ceil(k / 32), math.ceil(n / 32)
    if mt == 1:
        # DECODE (single token, M=32): a memory-bound skinny matmul. A program-config sweep over the
        # four GPT decode shapes showed the full-grid / one-N-tile-per-core layout is ~20-35% SLOWER
        # than consolidating onto FEWER cores that each compute several N-tiles with a 2-wide output
        # subblock — less activation-mcast fan-out and better weight-stream reuse dominate at M=32.
        # Keep in0_block_w=4 (the pre-optimization value): it fixes the bfp8 K-accumulation grouping,
        # so output is BIT-IDENTICAL to before (the grid/per_core_N/out_subblock_w changes only
        # repartition output tiles, not the reduction order). ibw=8 is ~1% faster but shifts the
        # accumulation enough to flip borderline greedy argmax picks in free-running decode
        # (exact-match prefix regressed 16/16 -> 10/16), so ibw=4 preserves exact output.
        ibw = next(b for b in (4, 2, 1) if kt % b == 0)
        pcn = 2 if nt <= 32 else (3 if nt <= 64 else 4)
        osw = 2 if pcn % 2 == 0 else 1
        ncols = math.ceil(nt / pcn)
        cx = min(gx, ncols)
        cy = math.ceil(ncols / cx)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cx, cy),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=osw,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=fused_activation,
            mcast_in0=True,
        )
    # PREFILL (M = seq len): compute-bound, different regime — keep the full-grid layout.
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=4,  # K-block (tiles); divides Kt for all GPT linears (Kt in {32, 128})
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=mt,
        per_core_N=math.ceil(nt / (gx * gy)),
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


class TtXttsGptBlock(LightweightModule):
    def __init__(
        self,
        state_dict,
        device,
        layer_idx=0,
    ):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx

        prefix = f"gpt.gpt.h.{layer_idx}."

        # Load layer norm parameters
        self.ln_1_weight = _to_device(state_dict[prefix + "ln_1.weight"], device)
        self.ln_1_bias = _to_device(state_dict[prefix + "ln_1.bias"], device)
        self.ln_2_weight = _to_device(state_dict[prefix + "ln_2.weight"], device)
        self.ln_2_bias = _to_device(state_dict[prefix + "ln_2.bias"], device)

        # Attention/MLP weights in bfloat8_b (memory-bound decode matmuls — see _to_device_w8);
        # biases are bf16 [1, N] (see _to_device_bias) so they fuse into the matmul epilogue under
        # the explicit program_config used in the forwards.
        def _w(name):  # bfloat4_b if the weight is in the bfp4 policy set, else bfloat8_b
            dtype = ttnn.bfloat4_b if name in _BFP4_WEIGHTS else ttnn.bfloat8_b
            return _to_device_w8(state_dict[prefix + name], device, dtype=dtype)

        self.attn_c_attn_weight = _w("attn.c_attn.weight")
        self.attn_c_attn_bias = _to_device_bias(state_dict[prefix + "attn.c_attn.bias"], device)
        self.attn_c_proj_weight = _w("attn.c_proj.weight")
        self.attn_c_proj_bias = _to_device_bias(state_dict[prefix + "attn.c_proj.bias"], device)

        self.mlp_c_fc_weight = _w("mlp.c_fc.weight")
        self.mlp_c_fc_bias = _to_device_bias(state_dict[prefix + "mlp.c_fc.bias"], device)
        self.mlp_c_proj_weight = _w("mlp.c_proj.weight")
        self.mlp_c_proj_bias = _to_device_bias(state_dict[prefix + "mlp.c_proj.bias"], device)

    def _qkv(self, x):  # [b, s, hidden] -> q, k, v each [b, heads, s, head_dim]
        # Split the [b, s, 3*hidden] c_attn output (GPT-2 [Q|K|V] block layout) into per-head Q, K, V.
        # ttnn.experimental.nlp_create_qkv_heads is measurably faster than the transformer-namespace
        # split_query_key_value_and_split_heads wrapper (~43 vs ~63 us/call at decode shape, identical
        # output) — it wants a 4D [b, 1, s, 3*hidden] input, so add the leading singleton dim (a
        # metadata reshape). transpose_k_heads=False keeps K as [b, heads, s, head_dim] (SDPA + the
        # decode KV cache expect that layout, not K^T).
        qkv = ttnn.linear(
            x,
            self.attn_c_attn_weight,
            bias=self.attn_c_attn_bias,
            program_config=_mm_1d_config(self.device, x.shape[-2], x.shape[-1], self.attn_c_attn_weight.shape[-1]),
            memory_config=L1,
        )
        b, s, three_h = qkv.shape
        qkv = ttnn.reshape(qkv, (b, 1, s, three_h))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=NUM_HEADS, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        return q, k, v

    def _attn_out(self, attn):  # [b, heads, s, head_dim] -> [b, s, hidden]
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)  # fused permute + reshape
        ttnn.deallocate(attn)
        proj = ttnn.linear(
            out,
            self.attn_c_proj_weight,
            bias=self.attn_c_proj_bias,
            program_config=_mm_1d_config(self.device, out.shape[-2], out.shape[-1], self.attn_c_proj_weight.shape[-1]),
            memory_config=L1,
        )
        ttnn.deallocate(out)
        return proj

    def _mlp(self, x):
        """c_fc (+ GELU fused into the matmul epilogue) -> c_proj. Consumes ``x``."""
        # c_fc fuses BOTH bias and GELU into the matmul epilogue via the program_config's
        # fused_activation. (GELU, False) == the old activation="gelu" (string "gelu" maps to
        # UnaryOpType.GELU with param False), so the math is unchanged (validated by PCC).
        h = ttnn.linear(
            x,
            self.mlp_c_fc_weight,
            bias=self.mlp_c_fc_bias,
            program_config=_mm_1d_config(
                self.device,
                x.shape[-2],
                x.shape[-1],
                self.mlp_c_fc_weight.shape[-1],
                fused_activation=(ttnn.UnaryOpType.GELU, False),
            ),
            memory_config=L1,
        )
        ttnn.deallocate(x)
        out = ttnn.linear(
            h,
            self.mlp_c_proj_weight,
            bias=self.mlp_c_proj_bias,
            program_config=_mm_1d_config(self.device, h.shape[-2], h.shape[-1], self.mlp_c_proj_weight.shape[-1]),
            memory_config=L1,
        )
        ttnn.deallocate(h)
        return out

    def _ln(self, x, weight, bias):
        """DECODE layer-norm via the shared width-sharded kernel (``sharded_decode_ln``). Consumes ``x``."""
        return sharded_decode_ln(x, weight, bias, self.device)

    def _residual_ffn(self, x, sharded=False):
        """Shared post-attention half: ``x + mlp(ln_2(x))``. Consumes and replaces ``x``.
        ``sharded`` routes ln_2 through the width-sharded decode kernel (see ``_ln``)."""
        h = (
            self._ln(x, self.ln_2_weight, self.ln_2_bias)
            if sharded
            else ttnn.layer_norm(
                x, weight=self.ln_2_weight, bias=self.ln_2_bias, epsilon=LAYER_NORM_EPS, memory_config=L1
            )
        )
        m = self._mlp(h)  # consumes h
        y = ttnn.add(x, m, memory_config=L1)
        ttnn.deallocate(x)
        ttnn.deallocate(m)
        return y

    def forward_prefill(self, x):
        """PREFILL — one of the block's two forwards (the other is ``forward_decode``).

        Full causal attention over the prompt, plus the per-layer K, V (each
        ``[b, heads, seq, head_dim]``) used to seed the decode KV cache. K/V are kept
        (returned for the cache); every other intermediate is deallocated. Also serves the
        full teacher-forced pass (callers that want only the hidden state take ``[0]``)."""
        # print(f"[TtXttsGptBlock.forward_prefill] layer={self.layer_idx} x={list(x.shape)}")
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS, memory_config=L1)
        q, k, v = self._qkv(h)
        ttnn.deallocate(h)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, memory_config=L1)
        ttnn.deallocate(q)  # k, v kept for the cache
        ao = self._attn_out(attn)
        xa = ttnn.add(x, ao, memory_config=L1)
        ttnn.deallocate(x)
        ttnn.deallocate(ao)
        return self._residual_ffn(xa), k, v

    def forward_decode(self, x, k_cache, v_cache, onehot, add_mask, write_idx=None):
        """DECODE — one of the block's two forwards. One token over a FIXED-size KV cache
        (no concat growth: concat on a tile-misaligned seq dim forces untilize->concat->retilize,
        ~15% of the step — this path avoids all of it).

        ``k_cache``/``v_cache`` are ``[1, heads, MAX, head_dim]`` PERSISTENT buffers updated IN
        PLACE at the current position; attention then runs over the whole cache with an additive
        position mask (``add_mask`` ``[1, 1, 1, MAX]``: 0 for cached positions, -inf ahead). Two
        cache-write modes:
          * EAGER (``write_idx`` = Python int): ``ttnn.update_cache`` writes ONLY that row — O(1),
            ~2x faster than touching the whole cache.
          * TRACED (``write_idx`` None): a device one-hot select ``where(onehot, newKV, cache)``
            ([1,1,MAX,1], 1 at the write row) — data-driven, so one capture replays at any position.
        Returns the FFN output."""
        # print(f"[TtXttsGptBlock.forward_decode] layer={self.layer_idx} x={list(x.shape)} write_idx={write_idx}")
        h = self._ln(x, self.ln_1_weight, self.ln_1_bias)  # width-sharded decode LN (see _ln)
        q, k, v = self._qkv(h)  # each [1, heads, 1, head_dim]
        ttnn.deallocate(h)
        if write_idx is not None:
            ttnn.update_cache(k_cache, k, write_idx)  # O(1): write only row write_idx
            ttnn.update_cache(v_cache, v, write_idx)
        else:
            # data-driven select at the one-hot row (trace-safe; whole-cache elementwise).
            ttnn.where(onehot, k, k_cache, output_tensor=k_cache)
            ttnn.where(onehot, v, v_cache, output_tensor=v_cache)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # Masked attention over the full fixed cache, fused into ONE SDPA op (scale + q·Kᵀ + additive
        # mask + softmax + ·V) instead of permute+matmul+mul+add+softmax+matmul. ``add_mask``
        # [1, 1, 1, MAX] is 0 for cached positions, -inf ahead (broadcasts over heads and the 1 query).
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k_cache, v_cache, attn_mask=add_mask, is_causal=False, scale=1.0 / math.sqrt(HEAD_DIM), memory_config=L1
        )  # [1, heads, 1, head_dim]
        ttnn.deallocate(q)
        ao = self._attn_out(attn)
        xa = ttnn.add(x, ao, memory_config=L1)
        ttnn.deallocate(x)
        ttnn.deallocate(ao)
        return self._residual_ffn(xa, sharded=True)  # decode: width-sharded ln_2
