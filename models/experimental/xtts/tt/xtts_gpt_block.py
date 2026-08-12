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

from models.experimental.xtts.config import (  # noqa: F401 — re-exported for callers
    FFN_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    NEG_INF,  # additive attention-mask fill for masked-out (future) positions
    NUM_HEADS,
)
from models.common.lightweightmodule import LightweightModule

# Per-weight block-float width for the decode matmuls (memory-bound: fewer weight bytes = faster).
# bfloat4_b (4-bit) halves the DRAM stream vs bfloat8_b but is lower precision — only the weights
# that keep the accuracy gates go here. The rest stay bfloat8_b. Names are the c_attn / c_proj /
# mlp_c_fc / mlp_c_proj suffixes.
#
# NOW EMPTY (everything bfloat8_b). ``attn.c_attn.weight`` and ``attn.c_proj.weight`` were bfp4,
# chosen against the block-decode PCC/Frobenius test + an end-to-end EXACT-CODE-MATCH test. Codes
# can match while the latents drift, and it is the latents — not the codes — that the vocoder turns
# into audio, so that gate could not see this: with those two at bfp4, test_tt_inference's
# end-to-end spectrogram-magnitude PCC sat at 0.9840-0.9866 (below its 0.99 bar) for as long as the
# test has existed; promoting both to bfloat8_b takes it to 0.99468, the first config to pass. GPT
# latent PCC also improves (test_tt_gpt_generate 0.99883 -> 0.99954). c_attn produces the K/V that
# persist in the KV cache, so its error compounds across every subsequent decode step, which is why
# a per-step-looking 4-bit choice cost this much end-to-end.
# Promoting c_attn ALONE (c_proj left at bfp4) measured WORSE end-to-end (0.97276) than either
# endpoint — do not treat these two as independently tunable without re-measuring the pair.
# Cost: two weight streams double per decode step (~63 MB/step over 30 layers). NOT yet measured in
# device time — eager-mode host dispatch (~150 ms/token) swamps it in wall-clock, and a full decode
# overflows the device profiler's marker buffer, so it needs a ReadDeviceProfiler drain in the
# generate loop to measure. If decode throughput regresses, measure that before reverting.
_BFP4_WEIGHTS: set[str] = set()
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
_PREFILL_LN_CACHE = {}  # (device-id, Mt) -> (sharded memory_config, sharded LN program_config)
_PREFILL_MLP_CPROJ_CACHE = {}  # (device-id, M) -> (in_mc over FFN, out_mc over hidden, 1D matmul PC)

# Prefill LN width-shards hidden over `_PREFILL_LN_MAX_CORES`. Interleaved LN only parallelizes
# over Mt (= ceil(seq/32)), so at the demo prompt (seq=64 -> Mt=2) it stuck on 2 cores / ~20 us.
# Width@16 (8x2) is within ~0.1 us of width@32 and — critically — matches the mlp c_proj output
# shard grid, so the FFN residual can stay sharded into the next LN and skip that ITS.
# Block-sharding can hit ~9 us but needs nr|Mt so it is not seq-length-portable.
_PREFILL_LN_MAX_CORES = 16

# Prefill mlp c_proj: width-shard the 4096-wide GELU activation over 16 cores (ITS + 1D mcast),
# and emit the hidden activation already width-sharded on those same 16 cores (LN layout). The
# residual add then accepts interleaved residual + sharded mlp out -> sharded, so the next LN
# (ln_f / next-layer ln_1) skips its ITS. Tracy (c_proj->add->LN): current 39.6 us -> 38.6 us.
_PREFILL_MLP_CPROJ_CORES = 16


def _fit_core_grid(nc, max_x, max_y):
    """Largest-x rectangle holding exactly ``nc`` cores inside the device compute grid."""
    for gx in range(min(nc, max_x), 0, -1):
        if nc % gx == 0 and nc // gx <= max_y:
            return gx, nc // gx
    return None


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


def _prefill_ln_cfg(device, mt):
    """Width-sharded PREFILL layer-norm config for seq tile-rows ``mt``: hidden split over
    ``_PREFILL_LN_MAX_CORES`` (falling back if the device grid cannot host that many). Cached per
    (device, mt) because shard height = mt*32. Must be populated by an eager warmup before any
    trace capture — ``create_sharded_memory_config`` is host-side, but the first call still has to
    happen outside the capture."""
    key = (id(device), mt)
    if key not in _PREFILL_LN_CACHE:
        grid = device.compute_with_storage_grid_size()
        nc = _PREFILL_LN_MAX_CORES
        placed = _fit_core_grid(nc, grid.x, grid.y)
        if placed is None:
            # Device smaller than `_PREFILL_LN_MAX_CORES`: walk down the divisors of Kt.
            for cand in (8, 4, 2, 1):
                placed = _fit_core_grid(cand, grid.x, grid.y)
                if placed is not None:
                    nc = cand
                    break
        gx, gy = placed
        bw = (HIDDEN_SIZE // 32) // nc  # width tiles per core
        mc = ttnn.create_sharded_memory_config(
            shape=(mt * 32, HIDDEN_SIZE // nc),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=min(bw, 4),  # sharded LN caps subblock_w at 4 in fp32-dest mode
            block_h=mt,
            block_w=bw,
            inplace=False,
        )
        _PREFILL_LN_CACHE[key] = (mc, pc)
    return _PREFILL_LN_CACHE[key]


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


def sharded_prefill_ln(x, weight, bias, device):
    """Width-sharded PREFILL layer-norm over seq ``x`` ``[1, S, 1024]``. Skips the interleaved->
    sharded copy when ``x`` is already on the LN shard layout (FFN residual path). Always returns
    interleaved L1. Does not consume ``x`` (caller still owns it)."""
    mt = -(-x.shape[-2] // 32)
    mc, pc = _prefill_ln_cfg(device, mt)
    if x.is_sharded() and x.memory_config() == mc:
        h = ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=LAYER_NORM_EPS, program_config=pc, memory_config=mc)
    else:
        xs = ttnn.to_memory_config(x, mc)
        h = ttnn.layer_norm(xs, weight=weight, bias=bias, epsilon=LAYER_NORM_EPS, program_config=pc, memory_config=mc)
        ttnn.deallocate(xs)
    out = ttnn.to_memory_config(h, L1)
    ttnn.deallocate(h)
    return out


def _prefill_mlp_cproj_cfg(device, m):
    """Width-sharded PREFILL mlp ``c_proj`` config for activation rows ``m``.

    Returns ``(in_mc, out_mc, pc)``: FFN width-sharded over 16 cores for the matmul input, hidden
    width-sharded on the same grid (matches ``_prefill_ln_cfg``) for the matmul output so the
    residual add can feed the next LN without an ITS. Cached per (device, m); populate eagerly
    before any trace capture."""
    key = (id(device), m)
    if key not in _PREFILL_MLP_CPROJ_CACHE:
        grid = device.compute_with_storage_grid_size()
        nc = _PREFILL_MLP_CPROJ_CORES
        placed = _fit_core_grid(nc, grid.x, grid.y)
        if placed is None:
            for cand in (8, 4, 2, 1):
                placed = _fit_core_grid(cand, grid.x, grid.y)
                if placed is not None:
                    nc = cand
                    break
        gx, gy = placed
        in_mc = ttnn.create_sharded_memory_config(
            shape=(m, FFN_SIZE // nc),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # Same grid / strategy as prefill LN so residual output is LN-ready (skip ITS).
        out_mc, _ = _prefill_ln_cfg(device, math.ceil(m / 32))
        mt, kt, nt = math.ceil(m / 32), FFN_SIZE // 32, HIDDEN_SIZE // 32
        pcn = math.ceil(nt / (gx * gy))
        ibw = next(b for b in (8, 4, 2, 1) if kt % b == 0)
        osw = next(w for w in (4, 2, 1) if pcn % w == 0)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=osw,
            per_core_M=mt,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        _PREFILL_MLP_CPROJ_CACHE[key] = (in_mc, out_mc, pc)
    return _PREFILL_MLP_CPROJ_CACHE[key]


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
        # per_core_N picked from a decode (M=32) core-count sweep (output-neutral — PCC identical
        # across all pcn): c_proj/mlp_c_proj (nt=32) want pcn=3 (11 cores, -9%/-2% vs pcn=2),
        # c_attn (nt=96) pcn=4, mlp_c_fc (nt=128) pcn=6 (26 cores, -2% vs pcn=4).
        pcn = 3 if nt <= 32 else (4 if nt <= 96 else 6)
        osw = 2 if pcn % 2 == 0 else 1
        ncols = math.ceil(nt / pcn)
        cx = min(gx, ncols)
        cy = math.ceil(ncols / cx)
        if cy > gy:
            cy = gy
            cx = min(gx, math.ceil(ncols / cy))
            pcn = math.ceil(nt / (cx * cy))
            osw = 2 if pcn % 2 == 0 else 1
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
    # PREFILL (M = seq len). When Mt is small, a 2D mcast only activates Mt rows of the grid
    # (e.g. Mt=2 -> 24 cores on c_attn) and leaves most of the chip idle. Prefer 1D whenever Mt
    # cannot fill the grid Y; keep the prior full-grid 2D when it can (long prompts, M~416).
    #
    # GELU-aware aggressive sweep at M=64 (1D/2D interleaved + width/height/block-sharded):
    #   c_attn     64x1024x3072: max-N + ibw=8 (96c) — prior sweep
    #   attn_c_proj 64x1024x1024: max-N + ibw=8 (32c)
    #   mlp_c_fc   64x1024x4096 + fused GELU: 64c / pcn=2 / ibw=8 / osw=2 (~26.6 us). A no-GELU
    #     sweep's 32c/pcn=4/ibw=4 winner is a trap — with fused GELU it regresses to ~32.5-33 us.
    #     Sharded paths were slower or rejected.
    #   mlp_c_proj 64x4096x1024: interleaved max-N + ibw=8 is ~27.6 us; the prefill path in
    #     ``_mlp`` instead width-shards + ITS onto 16 cores (~24.9 + 1.2 ITS = ~26.1 us net).
    ibw = next(b for b in (8, 4, 2, 1) if kt % b == 0)  # K-block (tiles); Kt in {32, 128}
    if mt < gy:
        # Maximize cores along N: smallest pcn whose ceil(Nt/pcn) rectangle fits the device grid.
        cx = cy = pcn = None
        for trial_pcn in (1, 2, 3, 4, 6, 8, 12, 16):
            ncols = math.ceil(nt / trial_pcn)
            placed = _fit_core_grid(ncols, gx, gy)
            if placed is None:
                continue
            cx, cy = placed
            pcn = math.ceil(nt / (cx * cy))
            break
        if cx is not None:
            osw = next(w for w in (4, 2, 1) if pcn % w == 0)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(cx, cy),
                in0_block_w=ibw,
                out_subblock_h=1,
                out_subblock_w=osw,
                per_core_M=mt,
                per_core_N=pcn,
                fuse_batch=True,
                fused_activation=fused_activation,
                mcast_in0=True,
            )
    pcm = max(1, math.ceil(mt / gy))
    pcn = max(1, math.ceil(nt / gx))
    osw = next(w for w in (4, 2, 1) if pcn % w == 0)  # out_subblock_w divides per_core_N, <=4
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
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
        # Interleaved nlp_create_qkv_heads parallelizes only over Mt (= ceil(S/32)), so at the demo
        # prompt (S=64) it uses 2 cores — there is no core-count knob on this op. The sharded
        # factory can use 16 cores but needs width-sharded input + S<=32 per call, which forces
        # Slice/ITS/STI/Concat around it; we keep the single-op path instead. reshape to 4D
        # [b,1,s,3H] is metadata for the op. transpose_k_heads=False keeps K as
        # [b, heads, s, head_dim] (SDPA + decode KV cache expect that layout, not K^T).
        qkv = ttnn.linear(
            x,
            self.attn_c_attn_weight,
            bias=self.attn_c_attn_bias,
            program_config=_mm_1d_config(self.device, x.shape[-2], x.shape[-1], self.attn_c_attn_weight.shape[-1]),
            memory_config=L1,
        )
        b, s, three_h = qkv.shape
        qkv = ttnn.reshape(qkv, (b, 1, s, three_h))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NUM_HEADS, transpose_k_heads=False, memory_config=L1
        )
        ttnn.deallocate(qkv)
        return q, k, v

    def _attn_out(self, attn, shard_out=False):
        """``[b, heads, s, head_dim] -> [b, s, hidden]``. Prefill ``shard_out=True`` writes the
        projection onto the prefill-LN width-sharded layout so the attention residual add can feed
        ``ln_2`` without an ITS. The matmul program_config must use that same core grid —
        otherwise ttnn overrides the output shard spec and a Reshard sneaks back in."""
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)  # fused permute + reshape
        ttnn.deallocate(attn)
        m, k, n = out.shape[-2], out.shape[-1], self.attn_c_proj_weight.shape[-1]
        if shard_out:
            grid = self.device.compute_with_storage_grid_size()
            mt = math.ceil(m / 32)
            if mt < int(grid.y):
                mem, _ = _prefill_ln_cfg(self.device, mt)
                nc = _PREFILL_LN_MAX_CORES
                placed = _fit_core_grid(nc, int(grid.x), int(grid.y))
                gx, gy = placed
                nt = math.ceil(n / 32)
                pcn = math.ceil(nt / (gx * gy))
                kt = math.ceil(k / 32)
                ibw = next(b for b in (8, 4, 2, 1) if kt % b == 0)
                osw = next(w for w in (4, 2, 1) if pcn % w == 0)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=ibw,
                    out_subblock_h=1,
                    out_subblock_w=osw,
                    per_core_M=mt,
                    per_core_N=pcn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                proj = ttnn.linear(
                    out, self.attn_c_proj_weight, bias=self.attn_c_proj_bias, program_config=pc, memory_config=mem
                )
                ttnn.deallocate(out)
                return proj
        proj = ttnn.linear(
            out,
            self.attn_c_proj_weight,
            bias=self.attn_c_proj_bias,
            program_config=_mm_1d_config(self.device, m, k, n),
            memory_config=L1,
        )
        ttnn.deallocate(out)
        return proj

    def _mlp(self, x, decode=False):
        """c_fc (+ GELU fused into the matmul epilogue) -> c_proj. Consumes ``x``.

        Prefill ``c_proj`` width-shards the 4096-wide activation (ITS + 16-core 1D mcast) and
        emits the hidden activation already width-sharded on the prefill-LN layout so the
        residual add can skip the next LN's ITS. Decode stays interleaved (``_mm_1d_config``)."""
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
        if decode:
            out = ttnn.linear(
                h,
                self.mlp_c_proj_weight,
                bias=self.mlp_c_proj_bias,
                program_config=_mm_1d_config(self.device, h.shape[-2], h.shape[-1], self.mlp_c_proj_weight.shape[-1]),
                memory_config=L1,
            )
            ttnn.deallocate(h)
            return out
        # Prefill: only worth it while Mt still uses the 1D path (same regime as _mm_1d_config).
        grid = self.device.compute_with_storage_grid_size()
        mt = math.ceil(h.shape[-2] / 32)
        if mt < int(grid.y):
            in_mc, out_mc, pc = _prefill_mlp_cproj_cfg(self.device, h.shape[-2])
            hs = ttnn.to_memory_config(h, in_mc)
            ttnn.deallocate(h)
            out = ttnn.linear(
                hs, self.mlp_c_proj_weight, bias=self.mlp_c_proj_bias, program_config=pc, memory_config=out_mc
            )
            ttnn.deallocate(hs)
            return out
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

    def _residual_ffn(self, x, decode=False):
        """Shared post-attention half: ``x + mlp(ln_2(x))``. Consumes and replaces ``x``.
        ``decode=True`` routes ln_2 through the width-sharded decode kernel (8 cores, M=1);
        prefill uses the max-core width-sharded prefill kernel (see ``sharded_prefill_ln``).

        Prefill: mlp ``c_proj`` returns width-sharded hidden; the residual add keeps that layout
        (interleaved ``x`` + sharded ``m`` -> sharded) so the next LN skips its ITS."""
        h = (
            self._ln(x, self.ln_2_weight, self.ln_2_bias)
            if decode
            else sharded_prefill_ln(x, self.ln_2_weight, self.ln_2_bias, self.device)
        )
        m = self._mlp(h, decode=decode)  # consumes h
        if decode or not m.is_sharded():
            y = ttnn.add(x, m, memory_config=L1)
        else:
            y = ttnn.add(x, m, memory_config=m.memory_config())
        ttnn.deallocate(x)
        ttnn.deallocate(m)
        return y

    def forward_prefill(self, x):
        """PREFILL — one of the block's two forwards (the other is ``forward_decode``).

        Full causal attention over the prompt, plus the per-layer K, V (each
        ``[b, heads, seq, head_dim]``) used to seed the decode KV cache. K/V are kept
        (returned for the cache); every other intermediate is deallocated. Also serves the
        full teacher-forced pass (callers that want only the hidden state take ``[0]``)."""
        h = sharded_prefill_ln(x, self.ln_1_weight, self.ln_1_bias, self.device)
        q, k, v = self._qkv(h)
        ttnn.deallocate(h)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, memory_config=L1)
        ttnn.deallocate(q)  # k, v kept for the cache
        ao = self._attn_out(attn, shard_out=True)  # LN-layout WS so ln_2 skips ITS
        xa = ttnn.add(x, ao, memory_config=ao.memory_config() if ao.is_sharded() else L1)
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
        return self._residual_ffn(xa, decode=True)  # decode: width-sharded ln_2
