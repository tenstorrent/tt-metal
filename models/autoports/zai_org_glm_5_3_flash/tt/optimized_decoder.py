# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized TTNN decoder layer for zai-org/GLM-5.3-Flash (`glm5_next`), single 1x1 Blackhole mesh.

Same public contract as `fused_decoder.FusedDecoder` / `functional_decoder.FunctionalDecoder` (streams
`[1, 1, T, 16384]` fp32 in / out, chunk planner for any logical length >= 1, paged MLA latent cache, persistent
per-slot KDA state, traced decode with `cur_pos == -1` inactive slots). On top of the fused op graph
(`FusedDecoderConfig` defaults) this stage changes (defaults = `OptimizedDecoderConfig`, chosen by measurement;
doc/optimized_decoder/README.md):

    decode projections  every weight-streaming decode matmul (KDA q|k|v, f_a|g_a|b, f_b, g_b, o; MLA q_a|kv_a, q_b,
                        o; dense MLP / shared expert gate|up and down; router) runs as a DRAM-sharded matmul
                        (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`): weights width-sharded over the 8
                        DRAM banks, activations width-sharded in L1, two readers per bank with the
                        `MESH_ROW_PLUS_X` reader placement (P4), per-role storage grid and K block.
    prefill projections the same DRAM-sharded weights (no second copy), read by explicit 2D multicast programs on
                        the full core grid (per-core M = ceil(M / core rows)).
    routed experts      `fused_experts` with the plan's LoFi fidelity (bfloat4_b weights).
    decode activations  intermediates in L1 (interleaved) instead of DRAM.
    prefill warm-up     every bucket is compiled once per layer kind and device (Stage 1 C-4); later layers of the same
                        kind only warm their own expert program (fused_experts keys its program on the weights).
    traced prefill      `prepare_prefill_traces` / `capture_prefill_traces` / `run_prefill_trace` /
                        `prefill_forward_traced`: one trace per bucket of a bounded set (align x 2^i up to 2048: KDA 7,
                        MLA 6), persistent input / output buffers, slot-independent (KDA through a scratch recurrent
                        state, MLA through the device page tables / chunk start), prompts split by `plan_traced_chunks`
                        (full buckets + one tail padded by less than the alignment); inside a trace the MoE runs the
                        expert-sorted pair path (dispatch-bound eagerly, the fastest once captured; pair tokens computed
                        and the inverse permutation scattered, no gathers).
    prefill mHC         the composite mHC slices each sublayer's residual into its 4 streams once (shared by pre and
                        post), sum of squares by ttnn.square.

Precision policy (plan B4 / Q1): routed experts bfloat4_b, LoFi in decode, HiFi2 in prefill; KDA / MLA projections, shared expert, dense MLP bfloat8_b
HiFi2; low-rank KDA projections, router, mHC bf16 / fp32 HiFi4; residual streams fp32; KDA state fp32.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as _mpost
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as _mhc
from models.autoports.zai_org_glm_5_3_flash.tt.config import round_up
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import (
    BLOCK_SIZE,
    PREFILL_CHUNK,
    DecodeTrace,
    bucket_for,
)
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import (
    CompositeHyperConnection,
    FusedDecoder,
    FusedDecoderConfig,
    FusedDecoderLayer,
    FusedHyperConnection,
    FusedKDA,
    FusedMLA,
    FusedMoE,
    FusedRouter,
    FusedSwiGLU,
    StreamMajorHyperConnection,
    from_stream_major,
    to_flat,
    to_stacked,
)
from models.autoports.zai_org_glm_5_3_flash.tt.kda import CONV_K
from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre
from models.autoports.zai_org_glm_5_3_flash.tt.kda_rec import KdaRec
from models.autoports.zai_org_glm_5_3_flash.tt.precision import ACT_DTYPE
from models.autoports.zai_org_glm_5_3_flash.tt.router_tail import RouterTail
from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource, TorchStateDictSource

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
TILE = 32
FP32 = ttnn.float32
BF16 = ttnn.bfloat16
_PLACEMENT = getattr(
    ttnn._ttnn.operations.matmul, "DramReaderPlacement", None
)  # the fork's DS matmul enum; absent here

__all__ = ["OptimizedDecoder", "OptimizedDecoderConfig", "DSLinear", "PrefillTrace", "plan_traced_chunks"]


def traced_buckets(align: int) -> tuple[int, ...]:
    """Bounded traced-prefill bucket set of an attention kind: align * 2^i up to the 2048-token chunk."""
    out, p = [], align
    while p <= PREFILL_CHUNK:
        out.append(p)
        p *= 2
    return tuple(out)


def plan_traced_chunks(seq_len: int, align: int, buckets: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """[(offset in the new input, valid tokens, physical length)] for traced prefill: full 2048-token chunks, then the
    remainder split into descending bucket sizes (every chunk but the last exactly full), the last piece (< align
    tokens) padded to `align`. Every physical chunk is a bucket and only the last one has a padded tail, which is
    shorter than the alignment: the Stage 1 padding contract (a KDA chunk never holds a fully padded 32-token
    recurrence chunk; MLA chunk starts stay multiples of the 64-token page)."""
    if seq_len < 1:
        raise ValueError("prefill needs at least one token")
    out, off = [], 0
    while off < seq_len:
        r = seq_len - off
        p = max((b for b in buckets if b <= r), default=None)
        if p is None:
            out.append((off, r, bucket_for(r, align)))
            break
        out.append((off, p, p))
        off += p
    return out


@dataclass
class PrefillTrace:
    """One captured prefill chunk of a layer: persistent input / output buffers of a padded bucket and the trace."""

    bucket: int
    x: ttnn.Tensor  # [1, 1, bucket, 16384] fp32 DRAM, written before each replay (host or device copy)
    y: ttnn.Tensor  # [1, 1, bucket, 16384] fp32 DRAM, written by the trace (copy of the layer output)
    trace_id: int | None = None
    # host_io: x / y ROW_MAJOR, tilized / untilized inside the trace (a host [bucket, 16384] fp32 transfer then needs no
    # host-side tile conversion); else TILE (device-to-device chaining of layers)
    host_io: bool = True


@dataclass
class OptimizedDecoderConfig(FusedDecoderConfig):
    """Optimization knobs on top of FusedDecoderConfig (defaults = the measured best; README)."""

    # fused-mHC fn projection (FusedDecoderConfig field): "flat_k32" = sharded rms_norm + 512-way K-split batched
    # matmul for one decode row (OptimizedHyperConnection), the fused decoder's auto policy for more rows
    mhc_proj: str = "flat_k32"
    # DRAM-sharded decode projections (False = the fused decoder's interleaved linears)
    ds_decode: bool = True
    # readers per DRAM bank (1..3, Blackhole) and their placement (NOC_HOPS | ROW_PLUS_X | MESH_ROW_PLUS_X)
    ds_readers: int = 2
    ds_placement: str = "MESH_ROW_PLUS_X"
    # per-role overrides "role:value;role:value" (roles: kda_qkv kda_lowrank kda_fb kda_gb kda_o mla_qa_kva mla_qb
    # mla_o mlp_gate_up mlp_down shared_gate_up shared_down router): activation storage cores, K block, readers
    # (measured, E3/ds_sweep.jsonl + ab_variants.jsonl `roles`: the low-rank / router / dense-down storage grid and K
    # block from the per-role sweep; every other role's heuristic default is within 0.5 us of its sweep optimum)
    ds_cores: str = "kda_lowrank:8;mlp_down:32"
    ds_block_w: str = "kda_lowrank:32;router:32;mlp_down:6"
    ds_readers_per_role: str = ""
    # roles kept on the interleaved linear (comma-separated): the K = 128 f_b / g_b projections stream 2 MB each and the
    # DRAM-sharded program's fixed cost dominates (L4 B=8 sweep: 18.2 us DS vs 8.7-8.8 us interleaved, E3/ds_sweep.jsonl)
    ds_skip_roles: str = "kda_fb,kda_gb"
    # prefill matmuls read the DRAM-sharded weights (no interleaved copy kept)
    ds_prefill_weights: bool = True
    # decode intermediates in L1 interleaved
    decode_l1: bool = True
    # fused_experts math fidelity (plan B4: LoFi for the bfloat4_b routed experts)
    experts_math_fidelity: str = "LoFi"
    # fused_experts math fidelity of prefill calls ("" = experts_math_fidelity): HiFi2 — LoFi prefill puts short sparse_dsa
    # prompts below the PCC bar (L3, 31-33 tokens: 0.994995 LoFi vs 0.995045 HiFi2 vs 0.995033 Stage 2 HiFi4;
    # E3/variants/{lofi_default,hifi2_prefill}); decode stays LoFi (DRAM-bound, every decode gate passes)
    experts_math_fidelity_prefill: str = "HiFi2"
    # prefill warm-up once per layer kind and device (C-4)
    warm_once_per_kind: bool = True
    # KDA prefill chunk scan: the per-chunk fp32 head matmuls on the full core grid (default program: 4-16 cores)
    kda_scan_full_grid: bool = True
    # MoE prefill: from this many rows, run the routed experts expert-sorted (the P x top_k (token, expert) pairs sorted
    # by expert, 32-pair fused_experts calls with top-1 routing -> ~1-2 experts streamed per call instead of the ~150-expert
    # union of 32 tokens; Stage 1 C-2). 0 = always the token micro-batch path.
    moe_prefill_sorted_min_rows: int = 0
    # the same threshold inside a traced prefill (capture_prefill_traces): the sorted path's ~P x top_k / 32 calls are
    # host-dispatch bound eagerly (2048 rows: 404 ms eager vs 67 ms traced, token path 326 ms, reorder 225 ms; L4,
    # moe_prefill.jsonl) but the fastest path once captured from 64 rows (traced: 64 rows 10.6 vs 12.3 ms token, 128 rows
    # 15.4 vs 22.5, 256 rows 19.1 vs 39.0; 32 rows 7.5 vs 6.3 -> token path)
    moe_prefill_sorted_min_rows_traced: int = 64
    # fused_experts weight lists bound once in a FusedExpertsWeightSet (no per-call conversion of 2 x 288 tensors)
    experts_weight_set: bool = True
    # MoE prefill token path: run the 32-token fused_experts calls over the chunk's tokens reordered by their top-1
    # expert (stable), so each call's expert union shrinks (L4, 2048 rows: 128.9 -> 85.8 experts per call); the
    # outputs are put back in token order. 0 = off; else from this many rows.
    moe_prefill_reorder_min_rows: int = 512
    # router decode selection: top-k over the choice row padded with -1e9 to this width (a power of two >= 1024 takes
    # ttnn.topk's multi-core bitonic path; 0 = the fused router's router_topk_width on one core)
    router_topk_pad: int = 0
    # KDA decode: inactive-row zeroing / conv-history keep with a pre-broadcast full-width mask instead of a [B,1]
    # broadcast condition
    kda_full_mask: bool = False
    # KDA decode: f_a|g_a|b folded into the q|k|v DRAM-sharded matmul at the q|k|v policy (bfloat8_b HiFi2 instead of
    # the low-rank bf16 HiFi4; Stage 2 review C-7 A/B; decode only, prefill keeps the separate projections)
    kda_lowrank_merge: bool = False
    # KDA decode conv history: the active / inactive select writes straight into the persistent history buffer
    # (elementwise, each output tile depends only on the same input tiles) instead of select + copy
    kda_hist_inplace: bool = True
    # decode mHC tensors (stacked / stream-major streams, coefficients, mixes) in L1 instead of DRAM
    mhc_l1: bool = True
    # one-row fused-path mHC coefficients: "fused" = fused_hyperconnection (single-core FusedPrePost + Sinkhorn);
    # "split_sinkhorn" = mhc_split_sinkhorn (the stream-major path's kernel) + a two-op collapse, mix_streams unchanged
    mhc_coeff: str = "fused"
    # MLA paged latent cache dtype ("bfloat16" = Stage 1/2; "bfloat8_b" = plan B4): prefill casts the latents to it before
    # paged_fill_cache, decode updates with the bf16 latent (paged_update_cache converts on write)
    mla_cache_dtype: str = "bfloat16"
    # traced prefill (capture_prefill_traces / prefill_forward_traced): the bounded bucket set, "pow2" = the attention
    # kind's alignment times powers of two up to the 2048-token chunk (KDA 32..2048: 7 traces, MLA 64..2048: 6)
    prefill_trace_buckets: str = "pow2"
    # tt-perf-report advice (decode rows <= 32), measured (E3 ab_variants.jsonl 23:2xZ):
    # * MLA absorbed-head matmuls ([64 heads, T, 256|512] @ [64, 256|512, 512|256]): explicit K block 8 / 16 is 0..+8 us
    #   per step and HiFi2 leaves the attention block unchanged (0.587 / 0.690 / 0.854 vs 0.585 / 0.691 / 0.854 ms at
    #   B = 1 / 8 / 32) -> default program, HiFi4 (0 / "HiFi4");
    # * stream-major mHC projection (decode rows > 4; 4 batched [T, 4096] @ [4096, 32] fp32): explicit K block 32
    #   instead of the default in0_block_w = 1: L4 B=8 3.652 vs 3.728 ms, B=32 7.862 vs 7.934 ms, PCC 1.000000
    mla_head_block_w: int = 0
    mla_head_fidelity: str = "HiFi4"
    mhc_sm_proj_block_w: int = 32
    # * MLA decode W_uv head matmul: its input (the attention output permuted to [1, 64, B, 512]) in L1 (advice "input 0
    #   in L1" at B <= 16, where the grouped attention calls' concat / permute land in DRAM): L3 attention block -5 /
    #   -6 us at B=8 (0.681 vs 0.686 / 0.687 ms), -2 us at B=1, neutral at B=32, PCC 1.000000 (E3 ab_variants `uvl1`)
    mla_uv_in0_l1: bool = True
    # decode rows > 4: build the stream-major streams [1,4,T,4096] in L1 (the concat output; the stream-major ops keep
    # their input's memory config) instead of the concat's DRAM default (tt-perf-report advice "input 0 in L1"):
    # L4 B=8 3.606 vs 3.651 / 3.649 ms (two base runs), B=32 7.825 vs 7.870 / 7.862 ms, PCC 1.000000
    mhc_sm_l1: bool = True
    # prefill tt-perf-report advice on the traced 2048-token chunk (self-review R4), A/B in E3/ab_prefill.jsonl (traced
    # 2048 / 128-token bucket walls, two base runs; rounds 05:58Z and 06:30Z):
    # * DS prefill 2D programs: per-core M block = ceil(M tiles / core rows) rounded up to a multiple of this (4 = the
    #   sub-block friendly rounding: 2048 rows -> 8 tiles x 8 of the 10 core rows; 1 -> 7 tiles x 10 rows): 1 is
    #   -1.2 ms on the dense 2048 bucket (L0), -0.7 ms L4, neutral L3; same PCC (not bitwise: other blocking)
    prefill_m_block_round: int = 1
    # * fidelity of the DS prefill projections ("" = their decode compute config, HiFi2 / HiFi4): LoFi is no faster
    #   (59.1 vs 59.1-59.9 ms L0) and lowers the PCC (0.99931 vs 0.99985) -> ""
    prefill_proj_fidelity: str = ""
    # * DS prefill 2D programs: largest K block (tiles; bounded by the L1 budget) and the activation moved to L1
    #   interleaved first when it is at most this many MB (0 = never; advice "input 0 in L1"): K block 16 = 8 within
    #   0.1 ms on every kind; L1 input +0.3..0.6 ms (2048) / +0.8 ms (128) on L0 -> 8, 0
    prefill_k_block: int = 8
    prefill_in0_l1_mb: int = 0
    # * traced expert-sorted MoE: the token of every sorted pair computed (pair index mod P) and the routing weights
    #   applied after the inverse permutation, instead of two ttnn.gather calls over the P x k pairs (4.85 ms each at
    #   2048 rows; pure data movement either way, bitwise): -9.5 / -9.7 ms on the L4 / L3 2048 bucket
    moe_sorted_gather_free: bool = True
    # * inverse permutation of the sorted pairs: "sort" (argsort of the permutation, single-core 3.5 ms at 2048 rows) |
    #   "scatter" (arange scattered to it): scatter a further -2.5 / -2.7 ms (L4 / L3, bitwise)
    moe_sorted_inverse: str = "scatter"
    # * composite mHC prefill: pre's 4 stream slices are post's residual streams, post's 4 mixed streams the next pre's
    #   streams (no re-slicing of the same [T, 16384] residual: 12 x ~165 us per 2048 chunk; bitwise): -2.0 ms (2048)
    mhc_prefill_reuse_streams: bool = True
    # * composite mHC prefill sum of squares: "mul" (x * x then sum, Stage 1) | "square" (ttnn.square then sum; one
    #   input read instead of two, bitwise): -0.4 ms L0 / L3 (2048), neutral L4
    mhc_prefill_ss: str = "square"
    # * composite mHC prefill fn projection x [T, 16384] @ fn^T [16384, 24]: 0 = default program; else a 1D program
    #   with in1 multicast (M split over the cores) and this K block: +0.24 / -0.49 / -0.15 ms (noise) -> 0
    mhc_prefill_proj_block_w: int = 0
    # ---- Stage 7a fused streaming programs (doc/ttnn/07a-fused-streaming-layer; per op group). Default ON since
    # 2026-10-19 (numerics vs the CPU oracle / op chain, Stage 3 real-weight gates flags-on L0 / L3 / L4, watcher pass per
    # kind, W1c Tracy KPI rows; contract.md section 13); set False to fall back to the op chain per group.
    # mhc_pre: the decode mHC pre of each sublayer (norm + fn projection + coefficients / Sinkhorn + collapse + the
    # sublayer's input RMSNorm) as ONE multi-core generic_op program (tt/mhc_pre.py) for every decode row count <= 32
    # (fused and stream-major paths); the residual copy the following mix needs comes out of the same program
    fused_mhc_pre: bool = True
    # write the mhc_pre debug taps (tt/mhc_pre.py TAPS) of every call into the program's tap tensor
    fused_mhc_debug: bool = False
    # kda_pre: the KDA decode glue between the q|k|v / low-rank projections and fused_recurrent_kda (low-rank slices,
    # f_b / g_b projections, gates, conv + SiLU, conv-history shift, q / k / v split) as ONE generic_op program
    # (tt/kda_pre.py); row-packed with the history shift up to `fused_kda_rows_max` decode rows, whole tiles above
    fused_kda_pre: bool = True
    fused_kda_rows_max: int = 4
    # (kda_pre's whole-tile mode above fused_kda_rows_max was an opt-in `fused_kda_tiles` flag: W1a measured it break-
    # even / slower at B = 8 / 32 and one user's row deviated at B = 32 (PCC 0.9967 vs the chain, review 2026-10-19), so the
    # flag is removed: rows above fused_kda_rows_max always take the op chain)
    # kda_rec: the KDA decode recurrence (fused_recurrent_kda's arithmetic, state in place) + the sigmoid-gated o
    # RMSNorm (rms_norm / sigmoid multiply / reshapes / typecast at B = 1, sigmoid_gated_rms_norm + permutes at B >= 2)
    # as ONE generic_op program writing o_proj's bf16 input directly (tt/kda_rec.py; head h on core h), decode rows <= 32.
    # Default ON since 2026-10-19 17:12Z: state bitwise vs fused_recurrent_kda, o_proj input PCC 0.999998, Stage 3 gates
    # L4 / L0 8/8 with it on, watcher-clean layer_check L4 / L0
    fused_kda_rec: bool = True
    # kda_rec runs one head per core (64 cores, B items of ~64 us of SFPU work each, serial): traced program vs chain
    # 2026-10-19 16:56Z (dev_kda_rec, L0): B = 1 142.6 vs 168.8 us, 2 206.7 vs 223.0, 4 335.6 vs 311.5, 8 619 vs 475,
    # 32 2304 vs 1561 (the chain spreads the B x 64 items over all cores), so wider steps take the chain
    fused_kda_rec_rows_max: int = 2
    # router_tail: the decode router tail (sigmoid, + bias, top-8, id / score typecasts and slices) as ONE generic_op
    # program reading the router matmul's sharded output directly (tt/router_tail.py; one core per token row)
    fused_router_tail: bool = True
    # moe_post: the post-mix of both sublayers (X'_j = post_j * (y [+ shared expert]) + sum_i C[i][j] X_i) as ONE
    # generic_op program (tt/mhc_post.py); at the layer end it also folds the shared-expert add and writes the flat
    # [1,1,T,16384] DRAM output directly (no to_flat / from_stream_major + copy). Needs fused_mhc_pre (its coefficients)
    fused_moe_post: bool = True

    def role_value(self, field_name: str, role: str):
        for item in filter(None, getattr(self, field_name).split(";")):
            r, v = item.split(":")
            if r == role:
                return int(v)
        return None


# ======================================================================================================================
def _storage_cores(k_tiles: int, n_tiles: int, cap: int = 64) -> int:
    """Largest activation-shard core count <= cap dividing both the K and the N tiles, laid out 8 wide."""
    return next(c for c in range(cap, 0, -1) if k_tiles % c == 0 and n_tiles % c == 0 and (c <= 8 or c % 8 == 0))


def _block_w(k_tiles: int, cores: int, per_worker_n: int, budget_tiles: int = 96) -> int:
    """Largest K block (in tiles) that divides K, is a divisor or a multiple of the per-core K shard, and keeps one in1
    block (block x per-worker N tiles) within `budget_tiles` (L1 double buffer of the weight stream)."""
    per_core_k = k_tiles // cores
    best = 1
    for bw in range(1, k_tiles + 1):
        if k_tiles % bw or (per_core_k % bw and bw % per_core_k):
            continue
        if bw * per_worker_n <= budget_tiles:
            best = bw
    return best


def _grid(cores: int) -> ttnn.CoreGrid:
    return ttnn.CoreGrid(x=cores, y=1) if cores <= 8 else ttnn.CoreGrid(x=8, y=cores // 8)


def _round_block(tiles: int, cores: int, quantum: int = 4) -> int:
    """Per-core block (tiles) covering `tiles` over at most `cores` cores, rounded up to a multiple of `quantum` (<= 4;
    2 for small blocks) so it has useful sub-block divisors; the block count stays <= cores."""
    b = -(-tiles // cores)
    q = min(quantum, 4 if b >= 4 else (2 if b >= 2 else 1))
    return -(-b // q) * q


def _divisor_at_most(n: int, cap: int) -> int:
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def prefill_program(
    grid,
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    out_block_tiles: int = 96,
    max_block_w: int = 8,
    weight_tile_bytes: int = 1088,
    l1_budget: int = 900 * 1024,
    m_round: int = 4,
    n_quantum: int = 4,
    max_subblock: int = 4,
):
    """2D multicast matmul config for a prefill projection [M, K] x [K, N] over the device grid: per-core M / N blocks
    rounded to sub-block friendly sizes, the largest K block <= `max_block_w` dividing K, and the largest output
    sub-block (<= 4 tiles: fp32 destination accumulation) dividing the out block. The output block spans the core's
    whole N range (out_block_w = per_core_N) and is bounded to <= `out_block_tiles` tiles through out_block_h only: with
    the DRAM width-sharded weight (IN1_DRAM_WIDTH_SHARDED reader) an N range split into several out blocks produces
    wrong results (probe_ds_prefill.py: PCC 0.09-0.38 vs the interleaved linear whenever out_block_w < per_core_N, exact
    otherwise)."""
    per_core_m = _round_block(m_tiles, grid.y, m_round)
    per_core_n = _round_block(n_tiles, grid.x, n_quantum)  # n_quantum 1: every grid column busy (TP8 N = 96 tiles)
    out_w = per_core_n
    out_h = _divisor_at_most(per_core_m, max(1, out_block_tiles // out_w))
    # L1: double-buffered in0 (per_core_M x K block, bf16) and in1 (K block x out_w, weight format) blocks, the output block
    # (bf16) and its fp32 partials; the largest K block that fits the budget
    fixed = out_h * out_w * (2048 + 4096)
    fits = lambda bw: fixed + 2 * bw * (per_core_m * 2048 + out_w * weight_tile_bytes) <= l1_budget
    max_block_w = max([d for d in range(1, max_block_w + 1) if k_tiles % d == 0 and fits(d)] or [1])
    sub = max(
        (
            (h, w)
            for h in range(1, max_subblock + 1)
            for w in range(1, max_subblock + 1)
            if h * w <= max_subblock and out_h % h == 0 and out_w % w == 0
        ),
        key=lambda hw: (hw[0] * hw[1], hw[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=_divisor_at_most(k_tiles, max_block_w),
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        out_block_h=out_h,
        out_block_w=out_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


class DSLinear:
    """Decode linear y = x @ W on a DRAM width-sharded weight (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig).

    The weight [K, N] is padded to a whole number of tiles per DRAM bank and reader (zero columns, sliced off the
    output) and width-sharded over the banks; the activation [1, 1, M <= 32, K] is width-sharded over `cores` L1
    storage cores (K / cores columns each); the output is L1 width-sharded over the same cores and returned
    interleaved (`out_memory_config`) or sharded (`out_memory_config=None`)."""

    def __init__(self, device, weight: ttnn.Tensor, *, role: str, cfg: OptimizedDecoderConfig, ckc, dtype):
        k, n = int(weight.shape[-2]), int(weight.shape[-1])
        banks = device.dram_grid_size().x
        readers = cfg.role_value("ds_readers_per_role", role) or cfg.ds_readers
        n_pad = round_up(n, TILE * banks * readers)
        if n_pad != n:
            pad = [(0, 0)] * (len(weight.shape) - 1) + [(0, n_pad - n)]
            weight = ttnn.pad(weight, pad, 0.0)
        kt, nt = k // TILE, n_pad // TILE
        cores = cfg.role_value("ds_cores", role) or _storage_cores(kt, nt)
        if kt % cores or nt % cores:
            raise ValueError(f"{role}: {cores} storage cores do not divide K {kt} / N {nt} tiles")
        per_worker_n = nt // (banks * readers)
        bw = cfg.role_value("ds_block_w", role) or _block_w(kt, cores, per_worker_n)
        shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (k, n_pad // banks),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.weight = ttnn.to_memory_config(
            weight, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)
        )
        self.in_config = ttnn.create_sharded_memory_config(
            (TILE, k // cores),
            _grid(cores),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        try:  # the fork's DRAM-streaming matmul config (reader count / placement kwargs); the megakernel phases
            # read this linear's weight / geometry only, so on tenstorrent/tt-metal the eager program is left unset
            self.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=bw,
                per_core_M=1,
                per_core_N=nt // cores,
                fused_activation=None,
                num_workers_per_dram_bank=readers,
                dram_reader_placement=getattr(_PLACEMENT, cfg.ds_placement),
            )
        except (TypeError, AttributeError):
            self.program = None
        self.role, self.k, self.n, self.n_pad = role, k, n, n_pad
        self.cores, self.block_w, self.readers = cores, bw, readers
        self.ckc, self.dtype = ckc, dtype
        self.ocfg, self.device = cfg, device
        self.grid = device.compute_with_storage_grid_size()
        self._prefill_programs = {}

    def prefill(self, x: ttnn.Tensor, memory_config=DRAM, dtype=None) -> ttnn.Tensor:
        """Prefill-sized x [.., M, K] (any M) through the same DRAM-sharded weight with a 2D multicast program."""
        m_tiles = -(-int(x.shape[-2]) // TILE) * (int(x.shape[-3]) if len(x.shape) > 2 else 1)
        m_round, fid, kb = self.ocfg.prefill_m_block_round, self.ocfg.prefill_proj_fidelity, self.ocfg.prefill_k_block
        prog = self._prefill_programs.get((m_tiles, m_round, kb))
        if prog is None:
            wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}.get(self.weight.dtype, 2048)
            prog = prefill_program(
                self.grid,
                m_tiles,
                self.k // TILE,
                self.n_pad // TILE,
                weight_tile_bytes=wtb,
                m_round=m_round,
                max_block_w=kb,
            )
            self._prefill_programs[(m_tiles, m_round, kb)] = prog
        x_l1 = None
        l1_mb = self.ocfg.prefill_in0_l1_mb
        if l1_mb and x.memory_config().buffer_type == ttnn.BufferType.DRAM:
            if m_tiles * TILE * self.k * 2 <= l1_mb * 1e6:
                x = x_l1 = ttnn.to_memory_config(x, L1)
        ckc = self.ckc
        if fid:
            ckc = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fid),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        y = ttnn.linear(
            x,
            self.weight,
            program_config=prog,
            memory_config=memory_config,
            compute_kernel_config=ckc,
            dtype=dtype or self.dtype,
        )
        if x_l1 is not None:
            ttnn.deallocate(x_l1)
        if self.n_pad != self.n:
            y = ttnn.slice(y, [0] * len(y.shape), list(y.shape)[:-1] + [self.n])
        return y

    def describe(self) -> dict:
        return dict(
            role=self.role,
            K=self.k,
            N=self.n,
            N_padded=self.n_pad,
            storage_cores=self.cores,
            in0_block_w=self.block_w,
            readers=self.readers,
            per_core_N=self.n_pad // TILE // self.cores,
            weight_dtype=str(self.weight.dtype),
        )

    def __call__(self, x: ttnn.Tensor, out_memory_config=L1) -> ttnn.Tensor:
        xs = x if x.memory_config() == self.in_config else ttnn.to_memory_config(x, self.in_config)
        y = ttnn.linear(
            xs,
            self.weight,
            program_config=self.program,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.ckc,
            dtype=self.dtype,
        )
        if xs is not x:
            ttnn.deallocate(xs)
        if out_memory_config is None:
            return y
        out = ttnn.sharded_to_interleaved(y, out_memory_config)
        ttnn.deallocate(y)
        if self.n_pad != self.n:
            t = out.shape[-2]
            out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, t, self.n))
        return out


def _use_ds_weight(module, attr: str, ds: DSLinear, cfg: OptimizedDecoderConfig):
    """Prefill reads the DRAM-sharded weight (padded columns are sliced by every prefill consumer that needs it):
    drop the interleaved copy so the layer keeps one copy of each projection."""
    if cfg.ds_prefill_weights and ds.n_pad == ds.n:
        ttnn.deallocate(getattr(module, attr))
        setattr(module, attr, ds.weight)


class FdTileCacheSource(TileCacheSource):
    """Tile-cache loads bound to an open file descriptor with the manifest's shape / dtype / layout as expectations (P4
    lever B: `ttnn.load_tensor("/proc/self/fd/<n>", expected_shape=, expected_dtype=, expected_layout=)` fails closed
    on a mismatched or swapped file and checks the data-region size before the read). Payload binding assumes one host
    tensor per file (P4 review F2): every tile-cache file is a single unsharded tensor (the decoder runs on a 1x1 mesh
    and replicated tensors are cached unsharded), so the single-shard contract holds here."""

    def _load(self, e: dict, device=None) -> ttnn.Tensor:
        fd = os.open(os.path.join(self.cache_dir, e["path"]), os.O_RDONLY | os.O_CLOEXEC)
        try:
            return ttnn.load_tensor(
                f"/proc/self/fd/{fd}",
                device=device,
                expected_shape=tuple(e["shape"]),
                expected_dtype=getattr(ttnn, e["dtype"]),
                expected_layout=getattr(ttnn, f"{e['layout']}_LAYOUT"),
            )
        finally:
            os.close(fd)

    def device(self, layer_idx, name, mesh_device):
        from models.autoports.zai_org_glm_5_3_flash.tt.weights import cache_dtype

        t = self._load(self._entry(layer_idx, name), mesh_device)
        if t.dtype != cache_dtype(name):
            raise ValueError(f"layers.{layer_idx}.{name}: cache dtype {t.dtype} != dtype map {cache_dtype(name)}")
        return t

    def host(self, layer_idx, name):
        e = self._entry(layer_idx, name)
        return ttnn.to_torch(self._load(e)).reshape(e["shape"])


def _check_expert_weights(source, layer_idx: int, gate_up, down):
    """Fail closed on the routed-expert tensors the fused-decoder loader read (its path loads go through an owned
    descriptor; this adds the manifest's shape / dtype expectation per expert)."""
    if not isinstance(source, TileCacheSource):
        return
    for e, (gu, dn) in enumerate(zip(gate_up, down)):
        for t, nm in ((gu, "gate_up_proj"), (dn, "down_proj")):
            ent = source._entry(layer_idx, f"mlp.experts.{e}.{nm}")
            if list(t.shape) != list(ent["shape"]) or t.dtype != getattr(ttnn, ent["dtype"]):
                raise ValueError(f"layer {layer_idx} expert {e} {nm}: {t.shape} {t.dtype} != manifest {ent}")


# ======================================================================================================================
class OptimizedKDA(FusedKDA):
    """FusedKDA with DRAM-sharded decode projections and L1 decode intermediates."""

    def build_decode(self, cfg: OptimizedDecoderConfig):
        self.ocfg = cfg
        self.mem = L1 if cfg.decode_l1 else DRAM
        if not cfg.ds_decode:
            return
        dev = self.device
        self.ds_qkv = DSLinear(dev, self.w_qkv, role="kda_qkv", cfg=cfg, ckc=self.ckc_proj, dtype=BF16)
        self.ds_lowrank = DSLinear(dev, self.w_lowrank, role="kda_lowrank", cfg=cfg, ckc=self.ckc_fp32, dtype=BF16)
        skip = set(filter(None, cfg.ds_skip_roles.split(",")))
        mk = lambda w, role, ckc, dt: None if role in skip else DSLinear(dev, w, role=role, cfg=cfg, ckc=ckc, dtype=dt)
        self.ds_fb = mk(self.w_fb, "kda_fb", self.ckc_fp32, FP32)
        self.ds_gb = mk(self.w_gb, "kda_gb", self.ckc_fp32, FP32)
        self.ds_o = DSLinear(dev, self.w_o, role="kda_o", cfg=cfg, ckc=self.ckc_proj, dtype=BF16)
        self.ds_qkvlr = None
        if cfg.kda_lowrank_merge:
            merged = ttnn.concat(
                [self.w_qkv, ttnn.typecast(self.w_lowrank, ttnn.bfloat8_b)], dim=-1, memory_config=DRAM
            )
            self.ds_qkvlr = DSLinear(dev, merged, role="kda_qkvlr", cfg=cfg, ckc=self.ckc_proj, dtype=BF16)
            ttnn.deallocate(merged)
        for attr, ds in (("w_qkv", self.ds_qkv), ("w_fb", self.ds_fb), ("w_gb", self.ds_gb), ("w_o", self.ds_o)):
            if ds is not None:
                _use_ds_weight(self, attr, ds, cfg)
        if cfg.ds_prefill_weights:  # prefill slices f_a | g_a | b out of the padded low-rank output
            ttnn.deallocate(self.w_lowrank)
            self.w_lowrank = self.ds_lowrank.weight
        self.kda_pre = KdaPre(self, rows_max=cfg.fused_kda_rows_max) if cfg.fused_kda_pre else None
        self.kda_rec = KdaRec(self) if cfg.fused_kda_rec else None

    def ds_linears(self):
        if not self.ocfg.ds_decode:
            return []
        extra = [self.ds_qkvlr] if getattr(self, "ds_qkvlr", None) is not None else []
        return [d for d in (self.ds_qkv, self.ds_lowrank, self.ds_fb, self.ds_gb, self.ds_o) if d is not None] + extra

    def _ones(self, dim1: int, rows: int, width: int) -> ttnn.Tensor:
        """Cached bf16 ones [1, dim1, rows, width] (operand of the full-width mask broadcast)."""
        cache = self.__dict__.setdefault("_ones_cache", {})
        key = (dim1, rows, width)
        if key not in cache:
            cache[key] = ttnn.ones((1, dim1, rows, width), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return cache[key]

    def _lin(self, ds_name: str, x, w, ckc, dtype):
        if self.ocfg.ds_decode and getattr(self, ds_name, None) is not None:
            return getattr(self, ds_name)(x, self.mem)
        return ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=self.mem, dtype=dtype)

    # ------------------------------------------------------------------------------------------------ prefill
    def _pf(self, ds_name: str, x, w, ckc, dtype):
        """Prefill projection: the DRAM-sharded weight with a 2D multicast program (one weight copy), else the
        interleaved linear."""
        if self.ocfg.ds_decode and self.ocfg.ds_prefill_weights and getattr(self, ds_name, None) is not None:
            return getattr(self, ds_name).prefill(x, DRAM, dtype)
        return ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=DRAM, dtype=dtype)

    def _mm(self, a, b):
        """Chunk-scan fp32 head matmuls ([64, r, c] x [64, c, d]) spread over the full core grid."""
        if not self.ocfg.kda_scan_full_grid:
            return super()._mm(a, b)
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.matmul(
            a, b, compute_kernel_config=self.ckc_fp32, dtype=FP32, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x)
        )

    def _lowrank(self, x):
        lr = self._pf("ds_lowrank", x, self.w_lowrank, self.ckc_fp32, BF16)
        t = x.shape[-2]
        f_a = ttnn.slice(lr, (0, 0, 0, 0), (1, 1, t, self.d))
        g_a = ttnn.slice(lr, (0, 0, 0, self.d), (1, 1, t, 2 * self.d))
        b = ttnn.slice(lr, (0, 0, 0, 2 * self.d), (1, 1, t, 2 * self.d + self.h))
        return f_a, g_a, b

    def _decay(self, f_a):
        f = ttnn.add(self._pf("ds_fb", f_a, self.w_fb, self.ckc_fp32, FP32), self.dt_bias)
        return ttnn.multiply(ttnn.sigmoid(ttnn.multiply(f, self.decay_rate)), self.lower_bound)

    def prefill(self, x, state, slot: int, actual_end, valid_end_f):
        """KDA.prefill (Stage 1's exact chunked path through the batched state) with the projections on the
        DRAM-sharded weights."""
        from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections

        c = x.shape[-2]
        conv_width = 3 * self.width
        qkv = self._pf("ds_qkv", x, self.w_qkv, self.ckc_proj, BF16)
        f_a, g_a, b = self._lowrank(x)
        gate_out = self._pf("ds_gb", g_a, self.w_gb, self.ckc_fp32, BF16)
        g32 = self._decay(f_a)
        g = ttnn.typecast(g32, BF16)
        beta = ttnn.sigmoid(ttnn.typecast(b, FP32))
        carry, s0 = self.read_slot(state, slot)
        selections = ChronologicalSelections(
            ttnn.experimental.kda.chronological_selections(
                self.zero_u32, 0, c, self.h, self.d, self.d, actual_end=actual_end
            )
        )
        qkv_rm = ttnn.to_layout(ttnn.reshape(qkv, (1, c, conv_width)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(qkv)
        new_carry = selections.select_local_final_history(qkv_rm, carry)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            qkv_rm,
            carry,
            *self.taps_prefill,
            self.width,
            self.width,
            self.width,
            program_config=self.conv_program,
            actual_start=self.zero_u32,
            sequence_parallel_axis=0,
            predecessor_carry=carry,
            memory_config=DRAM,
        )
        ttnn.deallocate(qkv_rm)
        y, final_state = self._chunk_recurrence(
            q,
            k,
            v,
            ttnn.reshape(g, (1, c, self.width)),
            ttnn.reshape(beta, (1, c, self.h)),
            s0,
            actual_end,
            g32,
            valid_end_f,
        )
        o = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            y,
            ttnn.reshape(gate_out, (1, c, self.width)),
            self.o_norm,
            self.h,
            epsilon=self.eps,
            memory_config=DRAM,
            compute_kernel_config=self.ckc_kda,
            output_dtype=BF16,
        )
        out = self._pf("ds_o", ttnn.reshape(o, (1, 1, c, self.width)), self.w_o, self.ckc_proj, BF16)
        self.write_slot(state, slot, new_carry, final_state)
        return out

    def read_slot(self, state, slot):
        """slot None = the traced-prefill scratch slot: the conv rows of the slot picked by the device one-hot
        `slot_mask` (as FusedKDA.read_slot) and the persistent scratch recurrent state `state.prefill_S`, which the host
        fills from / writes back to the request's slot around the replay (`OptimizedDecoder.run_prefill_trace`), so a
        prefill trace does not depend on the slot."""
        if slot is not None:
            return super().read_slot(state, slot)
        rows = ttnn.sum(ttnn.where(state.slot_mask, state.C, 0.0), dim=2, keepdim=True)  # [1,3,1,3HK]
        carry = ttnn.to_layout(ttnn.reshape(rows, (1, CONV_K - 1, 3 * self.width)), ttnn.ROW_MAJOR_LAYOUT)
        return carry, state.prefill_S

    def write_slot(self, state, slot, carry, recurrent):
        """slot None: the final recurrent state into the scratch `state.prefill_S` (in place), the conv rows blended
        into the slot_mask slot as FusedKDA.write_slot."""
        if slot is not None:
            return super().write_slot(state, slot, carry, recurrent)
        ttnn.copy(recurrent, state.prefill_S)
        rows = ttnn.reshape(ttnn.to_layout(carry, ttnn.TILE_LAYOUT), (1, CONV_K - 1, 1, 3 * self.width))
        ttnn.copy(ttnn.where(state.slot_mask, rows, state.C), state.C)

    def decode(self, x: ttnn.Tensor, state, active=None) -> ttnn.Tensor:
        """FusedKDA.decode (same ops and numerics policy) with the projections as DRAM-sharded matmuls and the
        intermediates in L1. x [1,1,B,4096] bf16; state updated in place; `active` as in FusedKDA.decode."""
        bsz = x.shape[-2]
        h, d, w = self.h, self.d, self.width
        conv_width = 3 * w
        mem = self.mem
        full = self.ocfg.kda_full_mask and active is not None
        if full:  # [1,1,B,1] -> full-width bf16 masks, built once per step (no broadcast condition in the selects)
            m_x = ttnn.multiply(self._ones(1, bsz, x.shape[-1]), active[1])
            m_h = ttnn.multiply(self._ones(CONV_K - 1, bsz, conv_width), active[1])
        if active is not None:
            x = ttnn.where(m_x if full else active[1], x, 0.0)
        if self.ocfg.ds_decode and self.ds_qkvlr is not None:
            both = self.ds_qkvlr(x, mem)
            qkv = ttnn.slice(both, (0, 0, 0, 0), (1, 1, bsz, conv_width), memory_config=mem)
            lr = ttnn.slice(both, (0, 0, 0, conv_width), (1, 1, bsz, conv_width + 2 * d + h), memory_config=mem)
        else:
            qkv = self._lin("ds_qkv", x, self.w_qkv, self.ckc_proj, BF16)
            lr = self._lin("ds_lowrank", x, self.w_lowrank, self.ckc_fp32, BF16)
        kp = getattr(self, "kda_pre", None)
        rows_ok = kp is not None and bsz <= kp.rows_max
        if kp is not None and rows_ok and active is not None and lr.shape[-1] >= 2 * d + h:
            return self._decode_fused_pre(x, qkv, lr, state, active, bsz)
        f_a = ttnn.slice(lr, (0, 0, 0, 0), (1, 1, bsz, d))
        g_a = ttnn.slice(lr, (0, 0, 0, d), (1, 1, bsz, 2 * d))
        b = ttnn.slice(lr, (0, 0, 0, 2 * d), (1, 1, bsz, 2 * d + h))
        gate_out = self._lin("ds_gb", g_a, self.w_gb, self.ckc_fp32, FP32)
        f = ttnn.add(self._lin("ds_fb", f_a, self.w_fb, self.ckc_fp32, FP32), self.dt_bias)
        sig = [ttnn.UnaryOpType.SIGMOID]
        z = ttnn.multiply(f, self.decay_rate)
        g = ttnn.multiply(z, self.lower_bound if active is None else active[2], input_tensor_a_activations=sig)
        beta = ttnn.multiply(b, 1.0 if active is None else active[0], input_tensor_a_activations=sig, dtype=FP32)

        hist = state.C  # [1,3,B,3HK] bf16 (persistent, DRAM)
        prod = ttnn.multiply(hist, self.taps_hist, dtype=FP32, memory_config=mem)
        # reduce over the 3 history rows; fast_reduce_nc reports the padded tile height as rows -> back to B rows
        red = ttnn.experimental.fast_reduce_nc(prod, dims=[1], memory_config=mem)
        ttnn.deallocate(prod)  # the largest decode temporary ([1,3,32,3HK] fp32): free L1 as soon as it is consumed
        acc = ttnn.slice(red, (0, 0, 0, 0), (1, 1, bsz, conv_width), memory_config=mem)
        if tuple(red.shape) != tuple(acc.shape):  # a full-extent slice (B = 32) is a view of red: keep it
            ttnn.deallocate(red)
        qkv32 = ttnn.typecast(qkv, FP32, memory_config=mem)
        acc2 = ttnn.addcmul(acc, qkv32, self.tap_cur, memory_config=mem)
        ttnn.deallocate(acc)
        ttnn.deallocate(qkv32)
        conv = ttnn.silu(acc2, memory_config=mem)
        ttnn.deallocate(acc2)
        new_hist = ttnn.concat(
            [
                ttnn.slice(hist, (0, 1, 0, 0), (1, CONV_K - 1, bsz, conv_width), memory_config=mem),
                ttnn.reshape(qkv, (1, 1, bsz, conv_width)),
            ],
            dim=1,
            memory_config=mem,
        )
        if active is not None and self.ocfg.kda_hist_inplace:
            ttnn.where(m_h if full else active[1], new_hist, hist, output_tensor=state.C)
        else:
            if active is not None:
                new_hist = ttnn.where(m_h if full else active[1], new_hist, hist, memory_config=mem)
            ttnn.copy(new_hist, state.C)

        ttnn.deallocate(new_hist)
        q = ttnn.slice(conv, (0, 0, 0, 0), (1, 1, bsz, w), memory_config=mem)
        k = ttnn.slice(conv, (0, 0, 0, w), (1, 1, bsz, 2 * w), memory_config=mem)
        v = ttnn.slice(conv, (0, 0, 0, 2 * w), (1, 1, bsz, 3 * w), memory_config=mem)
        ttnn.deallocate(conv)
        ttnn.deallocate(qkv)
        return self._recurrence_and_output(q, k, v, g, beta, gate_out, state, bsz)

    def _recurrence_and_output(self, q, k, v, g, beta, gate_out, state, bsz):
        """fused_recurrent_kda (in-place state) + gated RMSNorm + o_proj (shared by the op chain and kda_pre); with
        fused_kda_rec the recurrence + gated norm are ONE program (Stage 7a kda_rec) for decode rows <= 32."""
        h, d, w = self.h, self.d, self.width
        mem = self.mem
        rec = getattr(self, "kda_rec", None)
        rows_max = min(TILE, getattr(self.fcfg, "fused_kda_rec_rows_max", TILE))
        if rec is not None and bsz <= rows_max and self._rec_inputs_ok(q, k, v, g, beta, gate_out, bsz):
            on = rec(q, k, v, g, beta, gate_out, state.S, rows=bsz, out_memory_config=mem)
            return self._lin("ds_o", on, self.w_o, self.ckc_proj, BF16)
        o, _ = ttnn.experimental.kda.fused_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state.S,
            output_final_state=True,
            use_qk_l2norm=True,
            inplace_state=True,
            memory_config=mem,
        )  # o [B,1,H,V] fp32; the final state is written into state.S

        if self.fcfg.kda_gated_norm_op and bsz >= self.fcfg.kda_gated_norm_op_min_rows:
            tu = round_up(bsz, TILE)
            y = ttnn.reshape(ttnn.permute(o, (2, 1, 0, 3), memory_config=mem), (h, bsz, d))  # [H, B, V] head-major
            gate = ttnn.reshape(ttnn.typecast(gate_out, BF16, memory_config=mem), (1, bsz, w))
            if tu != bsz:
                y = ttnn.pad(y, ((0, 0), (0, tu - bsz), (0, 0)), 0.0, memory_config=mem)
                gate = ttnn.pad(gate, ((0, 0), (0, tu - bsz), (0, 0)), 0.0, memory_config=mem)
            on = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                y,
                gate,
                self.o_norm,
                h,
                epsilon=self.eps,
                memory_config=mem,
                compute_kernel_config=self.ckc_kda,
                output_dtype=BF16,
            )  # [1, Tu, H*V]
            if tu != bsz:
                on = ttnn.slice(on, (0, 0, 0), (1, bsz, w))
            on = ttnn.reshape(on, (1, 1, bsz, w))
        else:
            rows = ttnn.reshape(o, (1, 1, bsz * h, d))
            normed = ttnn.rms_norm(
                rows, weight=self.o_norm_row32, epsilon=self.eps, compute_kernel_config=self.ckc_fp32
            )
            gated = ttnn.multiply(
                normed,
                ttnn.reshape(gate_out, (1, 1, bsz * h, d)),
                input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            )
            on = ttnn.typecast(ttnn.reshape(gated, (1, 1, bsz, w)), BF16)
        return self._lin("ds_o", on, self.w_o, self.ckc_proj, BF16)

    def _rec_inputs_ok(self, q, k, v, g, beta, gate_out, bsz) -> bool:
        """kda_rec's input contract: q / k / v / g / gate_out [1,1,B,H*D] fp32, beta [1,1,B,H] fp32, tile layout,
        interleaved (both the op chain and kda_pre produce this)."""
        wide = (1, 1, bsz, self.width)
        for t, shape in ((q, wide), (k, wide), (v, wide), (g, wide), (gate_out, wide), (beta, (1, 1, bsz, self.h))):
            if (tuple(t.shape) != shape or t.dtype != FP32 or t.layout != ttnn.TILE_LAYOUT
                    or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):  # fmt: skip
                return False
        return True

    def _decode_fused_pre(self, x, qkv, lr, state, active, bsz):
        """Stage 7a: everything between the projections and the recurrence in ONE program (tt/kda_pre.py); at more
        than `fused_kda_rows_max` rows the conv-history shift stays the op chain's (slice + concat + where)."""
        q, k, v, g, beta, gate_out, hist_done = self.kda_pre(qkv, lr, state.C, active, rows=bsz)
        if not hist_done:
            mem, conv_width = self.mem, 3 * self.width
            hist = state.C
            new_hist = ttnn.concat(
                [
                    ttnn.slice(hist, (0, 1, 0, 0), (1, CONV_K - 1, bsz, conv_width), memory_config=mem),
                    ttnn.reshape(qkv, (1, 1, bsz, conv_width)),
                ],
                dim=1,
                memory_config=mem,
            )
            ttnn.where(active[1], new_hist, hist, output_tensor=state.C)
            ttnn.deallocate(new_hist)
        ttnn.deallocate(qkv)
        ttnn.deallocate(lr)
        return self._recurrence_and_output(q, k, v, g, beta, gate_out, state, bsz)


# ======================================================================================================================
class OptimizedMLA(FusedMLA):
    """FusedMLA with DRAM-sharded decode projections (q_a|kv_a, q_b, o) and L1 decode intermediates."""

    def build_decode(self, cfg: OptimizedDecoderConfig):
        self.ocfg = cfg
        self.mem = L1 if cfg.decode_l1 else DRAM
        if not cfg.ds_decode:
            return
        dev = self.device
        self.ds_qa_kva = DSLinear(dev, self.w_qa_kva, role="mla_qa_kva", cfg=cfg, ckc=self.ckc_proj, dtype=BF16)
        self.ds_qb = DSLinear(dev, self.w_qb, role="mla_qb", cfg=cfg, ckc=self.ckc_proj, dtype=BF16)
        self.ds_o = DSLinear(dev, self.w_o, role="mla_o", cfg=cfg, ckc=self.ckc_proj, dtype=BF16)
        for attr, ds in (("w_qa_kva", self.ds_qa_kva), ("w_qb", self.ds_qb), ("w_o", self.ds_o)):
            _use_ds_weight(self, attr, ds, cfg)

    def ds_linears(self):
        return [self.ds_qa_kva, self.ds_qb, self.ds_o] if self.ocfg.ds_decode else []

    def _head_matmul(self, a, b):
        """FusedMLA._head_matmul; decode rows (<= 32) with `mla_head_block_w` / `mla_head_fidelity` set take an explicit
        reuse program (one head per core, the whole N per core, K block `mla_head_block_w`) and that fidelity
        (tt-perf-report advice candidates: in0_block_w=1, HiFi2)."""
        bw, fid = self.ocfg.mla_head_block_w, self.ocfg.mla_head_fidelity
        if a.shape[-2] > TILE or (not bw and fid == "HiFi4"):
            return super()._head_matmul(a, b)
        grid = self.device.compute_with_storage_grid_size()
        ckc = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        if not bw:
            return ttnn.matmul(a, b, compute_kernel_config=ckc, dtype=BF16, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x))
        kt, nt = int(b.shape[-2]) // TILE, int(b.shape[-1]) // TILE
        program = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=_divisor_at_most(kt, bw),
            out_subblock_h=1,
            out_subblock_w=_divisor_at_most(nt, 4),
            per_core_M=1,
            per_core_N=nt,
        )
        return ttnn.matmul(a, b, compute_kernel_config=ckc, dtype=BF16, program_config=program)

    def allocate_cache(self, num_blocks: int) -> ttnn.Tensor:
        return ttnn.zeros(
            (num_blocks, 1, self.block_size, self.kv_lora),
            dtype=getattr(ttnn, self.ocfg.mla_cache_dtype),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=DRAM,
        )

    def prefill(self, x, cache, chunk_table, user_table, chunk_start):
        """MLA.prefill with the latents cast to the cache dtype before paged_fill_cache."""
        q_lat, latent = self._project(x)
        ttnn.experimental.paged_fill_cache(cache, ttnn.typecast(latent, cache.dtype), chunk_table, batch_idx=0)
        o_lat = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_lat,
            cache,
            cache,
            user_table,
            chunk_start_idx_tensor=chunk_start,
            scale=self.scale,
            program_config=self.prefill_program,
            compute_kernel_config=self.ckc_sdpa,
            memory_config=DRAM,
        )
        ttnn.deallocate(q_lat)
        return self._output(o_lat)

    def _lin(self, ds_name: str, x, w):
        if self.ocfg.ds_decode:
            return getattr(self, ds_name)(x, self.mem)
        return ttnn.linear(x, w, compute_kernel_config=self.ckc_proj, memory_config=self.mem, dtype=BF16)

    def _pf(self, ds_name: str, x, w):
        if self.ocfg.ds_decode and self.ocfg.ds_prefill_weights:
            return getattr(self, ds_name).prefill(x, DRAM, BF16)
        return ttnn.linear(x, w, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=BF16)

    def _project(self, x):
        """Prefill projection (FusedMLA._project on the DRAM-sharded weights)."""
        t = x.shape[-2]
        qk = self._pf("ds_qa_kva", x, self.w_qa_kva)
        q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, self.q_lora))
        c = ttnn.slice(qk, (0, 0, 0, self.q_lora), (1, 1, t, self.q_lora + self.kv_lora))
        ttnn.deallocate(qk)
        q_r = ttnn.rms_norm(q_r, weight=self.qa_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        latent = ttnn.rms_norm(c, weight=self.kva_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        q = self._pf("ds_qb", q_r, self.w_qb)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.h, num_kv_heads=0, transpose_k_heads=False, memory_config=DRAM
        )  # [1,64,T,256]
        return self._head_matmul(q, self.w_uk), latent

    def _output(self, o_lat):
        """Prefill output (FusedMLA._output on the DRAM-sharded o_proj weight)."""
        t = o_lat.shape[-2]
        o = self._head_matmul(o_lat, self.w_uv)  # [1,64,T,256]
        if self.fcfg.mla_concat == "nlp2":
            g = self.h // 2
            parts = [
                ttnn.experimental.nlp_concat_heads(ttnn.slice(o, (0, i * g, 0, 0), (1, (i + 1) * g, t, self.dv)))
                for i in range(2)
            ]
            o = ttnn.concat(parts, dim=-1)
        else:
            o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (1, 1, t, self.h * self.dv))
        return self._pf("ds_o", o, self.w_o)

    def _project_decode(self, x):
        t = x.shape[-2]
        qk = self._lin("ds_qa_kva", x, self.w_qa_kva)
        q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, self.q_lora))
        c = ttnn.slice(qk, (0, 0, 0, self.q_lora), (1, 1, t, self.q_lora + self.kv_lora))
        ttnn.deallocate(qk)
        q_r = ttnn.rms_norm(q_r, weight=self.qa_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        latent = ttnn.rms_norm(c, weight=self.kva_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        q = self._lin("ds_qb", q_r, self.w_qb)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.h, num_kv_heads=0, transpose_k_heads=False, memory_config=self.mem
        )  # [1,64,T,256]
        q_lat = self._head_matmul(q, self.w_uk)  # [1,64,T,512]
        return q_lat, latent

    def decode(self, x: ttnn.Tensor, cache: ttnn.Tensor, page_table: ttnn.Tensor, cur_pos: ttnn.Tensor) -> ttnn.Tensor:
        bsz = x.shape[-2]
        q_lat, latent = self._project_decode(x)
        q_lat = ttnn.permute(q_lat, (0, 2, 1, 3))  # [1,B,64,512]
        grid = self.device.compute_with_storage_grid_size()
        shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.kv_lora),
            core_grid=ttnn.num_cores_to_corerangeset(bsz, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        lat = ttnn.to_memory_config(ttnn.reshape(latent, (1, bsz, 1, self.kv_lora)), shard)
        ttnn.experimental.paged_update_cache(cache, lat, update_idxs_tensor=cur_pos, page_table=page_table)
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=0,
            k_chunk_size=(
                self.fcfg.mla_decode_k_chunk_small
                if self.fcfg.mla_decode_k_chunk_small and bsz <= self.fcfg.mla_decode_small_max_rows
                else self.fcfg.mla_decode_k_chunk
            ),
            exp_approx_mode=False,
        )
        heads = self.h if bsz >= self.fcfg.mla_single_call_min_batch else self.decode_heads
        groups = []
        for h0 in range(0, self.h, heads):
            q_g = q_lat if heads == self.h else ttnn.slice(q_lat, (0, 0, h0, 0), (1, bsz, h0 + heads, self.kv_lora))
            o_g = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                q_g,
                cache,
                head_dim_v=self.kv_lora,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos,
                scale=self.scale,
                program_config=program,
                compute_kernel_config=self.ckc_sdpa,
                memory_config=self.mem,
            )
            groups.append(ttnn.slice(o_g, (0, 0, 0, 0), (1, bsz, heads, self.kv_lora)))
        o_lat = groups[0] if len(groups) == 1 else ttnn.concat(groups, dim=2)
        mem = dict(memory_config=L1) if self.ocfg.mla_uv_in0_l1 else {}
        return self._output_decode(ttnn.permute(o_lat, (0, 2, 1, 3), **mem))  # [1,64,B,512]

    def _output_decode(self, o_lat):
        t = o_lat.shape[-2]
        o = self._head_matmul(o_lat, self.w_uv)  # [1,64,T,256]
        if self.fcfg.mla_concat == "nlp2":
            g = self.h // 2
            parts = [
                ttnn.experimental.nlp_concat_heads(ttnn.slice(o, (0, i * g, 0, 0), (1, (i + 1) * g, t, self.dv)))
                for i in range(2)
            ]
            o = ttnn.concat(parts, dim=-1)  # [1,1,T,16384]
        else:
            o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (1, 1, t, self.h * self.dv))
        return self._lin("ds_o", o, self.w_o)


# ======================================================================================================================
class OptimizedSwiGLU(FusedSwiGLU):
    """Shared expert / dense MLP: DRAM-sharded gate|up and down decode matmuls."""

    def build_decode(self, cfg: OptimizedDecoderConfig, prefix: str):
        self.ocfg = cfg
        self.mem = L1 if cfg.decode_l1 else DRAM
        if not cfg.ds_decode:
            return
        dev = self.gate_up.device()
        self.ds_gate_up = DSLinear(dev, self.gate_up, role=f"{prefix}_gate_up", cfg=cfg, ckc=self.ckc, dtype=BF16)
        self.ds_down = DSLinear(dev, self.down, role=f"{prefix}_down", cfg=cfg, ckc=self.ckc, dtype=BF16)
        _use_ds_weight(self, "gate_up", self.ds_gate_up, cfg)
        _use_ds_weight(self, "down", self.ds_down, cfg)

    def ds_linears(self):
        return [self.ds_gate_up, self.ds_down] if self.ocfg.ds_decode else []

    def _prefill(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not (self.ocfg.ds_decode and self.ocfg.ds_prefill_weights):
            return super().__call__(x)
        i = self.intermediate
        gu = self.ds_gate_up.prefill(x, DRAM, BF16)
        shape = list(gu.shape)
        gate = ttnn.slice(gu, [0, 0, 0, 0], shape[:-1] + [i])
        up = ttnn.slice(gu, [0, 0, 0, i], shape)
        act = ttnn.clamped_silu_glu(gate, up, self.limit)
        return self.ds_down.prefill(act, DRAM, BF16)

    def __call__(self, x: ttnn.Tensor, decode: bool = False) -> ttnn.Tensor:
        if not decode:
            return self._prefill(x)
        i = self.intermediate
        if self.ocfg.ds_decode:
            gu = self.ds_gate_up(x, self.mem)
        else:
            gu = ttnn.linear(x, self.gate_up, compute_kernel_config=self.ckc, memory_config=self.mem, dtype=BF16)
        shape = list(gu.shape)
        gate = ttnn.slice(gu, [0, 0, 0, 0], shape[:-1] + [i])
        up = ttnn.slice(gu, [0, 0, 0, i], shape)
        act = ttnn.clamped_silu_glu(gate, up, self.limit)
        if self.ocfg.ds_decode:
            return self.ds_down(act, self.mem)
        return ttnn.linear(act, self.down, compute_kernel_config=self.ckc, memory_config=self.mem, dtype=BF16)


class OptimizedRouter(FusedRouter):
    """FusedRouter with a DRAM-sharded decode logits matmul (bf16 weight, fp32 HiFi4, 288 columns padded to a whole
    number of tiles per bank and reader; padded logits are sliced off before the selection)."""

    def build_decode(self, cfg: OptimizedDecoderConfig):
        self.ocfg = cfg
        self.mem = L1 if cfg.decode_l1 else DRAM
        if cfg.ds_decode:
            import torch

            w = self.weights[self.num_experts]
            self.ds = DSLinear(w.device(), w, role="router", cfg=cfg, ckc=self.ckc, dtype=FP32)
            b = ttnn.to_torch(ttnn.get_device_tensors(self.biases[self.num_experts])[0]).float().reshape(-1)
            pad = torch.full((self.ds.n_pad - self.num_experts,), -1e9)
            self.bias_ds = ttnn.from_torch(
                torch.cat([b, pad]).reshape(1, 1, 1, -1),
                dtype=FP32,
                layout=ttnn.TILE_LAYOUT,
                device=w.device(),
                memory_config=DRAM,
            )
        self.tail = None
        if cfg.ds_decode and cfg.fused_router_tail:
            width = self.fcfg.router_topk_width
            bias320 = (
                ttnn.to_torch(ttnn.get_device_tensors(self.biases[width])[0]).float().reshape(-1)
            )  # [320]: -1e9 on the padded experts
            self.tail = RouterTail(self.weights[self.num_experts].device(), bias320)

    def ds_linears(self):
        return [self.ds] if self.ocfg.ds_decode else []

    def __call__(self, x: ttnn.Tensor, decode: bool = False):
        if not decode or not self.ocfg.ds_decode or self.fcfg.router_impl != "topk":
            return super().__call__(x)
        t = x.shape[-2]
        logits = self.ds(x, None)  # [1,1,B,N_padded] fp32, L1 width-sharded
        if self.tail is not None and not self.ocfg.router_topk_pad and self.fcfg.router_topk_width == 320 and t <= TILE:
            return self.tail(logits, rows=t)  # Stage 7a: the whole tail in one program, straight off the sharded logits
        logits = ttnn.sharded_to_interleaved(logits, self.mem)
        pad_to = self.ocfg.router_topk_pad
        if pad_to:
            bias = self.bias_ds  # padded logit columns (>= num_experts) get -1e9: never selected
        else:
            width = self.fcfg.router_topk_width
            if width != self.ds.n_pad:
                logits = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, t, width))
            bias = self.biases[width]
        # padded columns (>= num_experts) are 0 logits: the -1e9 bias of the padded selection row keeps them out of the
        # top-k, exactly as the fused router's zero-padded weight columns
        scores = ttnn.sigmoid(logits)
        choice = ttnn.add(scores, bias)
        if pad_to and pad_to > choice.shape[-1]:
            choice = ttnn.pad(choice, ((0, 0), (0, 0), (0, 0), (0, pad_to - choice.shape[-1])), -1e9)
        _, idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        if idx.dtype != ttnn.uint16:
            idx = ttnn.typecast(idx, ttnn.uint16)
        scores = ttnn.slice(scores, (0, 0, 0, 0), (1, 1, t, self.num_experts))
        return idx, ttnn.typecast(scores, BF16)


class OptimizedMoE(FusedMoE):
    """FusedMoE with LoFi routed experts and the expert-sorted prefill (moe_prefill_sorted_min_rows)."""

    def _fused_experts(self, x: ttnn.Tensor, **kw) -> ttnn.Tensor:
        """fused_experts with this layer's expert weights: through a FusedExpertsWeightSet built once (the lists are
        converted to C++ at construction, not on every call), or the list API."""
        mod = ttnn._ttnn.operations.experimental
        if self.cfg.experts_weight_set and hasattr(mod, "FusedExpertsWeightSet"):
            ws = self.__dict__.get("_weight_set")
            if ws is None:
                ws = self._weight_set = mod.FusedExpertsWeightSet(self.gate_up, self.down)
            return mod.fused_experts_weight_set(x, weights=ws, **kw)
        return ttnn.experimental.deepseek.moe.fused_experts(
            x, gate_up_weights=self.gate_up, down_weights=self.down, **kw
        )

    def phase_fidelity(self):
        """fused_experts fidelity of the current call: set per phase by __call__ (decode / prefill), the decode policy
        for direct calls of the routed paths (probes)."""
        return getattr(self, "_fid", None) or getattr(ttnn.MathFidelity, self.cfg.experts_math_fidelity)

    def _arange_row(self, n: int):
        """[1, 1, 1, n] uint32 row-major 0..n-1 (the scatter source / base of the inverse permutation), per length."""
        cache = getattr(self, "_arange_cache", None)
        if cache is None:
            cache = self._arange_cache = {}
        if n not in cache:
            import torch

            cache[n] = ttnn.from_torch(
                torch.arange(n, dtype=torch.int32).reshape(1, 1, 1, n),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
        return cache[n]

    def _sorted_consts(self, rows: int):
        """Per prefill length: token id of every (slot j, token p) pair in pair order j * P + p (fp32, exact), and the
        all-ones score row of the top-1 calls."""
        cache = getattr(self, "_sorted_cache", None)
        if cache is None:
            cache = self._sorted_cache = {}
        if rows not in cache:
            import torch

            n = rows * self.top_k
            tok = (torch.arange(n) % rows).float().reshape(1, 1, 1, n)
            mk = lambda t, dt: ttnn.from_torch(
                t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
            )
            ones = torch.ones(1, 1, TILE, self.num_experts)
            cache[rows] = (mk(tok, FP32), mk(ones, BF16))
        return cache[rows]

    def routed_sorted(self, x: ttnn.Tensor, idx: ttnn.Tensor, scores: ttnn.Tensor) -> ttnn.Tensor:
        """Routed experts of a prefill chunk, expert-sorted: the P x k (token, expert) pairs (pair order j * P + p) are
        sorted by expert, their token rows gathered, run through fused_experts in 32-pair calls with top-1 routing (all
        ones scores, scale 1: each pair's output is its expert's FFN of its token), scaled by the pair's routing weight
        (routed_scaling_factor * s / sum of the token's k selected scores, as fused_experts normalises in-kernel), put
        back in pair order and summed over the k pairs of each token. Same expert math as the token path; each call
        streams the ~1-2 experts its 32 sorted pairs select."""
        rows, h, k = x.shape[-2], self.hidden, self.top_k
        n = rows * k
        tok_table, ones = self._sorted_consts(rows)
        # routing weights per pair [1,1,P,k] fp32 (fused_experts' normalisation) and the expert id per pair
        sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
        w = ttnn.multiply(sel, ttnn.reciprocal(ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-20)))
        w = ttnn.multiply(w, self.scaling)
        flat = lambda t: ttnn.reshape(ttnn.transpose(t, -2, -1), (1, 1, 1, n))  # [1,1,P,k] -> [1,1,1,k*P] (j*P+p)
        ids_f = flat(ttnn.typecast(idx, FP32))
        w_f = flat(w)
        sorted_ids, perm = ttnn.sort(ids_f, dim=-1)
        gather_free = self.cfg.moe_sorted_gather_free
        col = lambda t: ttnn.transpose(t, -2, -1)  # [1,1,1,n] -> [1,1,n,1]
        if gather_free:
            # token of sorted pair s = perm[s] mod P (pair order j * P + p): exact in fp32 (integers < 2^24; the
            # quotient of perm + 0.5 is >= 0.5 / P away from an integer, so its floor is exact)
            perm_f = ttnn.typecast(perm, FP32)
            q = ttnn.floor(ttnn.multiply(ttnn.add(perm_f, 0.5), 1.0 / rows))
            tok_sorted = ttnn.subtract(perm_f, ttnn.multiply(q, float(rows)))
        else:
            tok_sorted = ttnn.gather(tok_table, dim=-1, index=perm)  # token of every sorted pair
            w_sorted = ttnn.gather(w_f, dim=-1, index=perm)
            w_col = ttnn.typecast(col(w_sorted), BF16)
        ids_col = ttnn.typecast(col(sorted_ids), ttnn.uint16)
        tok_rm = ttnn.to_layout(ttnn.typecast(ttnn.reshape(tok_sorted, (1, n)), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.to_layout(ttnn.reshape(x, (rows, h)), ttnn.ROW_MAJOR_LAYOUT)
        xs = ttnn.reshape(ttnn.embedding(tok_rm, x_rm, layout=ttnn.TILE_LAYOUT), (1, 1, n, h))
        ttnn.deallocate(x_rm)
        fid = self.phase_fidelity()
        outs = []
        for c0 in range(0, n, TILE):
            sl = lambda t, w_: ttnn.slice(t, (0, 0, c0, 0), (1, 1, c0 + TILE, w_))
            y = self._fused_experts(
                sl(xs, h),
                routing_indices=sl(ids_col, 1),
                routing_scores=ones,
                num_experts=min(self.num_experts, TILE),
                intermediate_size=self.intermediate,
                swiglu_limit=self.limit,
                top_k=1,
                routed_scaling_factor=1.0,
                routing_eps=0.0,
                experts_block_size=1,
                math_fidelity=fid,
            )
            outs.append(ttnn.reshape(y, (1, 1, TILE, h)))
        ttnn.deallocate(xs)
        ys = ttnn.concat(outs, dim=2)  # [1,1,n,h] in sorted order
        if not gather_free:
            ys = ttnn.multiply(ys, w_col)  # weighted in sorted order
        for o in outs:
            ttnn.deallocate(o)
        # back to pair order: the inverse permutation
        if self.cfg.moe_sorted_inverse == "scatter":
            ar = self._arange_row(n)
            perm_rm = ttnn.to_layout(ttnn.typecast(perm, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
            inv = ttnn.scatter(ar, -1, perm_rm, ar)  # inv[perm[s]] = s
        else:
            _, inv = ttnn.sort(ttnn.typecast(perm, FP32), dim=-1)  # argsort(perm)
        inv_rm = ttnn.to_layout(ttnn.typecast(ttnn.reshape(inv, (1, n)), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ys_rm = ttnn.to_layout(ttnn.reshape(ys, (n, h)), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(ys)
        yp = ttnn.embedding(inv_rm, ys_rm, layout=ttnn.TILE_LAYOUT)  # [1, n, h], pair order j*P + p
        ttnn.deallocate(ys_rm)
        if gather_free:
            yp = ttnn.multiply(ttnn.reshape(yp, (1, 1, n, h)), ttnn.typecast(col(w_f), BF16))  # weighted, pair order
        return ttnn.slice(
            ttnn.experimental.fast_reduce_nc(ttnn.reshape(yp, (1, k, rows, h)), dims=[1]), (0, 0, 0, 0), (1, 1, rows, h)
        )

    def routed_reordered(self, x: ttnn.Tensor, idx: ttnn.Tensor, scores: ttnn.Tensor, micro: int, block: int):
        """Token path over the chunk's rows reordered by top-1 expert (stable sort of the fp32 expert id, exact for
        ids < 2^24): the rows gathered into that order and routed again (same per-token selection), the 32-token calls
        run on them, and the outputs gathered back (inverse permutation = argsort of the permutation). Per token the
        same experts, weights and math."""
        rows, h = x.shape[-2], self.hidden
        key = ttnn.transpose(ttnn.typecast(ttnn.slice(idx, (0, 0, 0, 0), (1, 1, rows, 1)), FP32), -2, -1)  # [1,1,1,P]
        _, perm = ttnn.sort(key, dim=-1, stable=True)
        _, inv = ttnn.sort(ttnn.typecast(perm, FP32), dim=-1)
        rm_idx = lambda t: ttnn.to_layout(ttnn.typecast(ttnn.reshape(t, (1, rows)), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        perm_rm, inv_rm = rm_idx(perm), rm_idx(inv)
        gather_rows = lambda t, ix: ttnn.reshape(
            ttnn.embedding(
                ix, ttnn.to_layout(ttnn.reshape(t, (rows, t.shape[-1])), ttnn.ROW_MAJOR_LAYOUT), layout=ttnn.TILE_LAYOUT
            ),
            (1, 1, rows, t.shape[-1]),
        )
        xp = gather_rows(x, perm_rm)
        # routing of the reordered rows: the router is row-independent (per-row matmul / sigmoid / top-k), so every
        # token gets the selection it had in token order (embedding gathers bf16 tables only, not the uint16 ids)
        ip, sp = self.router(xp)
        y = self.routed(xp, ip, sp, micro, block)
        return gather_rows(y, inv_rm)

    def _call(self, x, idx, scores, block: int):
        rows = x.shape[-2]
        return self._fused_experts(
            x,
            routing_indices=idx,
            routing_scores=scores,
            num_experts=self.bound(rows),
            intermediate_size=self.intermediate,
            swiglu_limit=self.limit,
            top_k=self.top_k,
            routed_scaling_factor=self.scaling,
            routing_eps=1e-20,
            experts_block_size=min(block, self.bound(rows)),
            math_fidelity=self.phase_fidelity(),
        )  # [1, rows, H] bf16

    def __call__(self, x: ttnn.Tensor, *, decode: bool, x_router: ttnn.Tensor | None = None) -> ttnn.Tensor:
        return ttnn.add(self._routed_out(x, decode=decode, x_router=x_router), self.shared(x, decode=decode))

    def _routed_out(self, x: ttnn.Tensor, *, decode: bool, x_router: ttnn.Tensor | None = None) -> ttnn.Tensor:
        idx, scores = self.router(x if x_router is None else x_router, decode=decode)
        fid = (
            self.cfg.experts_math_fidelity
            if decode
            else (self.cfg.experts_math_fidelity_prefill or self.cfg.experts_math_fidelity)
        )
        self._fid = getattr(ttnn.MathFidelity, fid)
        if decode:
            micro, block = self.cfg.moe_decode_micro_batch, self.cfg.experts_block_size
        else:
            micro, block = self.cfg.moe_prefill_micro_batch, self.cfg.experts_block_size_prefill
        rows = x.shape[-2]
        sorted_rows, reorder_rows = self.cfg.moe_prefill_sorted_min_rows, self.cfg.moe_prefill_reorder_min_rows
        if getattr(self, "traced_prefill", False):
            sorted_rows = self.cfg.moe_prefill_sorted_min_rows_traced
        if not decode and sorted_rows and rows >= sorted_rows:
            routed = self.routed_sorted(x, idx, scores)
        elif not decode and reorder_rows and rows >= reorder_rows and rows > micro and x_router is None:
            routed = self.routed_reordered(x, idx, scores, micro, block)
        else:
            routed = self.routed(x, idx, scores, micro, block)
        return routed

    def parts(self, x: ttnn.Tensor, *, decode: bool, x_router: ttnn.Tensor | None = None):
        """(routed experts, shared expert) without the final add (Stage 7a: the fused post-mix adds them)."""
        return self._routed_out(x, decode=decode, x_router=x_router), self.shared(x, decode=decode)


def _upgrade(obj, cls, *args):
    """Turn a fused-decoder module into its optimized subclass in place (same weights and attributes; the subclass
    adds the decode-time state built by `build_decode`)."""
    obj.__class__ = cls
    obj.build_decode(*args)
    return obj


# ======================================================================================================================
class FusedMhcBinding:
    """Stage 7a: one sublayer's `mhc_pre` program (tt/mhc_pre.py) bound to its layer. `run` executes the program and
    parks its normalised output for the layer's next `_norm(h, weight)` (the sublayer's RMSNorm, already applied inside
    the program), so every caller of `hc.pre(...)` + `layer._norm(...)` — the layer's own forward and the multichip
    half-layer units alike — runs the fused program without a call-site change."""

    def __init__(self, layer, prog: "_mhc.MhcPre", weight, hn_dtype):
        self.layer, self.prog, self.weight, self.hn_dtype = layer, prog, weight, hn_dtype

    def run(
        self, x, *, t: int, x_mode: int, xout_mode: int, pc_mode: int, keep_last: bool = True
    ) -> "_mhc.MhcPreOutputs":
        out = self.prog(
            x, t=t, x_mode=x_mode, xout_mode=xout_mode, pc_mode=pc_mode, hn_dtype=self.hn_dtype,
            debug=self.layer.ocfg.fused_mhc_debug,
        )  # fmt: skip
        self.layer._mhc_pending = (self.weight, out.hn)
        # (post, coef) of this run for a following hyperconnection post (post_hook); kept only when the fused post-mix
        # exists, and dropped once used
        self.last = (out.post, out.coef) if keep_last and getattr(self.layer, "mhc_post", None) is not None else None
        return out

    def post_hook(self, y, x, post, *, stacked: bool):
        """The fused post-mix for a hyperconnection `post(y, x, post, comb)` call that follows this binding's pre (the
        multichip half-layer units): None (-> today's mix) unless `post` IS the post tensor of this binding's last run
        and the row count fits the program (<= 32 rows; stacked layout <= 4)."""
        mp = getattr(self.layer, "mhc_post", None)
        last = getattr(self, "last", None)
        t = y.shape[-2]
        if mp is None or last is None or post is not last[0] or t > TILE or (stacked and t > 4):
            return None
        self.last = None
        coef = last[1]
        if stacked:
            return mp(y, x, coef, t=t, x_mode=_mpost.X_STACKED, out_mode=_mpost.OUT_STACKED)
        return mp(y, x, coef, t=t, x_mode=_mpost.X_FLAT, out_mode=_mpost.OUT_FLAT, out_shape=tuple(x.shape))


class OptimizedHyperConnection(FusedHyperConnection):
    """Fused mHC (fused_hyperconnection + mix_streams) with the fn projection as (mhc_proj="flat_k32"):
        xn = rms_norm(x) (unweighted, width-sharded multi-core, fp32)  -> m = xn @ fn^T
    where the K = 16384 contraction is split into 512 one-tile chunks: [1, 512, T, 32] (a view of the flat stream rows,
    tile column c -> batch c) x [1, 512, 32, 32] (fn^T rows of chunk c, 24 columns zero-padded) on the full core grid,
    then fast_reduce_nc over the chunks. Same math as the fused decoder's rsqrt(mean(x^2) + eps) * (x @ fn^T)."""

    def build_decode(self, cfg: OptimizedDecoderConfig, mesh_device):
        self.ocfg = cfg
        k = self.width // TILE
        fn_pad = ttnn.pad(self.fn_t, ((0, 0), (0, 0), (0, 0), (0, TILE - self.n_mix)), 0.0)  # [1,1,16384,32]
        self.fn_k32 = ttnn.reshape(fn_pad, (1, k, TILE, TILE))
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.norm_cores = ttnn.CoreGrid(x=8, y=8)
        nc = self.norm_cores.num_cores
        self.xn_shard = ttnn.create_sharded_memory_config(
            (TILE, self.width // nc),
            self.norm_cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        bw = self.width // nc // TILE
        self.xn_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.norm_cores.x, self.norm_cores.y],
            subblock_w=max(sw for sw in (4, 3, 2, 1) if bw % sw == 0),
            block_h=1,
            block_w=bw,
            inplace=False,
        )

    fused = None  # FusedMhcBinding (Stage 7a mhc_pre), set by OptimizedDecoderLayer when ocfg.fused_mhc_pre

    def pre(self, x_flat, x_stk):
        """-> (h [T,1,1,4096] fp32, post [T,1,4,1], comb [T,1,4,4]) as FusedHyperConnection.pre; with
        mhc_coeff="split_sinkhorn" the coefficients come from mhc_split_sinkhorn and h = sum_i pre_i x_i. With the
        fused mhc_pre program (Stage 7a) h is returned already normalised ([1,1,T,4096], the layer's next _norm returns
        it as is)."""
        t, n = x_stk.shape[0], self.n
        if self.fused is not None and t <= TILE:
            src, mode = (x_flat, _mhc.X_FLAT) if x_flat is not None else (x_stk, _mhc.X_STACKED)
            out = self.fused.run(src, t=t, x_mode=mode, xout_mode=_mhc.XOUT_NONE, pc_mode=_mhc.PC_STACKED)
            return out.hn, out.post, out.comb
        # mhc_split_sinkhorn's static CBs (~1.22 MB / core) leave no room for L1 tensors beyond one decode row (mock:
        # CB / L1 clash at T = 2, 4): more rows keep fused_hyperconnection
        if self.ocfg.mhc_coeff != "split_sinkhorn" or t != 1:
            return super().pre(x_flat, x_stk)
        stacked = x_flat is None or self.proj_mode(t) == "stacked"
        m = self.mixes_stacked(x_stk) if stacked else self.mixes(x_flat)  # [1,1,T,24]
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            m, self.consts, n, self.iters, self.hc_eps
        )
        pre = ttnn.reshape(pre, (t, 1, n, 1))
        h = ttnn.sum(ttnn.multiply(x_stk, pre), dim=-2, keepdim=True)  # [T,1,1,4096]
        return h, ttnn.reshape(post, (t, 1, n, 1)), ttnn.reshape(comb, (t, 1, n, n))

    def post(self, y, x_stk, post, comb):
        if self.fused is not None:
            out = self.fused.post_hook(ttnn.reshape(y, (1, 1, y.shape[-2], y.shape[-1])), x_stk, post, stacked=True)
            if out is not None:
                return out
        return super().post(y, x_stk, post, comb)

    def proj_mode(self, t: int) -> str:
        p = self.fcfg.mhc_proj
        if p == "flat_k32":
            # the row-major reshape [1, 1, T, 16384] -> [1, 512, T, 32] is the per-tile-column chunking only for T == 1;
            # more rows fall back to the fused decoder's auto policy
            if t == 1:
                return "flat_k32"
            return "stacked" if t <= self.fcfg.mhc_stacked_max_rows else "ksplit"
        return super().proj_mode(t)

    def mixes(self, x: ttnn.Tensor) -> ttnn.Tensor:
        t = x.shape[-2]
        if self.proj_mode(t) != "flat_k32":
            return super().mixes(x)
        xn = ttnn.rms_norm(
            ttnn.to_memory_config(x, self.xn_shard),
            epsilon=self.norm_eps,
            program_config=self.xn_program,
            memory_config=self.xn_shard,
            compute_kernel_config=self.ckc,
        )
        xn = ttnn.sharded_to_interleaved(xn, L1)
        xk = ttnn.reshape(xn, (1, self.width // TILE, t, TILE))
        mk = ttnn.matmul(xk, self.fn_k32, compute_kernel_config=self.ckc, dtype=FP32, core_grid=self.grid)
        m = ttnn.experimental.fast_reduce_nc(mk, dims=[1], compute_kernel_config=self.ckc)
        return ttnn.slice(m, (0, 0, 0, 0), (1, 1, t, self.n_mix))


# ======================================================================================================================
class OptimizedCompositeHyperConnection(CompositeHyperConnection):
    """Prefill composite mHC (Stage 1 / 2 ops and numerics) with the stream slices shared across a layer
    (mhc_prefill_reuse_streams): `pre_streams` returns pre's 4 tile-aligned column slices of the residual next to
    (h, post, comb) and `post_streams` mixes from them and returns the 4 new streams unconcatenated, so the layer slices
    its input once and its attention output streams not at all (the Stage 1 path slices the same [T, 16384] residual in
    pre and again in post). Same values. Optional: the sum of squares with ttnn.square (mhc_prefill_ss) and the fn
    projection with an explicit in1-multicast program (mhc_prefill_proj_block_w)."""

    ocfg = None  # OptimizedDecoderConfig, set by OptimizedDecoderLayer

    def _project(self, x):
        bw = self.ocfg.mhc_prefill_proj_block_w if self.ocfg is not None else 0
        if not bw:
            return ttnn.matmul(x, self.fn_t, compute_kernel_config=self.ckc, dtype=FP32)
        m_tiles, k_tiles = -(-int(x.shape[-2]) // TILE), int(x.shape[-1]) // TILE
        grid = x.device().compute_with_storage_grid_size()
        cores = min(m_tiles, grid.x * grid.y)
        per_core_m = -(-m_tiles // cores)
        prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=_divisor_at_most(k_tiles, bw),
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=per_core_m,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        return ttnn.matmul(x, self.fn_t, program_config=prog, compute_kernel_config=self.ckc, dtype=FP32)

    def pre_streams(self, x: ttnn.Tensor, xs=None):
        """HyperConnection.pre -> (h, post, comb, xs) with xs = the 4 stream slices of x (made here if not given)."""
        t = x.shape[-2]
        mixes = self._project(x)
        sq = ttnn.square(x) if self.ocfg is not None and self.ocfg.mhc_prefill_ss == "square" else ttnn.multiply(x, x)
        ss = ttnn.sum(sq, dim=-1, keepdim=True)
        ttnn.deallocate(sq)
        inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / x.shape[-1]), self.norm_eps))
        mixes = ttnn.multiply(mixes, inv)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.hc_eps
        )
        r4 = lambda u: ttnn.reshape(u, (1, 1, t, u.shape[-1]))
        pre, post, comb = r4(pre), r4(post), r4(comb)
        xs = xs if xs is not None else self._streams(x)
        coeffs = self._cols(pre, self.n)
        h = ttnn.multiply(xs[0], coeffs[0])
        for s_, c in zip(xs[1:], coeffs[1:]):
            h = ttnn.addcmul(h, s_, c)
        return h, post, comb, xs

    def post_streams(self, y: ttnn.Tensor, xs, post: ttnn.Tensor, comb: ttnn.Tensor) -> list:
        """HyperConnection.post on the residual's stream slices xs -> the 4 new streams (not concatenated)."""
        if y.dtype != FP32:
            y = ttnn.typecast(y, FP32)
        n = self.n
        cmb = self._cols(comb, n * n)
        out = [ttnn.multiply(p, y) for p in self._cols(post, n)]
        for i in range(n):
            for j in range(n):
                out[j] = ttnn.addcmul(out[j], xs[i], cmb[i * n + j])
        return out


# ======================================================================================================================
class OptimizedStreamMajorHyperConnection(StreamMajorHyperConnection):
    """Stream-major mHC (decode rows > 4) whose fn projection takes an explicit reuse program with K block
    `mhc_sm_proj_block_w` (tt-perf-report advice: in0_block_w=1); 0 = the fused decoder's program. With the fused
    mhc_pre program (Stage 7a, `fused`) the whole pre is one program and h comes back normalised."""

    fused = None  # FusedMhcBinding, set by OptimizedDecoderLayer when ocfg.fused_mhc_pre

    def pre(self, x_sm: ttnn.Tensor):
        t = x_sm.shape[-2]
        if self.fused is None or t > TILE:  # the program covers decode rows (<= 32)
            return super().pre(x_sm)
        out = self.fused.run(x_sm, t=t, x_mode=_mhc.X_FLAT, xout_mode=_mhc.XOUT_NONE, pc_mode=_mhc.PC_STREAM_MAJOR)
        return out.hn, out.post, out.comb

    def post(self, y, x_sm, post_sm, comb):
        if self.fused is not None:
            out = self.fused.post_hook(y, x_sm, post_sm, stacked=False)
            if out is not None:
                return out
        return super().post(y, x_sm, post_sm, comb)

    def _sm_project(self, x_sm):
        bw = getattr(self.fcfg, "mhc_sm_proj_block_w", 0)
        if not bw:
            return super()._sm_project(x_sm)
        kt = int(x_sm.shape[-1]) // TILE
        mt = -(-int(x_sm.shape[-2]) // TILE)
        program = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(4, 1),
            in0_block_w=_divisor_at_most(kt, bw),
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=mt,
            per_core_N=1,
        )
        return ttnn.matmul(x_sm, self.fn_sm, compute_kernel_config=self.ckc, dtype=FP32, program_config=program)


class OptimizedDecoderLayer(FusedDecoderLayer):
    """FusedDecoderLayer whose attention / MLP / router modules are the optimized subclasses."""

    def __init__(self, cfg, layer_idx: int, mesh_device, source, *, block_size: int, ocfg: OptimizedDecoderConfig):
        super().__init__(cfg, layer_idx, mesh_device, source, block_size=block_size, fcfg=ocfg)
        self.ocfg = ocfg
        self.mem = L1 if ocfg.decode_l1 else DRAM
        _upgrade(self.attn_hc, OptimizedHyperConnection, ocfg, mesh_device)
        _upgrade(self.ffn_hc, OptimizedHyperConnection, ocfg, mesh_device)
        self.attn_hc_sm.__class__ = self.ffn_hc_sm.__class__ = OptimizedStreamMajorHyperConnection
        self.attn_hc_comp.__class__ = self.ffn_hc_comp.__class__ = OptimizedCompositeHyperConnection
        self.attn_hc_comp.ocfg = self.ffn_hc_comp.ocfg = ocfg
        # mhc_split_sinkhorn constants of the same sublayers (built by the stream-major modules)
        self.attn_hc.consts, self.ffn_hc.consts = self.attn_hc_sm.consts, self.ffn_hc_sm.consts
        _upgrade(self.attn, OptimizedKDA if self.spec.attention == "kda" else OptimizedMLA, ocfg)
        if self.spec.mlp == "dense":
            _upgrade(self.mlp, OptimizedSwiGLU, ocfg, "mlp")
        else:
            _check_expert_weights(source, layer_idx, self.mlp.gate_up, self.mlp.down)
            _upgrade(self.mlp.shared, OptimizedSwiGLU, ocfg, "shared")
            _upgrade(self.mlp.router, OptimizedRouter, ocfg)
            self.mlp.__class__ = OptimizedMoE
        self._mhc_pending = None
        if ocfg.fused_mhc_pre:
            self._build_fused_mhc(source)

    def _build_fused_mhc(self, source):
        """Stage 7a: the attention / FFN sublayers' mhc_pre programs (the host copies of fn, base, scale and the norm
        weight come from the weight source), bound to both the fused (<= 4 rows) and the stream-major hyperconnection
        objects of the same sublayer."""
        cfg, L = self.cfg, self.layer_idx
        n = cfg.hc_mult

        def make(which: str, norm_name: str, weight, hn_dtype):
            scale = source.host(L, f"hc_{which}_scale").float().reshape(-1)
            per_mix = torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)])
            prog = _mhc.MhcPre(
                self.device,
                source.host(L, f"hc_{which}_fn").reshape(n * cfg.hidden_size, (2 + n) * n),
                per_mix,
                source.host(L, f"hc_{which}_base").float().reshape(-1),
                source.host(L, norm_name).float().reshape(-1),
                hc_eps=cfg.hc_eps,
                x_eps=cfg.rms_norm_eps,
                h_eps=cfg.rms_norm_eps,
                iters=cfg.hc_sinkhorn_iters,
                hidden=cfg.hidden_size,
                n=n,
            )
            return FusedMhcBinding(self, prog, weight, hn_dtype)

        ffn_dtype = FP32 if (self.spec.mlp == "moe" and self.fcfg.router_fp32_input) else ACT_DTYPE
        attn = make("attn", "input_layernorm.weight", self.input_norm, ACT_DTYPE)
        ffn = make("ffn", "post_attention_layernorm.weight", self.post_attn_norm, ffn_dtype)
        self.attn_hc.fused = self.attn_hc_sm.fused = attn
        self.ffn_hc.fused = self.ffn_hc_sm.fused = ffn
        self.fused_mhc = (attn, ffn)
        self.mhc_post = _mpost.MhcPost(self.device) if self.ocfg.fused_moe_post else None

    def ds_linears(self) -> list[DSLinear]:
        mods = [self.attn] + ([self.mlp] if self.spec.mlp == "dense" else [self.mlp.shared, self.mlp.router])
        return [ds for m in mods for ds in m.ds_linears()]

    def _norm(self, h, weight, dtype=ACT_DTYPE):
        """Decode rows: width-sharded rms_norm, output to L1 interleaved; prefill: the fused decoder's norm. After a
        fused mhc_pre (Stage 7a) of this sublayer, h IS the program's normalised output: returned as is."""
        pending = self._mhc_pending
        if pending is not None and pending[0] is weight and h.buffer_address() == pending[1].buffer_address():
            self._mhc_pending = None
            if pending[1].dtype != dtype:
                raise ValueError(f"fused mhc_pre produced {pending[1].dtype}, the sublayer asks for {dtype}")
            return pending[1]
        if self.fcfg.norm_sharded and h.shape[-2] <= TILE:
            w_rm = self.input_norm_rm if weight is self.input_norm else self.post_attn_norm_rm
            out = ttnn.rms_norm(
                ttnn.to_memory_config(h, self.norm_shard),
                weight=w_rm,
                epsilon=self.cfg.rms_norm_eps,
                program_config=self.norm_program,
                memory_config=self.norm_shard,
                compute_kernel_config=self.ckc_norm,
            )
            out = ttnn.sharded_to_interleaved(out, self.mem)
            return ttnn.typecast(out, dtype) if out.dtype != dtype else out
        return super()._norm(h, weight, dtype)

    def forward(self, streams: ttnn.Tensor, attention, *, decode: bool) -> ttnn.Tensor:
        """FusedDecoderLayer.forward; decode rows with mhc_l1 run the mHC (fused or stream-major) on an L1 copy of the
        streams, so every mHC intermediate (stacked / stream-major streams, mixes, coefficients, collapsed input, mixed
        streams) stays in L1; the layer output is written back to DRAM ([1, 1, T, 16384] fp32 contract)."""
        if not decode and self.ocfg.mhc_prefill_reuse_streams and self.fcfg.mhc_prefill == "composite":
            return self._prefill_shared_streams(streams, attention)
        if not (decode and self.ocfg.mhc_l1):
            return super().forward(streams, attention, decode=decode)
        t = streams.shape[-2]
        mode = self.fcfg.mhc_decode
        if mode == "auto":
            mode = "fused" if t <= self.fcfg.mhc_auto_fused_max_rows else "stream_major"
        if mode == "composite":
            return super().forward(streams, attention, decode=decode)
        if self.ocfg.fused_mhc_pre and t <= TILE:  # the mhc_pre program serves <= 32 decode rows; wider: the op chain
            return self._forward_fused_mhc(streams, attention, mode)
        n, d = self.cfg.hc_mult, self.cfg.hidden_size
        x = ttnn.to_memory_config(streams, L1)
        if mode == "stream_major":
            sm_mem = dict(memory_config=L1) if self.ocfg.mhc_sm_l1 else {}
            x_sm = ttnn.concat(
                [ttnn.slice(x, (0, 0, 0, i * d), (1, 1, t, (i + 1) * d)) for i in range(n)], dim=1, **sm_mem
            )
            ttnn.deallocate(x)
            h, post, comb = self.attn_hc_sm.pre(x_sm)
            a = attention(self._norm(h, self.input_norm))
            s1 = self.attn_hc_sm.post(a, x_sm, post, comb)
            ttnn.deallocate(x_sm)
            h, post, comb = self.ffn_hc_sm.pre(s1)
            m = self._ffn(h, decode)
            s2 = self.ffn_hc_sm.post(m, s1, post, comb)
            ttnn.deallocate(s1)
            return ttnn.to_memory_config(from_stream_major(s2), DRAM)
        x_stk = to_stacked(x, n)
        h, post, comb = self.attn_hc.pre(x, x_stk)
        ttnn.deallocate(x)
        a = attention(self._norm(ttnn.reshape(h, (1, 1, t, d)), self.input_norm))
        s1 = self.attn_hc.post(a, x_stk, post, comb)
        ttnn.deallocate(x_stk)
        s1_flat = None if self.ffn_hc.proj_mode(t) == "stacked" else to_flat(s1)
        h, post, comb = self.ffn_hc.pre(s1_flat, s1)
        if s1_flat is not None:
            ttnn.deallocate(s1_flat)
        m = self._ffn(ttnn.reshape(h, (1, 1, t, d)), decode)
        s2 = self.ffn_hc.post(m, s1, post, comb)
        ttnn.deallocate(s1)
        return ttnn.to_memory_config(to_flat(s2), DRAM)

    def _forward_fused_mhc(self, streams: ttnn.Tensor, attention, mode: str) -> ttnn.Tensor:
        """Stage 7a whole-layer decode with mhc_pre: the attention pre reads the DRAM streams directly and writes the
        residual in the mix's layout (stacked [T,1,4,4096] up to 4 rows, stream-major [1,4,T,4096] above), so the L1
        copy + to_stacked / slice + concat ops disappear; the FFN pre reads the attention mix's output as is (no
        to_flat). Same mixes (mix_streams / stream-major post) and attention / FFN calls as `forward`."""
        t = streams.shape[-2]
        attn_b, ffn_b = self.fused_mhc
        mp = self.mhc_post if t <= TILE else None
        flat_out = (1, 1, t, self.cfg.hc_mult * self.cfg.hidden_size)
        if mode == "stream_major":
            o = attn_b.run(
                streams,
                t=t,
                x_mode=_mhc.X_FLAT,
                xout_mode=_mhc.XOUT_FLAT,
                pc_mode=_mhc.PC_STREAM_MAJOR,
                keep_last=False,
            )
            a = attention(self._norm(o.hn, self.input_norm))
            if mp is not None:
                s1 = mp(a, o.x_out, o.coef, t=t, x_mode=_mpost.X_FLAT, out_mode=_mpost.OUT_FLAT,
                        out_shape=tuple(o.x_out.shape))  # fmt: skip
            else:
                s1 = self.attn_hc_sm.post(a, o.x_out, o.post, o.comb)
            ttnn.deallocate(o.x_out)
            o = ffn_b.run(
                s1, t=t, x_mode=_mhc.X_FLAT, xout_mode=_mhc.XOUT_NONE, pc_mode=_mhc.PC_STREAM_MAJOR, keep_last=False
            )
            if mp is not None:
                y, y2 = self._ffn_parts(o.hn)
                out = mp(y, s1, o.coef, t=t, x_mode=_mpost.X_FLAT, out_mode=_mpost.OUT_FLAT, y2=y2, out_shape=flat_out,
                         out_memory_config=DRAM)  # fmt: skip
                ttnn.deallocate(s1)
                return out
            m = self._ffn(o.hn, True)
            s2 = self.ffn_hc_sm.post(m, s1, o.post, o.comb)
            ttnn.deallocate(s1)
            return ttnn.to_memory_config(from_stream_major(s2), DRAM)
        o = attn_b.run(
            streams, t=t, x_mode=_mhc.X_FLAT, xout_mode=_mhc.XOUT_STACKED, pc_mode=_mhc.PC_STACKED, keep_last=False
        )
        a = attention(self._norm(o.hn, self.input_norm))
        if mp is not None:
            s1 = mp(a, o.x_out, o.coef, t=t, x_mode=_mpost.X_STACKED, out_mode=_mpost.OUT_STACKED)
        else:
            s1 = self.attn_hc.post(a, o.x_out, o.post, o.comb)
        ttnn.deallocate(o.x_out)
        o = ffn_b.run(
            s1, t=t, x_mode=_mhc.X_STACKED, xout_mode=_mhc.XOUT_NONE, pc_mode=_mhc.PC_STACKED, keep_last=False
        )
        if mp is not None:
            y, y2 = self._ffn_parts(o.hn)
            out = mp(y, s1, o.coef, t=t, x_mode=_mpost.X_STACKED, out_mode=_mpost.OUT_FLAT, y2=y2, out_shape=flat_out,
                     out_memory_config=DRAM)  # fmt: skip
            ttnn.deallocate(s1)
            return out
        m = self._ffn(o.hn, True)
        s2 = self.ffn_hc.post(m, s1, o.post, o.comb)
        ttnn.deallocate(s1)
        return ttnn.to_memory_config(to_flat(s2), DRAM)

    def _ffn_parts(self, h):
        """The FFN sublayer's output as (y, y2) for the fused post-mix: MoE -> (routed experts, shared expert) unsummed
        (the add happens inside mhc_post), dense MLP -> (mlp, None)."""
        if self.spec.mlp != "moe":
            return self.mlp(self._norm(h, self.post_attn_norm), decode=True), None
        if self.fcfg.router_fp32_input:
            x32 = self._norm(h, self.post_attn_norm, dtype=FP32)
            return self.mlp.parts(ttnn.typecast(x32, ACT_DTYPE), decode=True, x_router=x32)
        return self.mlp.parts(self._norm(h, self.post_attn_norm), decode=True)

    def _prefill_shared_streams(self, streams: ttnn.Tensor, attention) -> ttnn.Tensor:
        """FusedDecoderLayer.forward's composite prefill path with the residual sliced once per sublayer input
        (OptimizedCompositeHyperConnection): attention pre / post share the 4 slices of the layer input, the FFN pre /
        post the attention post's 4 streams (concatenated once, for the FFN mixes' projection). Same ops and values
        minus 12 [T, 4096] fp32 slices of a [T, 16384] residual per layer."""
        ahc, fhc = self.attn_hc_comp, self.ffn_hc_comp
        h, post, comb, xs = ahc.pre_streams(streams)
        a = attention(self._norm(h, self.input_norm))
        ttnn.deallocate(h)
        s1 = ahc.post_streams(a, xs, post, comb)
        for t in xs:
            ttnn.deallocate(t)
        x1 = ttnn.concat(s1, dim=-1)
        h, post, comb, _ = fhc.pre_streams(x1, s1)
        ttnn.deallocate(x1)
        m = self._ffn(h, False)
        ttnn.deallocate(h)
        s2 = fhc.post_streams(m, s1, post, comb)
        for t in s1:
            ttnn.deallocate(t)
        out = ttnn.concat(s2, dim=-1)
        for t in s2:
            ttnn.deallocate(t)
        return out

    def _ffn(self, h, decode: bool):
        if self.spec.mlp != "moe":
            return self.mlp(self._norm(h, self.post_attn_norm), decode=decode)
        if self.fcfg.router_fp32_input:
            x32 = self._norm(h, self.post_attn_norm, dtype=FP32)
            return self.mlp(ttnn.typecast(x32, ACT_DTYPE), decode=decode, x_router=x32)
        return self.mlp(self._norm(h, self.post_attn_norm), decode=decode)


# per (device, layer-kind signature): prefill programs already compiled by an earlier layer of the same kind (C-4)
_WARMED_KINDS: set = set()
# the device objects of the signatures above, kept alive so that a closed device's id() is never reused by a later one
# (a reused id would skip the warm-up and let a program compile after a capture; self-review N2)
_WARMED_DEVICES: dict = {}


class OptimizedDecoder(FusedDecoder):
    """Optimized decoder layer with the FunctionalDecoder public contract (see module docstring)."""

    def __init__(self, layer: OptimizedDecoderLayer, hf_config, mesh_device, *, block_size: int = BLOCK_SIZE):
        super().__init__(layer, hf_config, mesh_device, block_size=block_size)
        self.ocfg = layer.ocfg

    @classmethod
    def from_weight_cache(
        cls,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        cache_dir: str | None = None,
        manifest_path: str | None = None,
        block_size: int = BLOCK_SIZE,
        source=None,
        optimized_config: OptimizedDecoderConfig | None = None,
        fused_config=None,  # accepted for loader symmetry with FusedDecoder; must be None or an OptimizedDecoderConfig
    ):
        ocfg = optimized_config or fused_config or OptimizedDecoderConfig()
        if not isinstance(ocfg, OptimizedDecoderConfig):
            raise TypeError(f"OptimizedDecoder needs an OptimizedDecoderConfig, got {type(ocfg).__name__}")
        source = source or FdTileCacheSource(cache_dir, manifest_path)
        layer = OptimizedDecoderLayer(hf_config, layer_idx, mesh_device, source, block_size=block_size, ocfg=ocfg)
        return cls(layer, hf_config, mesh_device, block_size=block_size)

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        block_size: int = BLOCK_SIZE,
        optimized_config: OptimizedDecoderConfig | None = None,
        fused_config=None,
    ):
        source = TorchStateDictSource({layer_idx: state_dict}, hf_config)
        return cls.from_weight_cache(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            block_size=block_size,
            source=source,
            optimized_config=optimized_config or fused_config,
        )

    def _kind_signature(self, cache) -> tuple:
        reach = bucket_for(min(cache.max_seq_len, PREFILL_CHUNK), self.prefill_align)
        return (id(self.device), self.spec.attention, self.spec.mlp, cache.max_batch, reach, repr(self.ocfg))

    def prepare_prefill(self, cache):
        """Compile every prefill program before any decode trace is captured. The programs depend only on the layer
        kind's shapes and configs, so the full bucket sweep runs once per kind and device (Stage 1 C-4); a later layer of
        the same kind only runs the buckets that compile the programs keyed on this layer's weights: fused_experts bakes
        the expert weight addresses into its kernels, and the MoE runs it in two forms — one call on a chunk of <= one
        micro-batch (the KDA 32-row bucket) and per-micro-batch slices of larger chunks — so a MoE layer warms its
        smallest bucket and the smallest bucket above the micro-batch (every other bucket reuses those programs).
        Overwrites slot 0's state/pages (call at setup)."""
        sig = self._kind_signature(cache)
        if not self.ocfg.warm_once_per_kind or sig not in _WARMED_KINDS:
            super().prepare_prefill(cache)
            _WARMED_KINDS.add(sig)
            _WARMED_DEVICES[id(self.device)] = self.device
            return
        reach = bucket_for(min(cache.max_seq_len, PREFILL_CHUNK), self.prefill_align)
        warm = [self.prefill_buckets[0]]
        if self.spec.mlp == "moe":
            micro = self.ocfg.moe_prefill_micro_batch
            above = [p for p in self.prefill_buckets if p > micro and p <= reach]
            if above and above[0] not in warm:
                warm.append(above[0])
        for p in warm:
            x = ttnn.zeros((1, 1, p, 4 * self.cfg.hidden_size), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=self.device)
            self.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=min(p, cache.max_seq_len))
            ttnn.deallocate(self.prefill_chunk(x, cache=cache, user_slot=0))
            ttnn.deallocate(x)
        self.reset_slot(cache, 0)
        ttnn.synchronize_device(self.device)

    def describe(self) -> dict:
        """Chosen per-role decode matmul configs (for the evidence rows)."""
        return {ds.role: ds.describe() for ds in self.layer.ds_linears()}

    def capture_decode_trace(self, streams, *, cache, page_table=None, cur_pos=None):
        """FunctionalDecoder.capture_decode_trace that never leaves the command queue in capture mode: an exception
        inside the capture ends and releases the trace before it propagates (a queue left capturing makes every later
        op fail and the device close wait forever, E3 work_log 2026-10-17 ~20:18Z)."""
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            out = self.decode_forward(streams, cache=cache, page_table=page_table, cur_pos=cur_pos)
        except BaseException:
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.release_trace(self.device, trace_id)
            raise
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        return DecodeTrace(self.device, trace_id, streams, out, cur_pos, page_table)

    # ------------------------------------------------------------------------------------------------ traced prefill
    def allocate_cache(self, *, max_batch: int, max_seq_len: int):
        """FusedDecoder.allocate_cache plus, for KDA, the traced-prefill scratch recurrent state `kda.prefill_S`
        [1, H, K, V] fp32 (persistent, allocated with the cache so it precedes every trace capture)."""
        cache = super().allocate_cache(max_batch=max_batch, max_seq_len=max_seq_len)
        if self.is_kda:
            z = cache.kda.zero_state
            cache.kda.prefill_S = ttnn.zeros(
                tuple(z.shape), dtype=z.dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
            )
        return cache

    def trace_buckets(self, cache) -> tuple[int, ...]:
        """The bounded traced-prefill bucket set reachable with this cache (doc/context_contract.json prefill.traced)."""
        if self.ocfg.prefill_trace_buckets != "pow2":
            raise ValueError(f"unknown prefill_trace_buckets {self.ocfg.prefill_trace_buckets!r}")
        reach = bucket_for(min(cache.max_seq_len, PREFILL_CHUNK), self.prefill_align)
        return tuple(p for p in traced_buckets(self.prefill_align) if p <= reach) or (self.prefill_align,)

    def _scratch_state_in(self, cache, slot: int):
        """KDA: the slot's recurrent state into the traced-prefill scratch state (eager). With one slot the dim-0 slice
        is full-extent, i.e. a view of the state tensor itself: copy the state directly and free nothing."""
        st = cache.kda
        if st.S.shape[0] == 1:
            ttnn.copy(st.S, st.prefill_S)
            return
        s = st.recurrent_of(slot)
        ttnn.copy(s, st.prefill_S)
        ttnn.deallocate(s)

    def _prime_chunk(self, cache, bucket: int):
        self.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=min(bucket, cache.max_seq_len))

    def _traced_body(self, t: PrefillTrace, cache):
        """The captured work of one bucket: the layer over the persistent input (slot-independent: KDA through the
        scratch slot, MLA through the chunk / user page tables and chunk start) and a copy of the output into the
        persistent output buffer (so the trace allocates nothing that outlives its capture)."""
        moe = self.layer.mlp if self.spec.mlp == "moe" else None
        if moe is not None:
            moe.traced_prefill = True  # the MoE takes its traced-prefill path (moe_prefill_sorted_min_rows_traced)
        x = ttnn.to_layout(t.x, ttnn.TILE_LAYOUT, memory_config=DRAM) if t.host_io else t.x
        try:
            y = self.prefill_chunk(x, cache=cache, user_slot=None)
        finally:
            if moe is not None:
                moe.traced_prefill = False
        if t.host_io:
            ttnn.deallocate(x)
            y_rm = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(y)
            y = y_rm
        ttnn.copy(y, t.y)
        ttnn.deallocate(y)

    def prepare_prefill_traces(
        self, cache, buckets: tuple[int, ...] | None = None, host_io: bool = True
    ) -> dict[int, PrefillTrace]:
        """Setup half 1 of traced prefill (DECISIONS 2026-10-17 ~20:20Z): the persistent input / output buffers of every
        bucket and one eager run of each (compiles what the kind's prepare_prefill has not — the bucket programs once
        per kind, this layer's expert program — every lazily built constant, and the eager replay-side programs: the
        slot-indexed KDA state read of every slot, the state write-back). Overwrites slot 0's state / pages.
        Setup order for a generator, so that no buffer allocated after a capture is alive at any replay
        (TT_METAL_TRACE_ALLOC_TRACKING=1 clean): allocate_cache / make_page_table -> prepare_prefill_traces -> warm and
        capture the decode traces (their outputs are allocated inside their own captures) -> capture_prefill_traces.
        Returns {bucket: PrefillTrace} without traces."""
        buckets = tuple(buckets or self.trace_buckets(cache))
        width = 4 * self.cfg.hidden_size
        layout = ttnn.ROW_MAJOR_LAYOUT if host_io else ttnn.TILE_LAYOUT
        mk = lambda p: ttnn.zeros((1, 1, p, width), dtype=FP32, layout=layout, device=self.device, memory_config=DRAM)
        io = {p: PrefillTrace(bucket=p, x=mk(p), y=mk(p), host_io=host_io) for p in buckets}
        for t in io.values():
            self._prime_chunk(cache, t.bucket)
            self._traced_body(t, cache)
        if self.is_kda:
            for slot in range(cache.max_batch):
                self._scratch_state_in(cache, slot)
            ttnn.fill_cache(cache.kda.S, cache.kda.prefill_S, 0)
        self.reset_slot(cache, 0)
        ttnn.synchronize_device(self.device)
        return io

    def capture_prefill_traces(self, cache, traces: dict[int, PrefillTrace]) -> dict[int, PrefillTrace]:
        """Setup half 2: capture one trace per bucket of `traces` (from prepare_prefill_traces, same cache).
        Overwrites slot 0's state / pages."""
        for t in traces.values():
            self._prime_chunk(cache, t.bucket)
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            try:
                self._traced_body(t, cache)
            except BaseException:
                ttnn.end_trace_capture(self.device, tid, cq_id=0)  # never leave the queue in capture mode
                ttnn.release_trace(self.device, tid)
                raise
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            t.trace_id = tid
        self.reset_slot(cache, 0)
        ttnn.synchronize_device(self.device)
        return traces

    def release_prefill_traces(self, traces: dict[int, PrefillTrace]):
        for t in traces.values():
            if t.trace_id is not None:
                ttnn.release_trace(self.device, t.trace_id)
                t.trace_id = None
            ttnn.deallocate(t.x)
            ttnn.deallocate(t.y)

    def prefill_host_chunk(self, rows, trace: PrefillTrace):
        """Host ttnn tensor of a [n <= bucket, 16384] torch chunk (zero-padded to the bucket) in the trace's input
        layout, for `run_prefill_trace(..., chunk=)`."""
        import torch

        p = trace.bucket
        chunk = torch.zeros(p, 4 * self.cfg.hidden_size)
        chunk[: rows.shape[0]] = rows.reshape(rows.shape[0], -1).float()
        layout = ttnn.ROW_MAJOR_LAYOUT if trace.host_io else ttnn.TILE_LAYOUT
        return ttnn.from_torch(chunk.reshape(1, 1, p, -1), dtype=FP32, layout=layout)

    def run_prefill_trace(
        self, t: PrefillTrace, cache, *, user_slot: int, chunk_start: int, n_valid: int, chunk=None
    ) -> ttnn.Tensor:
        """Replay one bucket for a chunk of `user_slot` (non-blocking): optional host chunk [1, 1, bucket, 16384] fp32
        (a host ttnn tensor) into the persistent input, the chunk's device inputs (begin_prefill_chunk), KDA: the
        slot's recurrent state into the scratch slot before and back after the replay (eager, the only slot-indexed
        work). Returns the persistent output buffer (valid until the next replay of this bucket)."""
        if t.trace_id is None:
            raise ValueError(f"bucket {t.bucket} has no captured trace")
        if chunk is not None:
            ttnn.copy_host_to_device_tensor(chunk, t.x)
        self.begin_prefill_chunk(cache, user_slot=user_slot, chunk_start=chunk_start, n_valid=n_valid)
        if self.is_kda:
            self._scratch_state_in(cache, user_slot)
        ttnn.execute_trace(self.device, t.trace_id, cq_id=0, blocking=False)
        if self.is_kda:
            ttnn.fill_cache(cache.kda.S, cache.kda.prefill_S, user_slot)
        return t.y

    def prefill_forward_traced(
        self, streams, *, cache, traces: dict[int, PrefillTrace], user_slot: int = 0, start_pos: int = 0
    ):
        """prefill_forward on the captured bucket traces: streams (torch, [1, seq_len, 4, 4096] or
        [1, 1, seq_len, 16384]) -> torch [1, seq_len, 16384] fp32. Chunks per `plan_traced_chunks` (every chunk a
        bucket; only the last one padded, by less than the alignment)."""
        import torch

        rows = streams.reshape(-1, 4 * self.cfg.hidden_size).float()
        seq_len = rows.shape[0]
        if start_pos + seq_len > cache.max_seq_len:
            raise ValueError(f"prefill end {start_pos + seq_len} > allocated max_seq_len {cache.max_seq_len}")
        outs = []
        for off, n, phys in plan_traced_chunks(seq_len, self.prefill_align, tuple(sorted(traces))):
            t = traces[phys]
            host = self.prefill_host_chunk(rows[off : off + n], t)
            y = self.run_prefill_trace(
                t, cache, user_slot=user_slot, chunk_start=start_pos + off, n_valid=n, chunk=host
            )
            outs.append(ttnn.to_torch(y).float().reshape(phys, -1)[:n])
        return torch.cat(outs).reshape(1, seq_len, -1)
