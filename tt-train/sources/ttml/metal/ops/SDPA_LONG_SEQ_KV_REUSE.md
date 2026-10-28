# SDPA at long sequence length: reuse K/V (and Q/dO) across heads and rows

## Problem

The ttml SDPA kernels (`sdpa_fw`, `sdpa_bw_q`, `sdpa_bw_kv`) distribute work as **one Q tile-row of
one head** (fw, bw-Q) or **one K/V tile-row** (bw-KV) per work item. Each item streams its whole
causal counterpart from DRAM:

* fw / bw-Q: for Q row `i` of head `h`, read K/V rows `0..i`.
* bw-KV: for K/V row `k`, read Q, dO, lse, u rows `k..St-1` of **all** heads of the group.

Per item the FLOP/byte ratio is fixed (~16 tile-matmuls per 32 KB of K/V), far below what the chip
needs to be compute-bound, so the kernels are DRAM-bound at any length *in principle*. It only
matters when there is enough work per core that DRAM time (quadratic in `S`) dominates launch and
pipeline-fill overheads — in practice `S >= ~2k`. At `S = 32k` (Qwen3.8-27B, TP=4: `Q[1,6,S,256]`,
`K/V[1,1,S,256]`) the picture per call was:

| kernel | DRAM traffic | time | effective BW | FPU util |
|--------|-------------:|-----:|-------------:|---------:|
| fw     | ~103 GB      | 266 ms | ~390 GB/s  | ~8 %     |
| bw-Q   | ~103 GB      | 258 ms | ~400 GB/s  |          |
| bw-KV  | ~126 GB      | 328 ms | ~385 GB/s  |          |

The 103 GB are `6 heads x (St^2/2) K/V tile-rows x 32 KB` — the six GQA heads of a group re-read
the identical K/V rows six times, and every Q row re-reads the prefix. K and V themselves are 34 MB.

## Fix

Make the work item fatter so the streamed operand is used several times per read:

**fw / bw-Q — `GROUPED_HEADS`.** Item = `(batch, kv_head, sub-group of d Q heads, Q row)`.
Reader stages `d` Q rows (bw-Q: also dO, O, lse) once, then streams each K/V chunk **once**;
compute loops the `d` heads per chunk with per-head state (fw: running max / sum / O in small FIFO
"ring" CBs of `d+1` entries, rescale fused into the PV matmul; bw-Q: `d` dQ accumulators in one
fp32 region written with out-of-order L1-accumulate packs). K/V traffic drops `d x`; FLOPs are
unchanged; per-head arithmetic is identical (output bit-identical, grads within bf16 rounding).

**bw-KV — `GROUPED_KV_ROWS`.** The kernel already loops all heads inside a K/V row, so the reuse
axis is rows: item = `R` consecutive K/V rows. `R` K/V rows and `R` dK/dV accumulators stay
resident; the causal Q/dO/lse/u suffix is streamed once per item. Traffic drops `~R x`.

**Gating (why short sequences are unaffected).** Fatter items mean fewer items; at small `S` that
under-fills the 110-core grid. `metal/common/sdpa_grouped_heads.hpp` picks `d` / `R` per call with a
small cost model fitted on p150 (`pick_heads_per_item`, `pick_kv_rows_per_item`): it minimises
`max(DRAM time, busiest-core compute)` over the divisors, requires the CBs to fit in L1, and falls
back to the legacy layout (`d = R = 1`, same kernels, same code path) whenever grouping does not win.
Grouping is only used in the balanced-causal path. Overrides for A/B: `TTML_SDPA_DISABLE_GROUPED_HEADS=1`,
`TTML_SDPA_HEADS_PER_ITEM=<d>`, `TTML_SDPA_KV_ROWS_PER_ITEM=<R>`.

## Results

Standalone (B=1, Hq=6, Hkv=1, D=256, host-timed, legacy -> new):

| S   | fw            | bw (dQ + dK/dV) | picked        |
|-----|---------------|-----------------|---------------|
| 2k  | 1.3 -> 0.8 ms | 3.9 -> 3.4 ms   | d=2, R=1      |
| 4k  | 4.7 -> 2.0 ms | 11.8 -> 9.6 ms  | d=6, R=1      |
| 8k  | 17.1 -> 7.4 ms| 43.5 -> 23.8 ms | d=6, R=2      |

32k, 4-layer Qwen3.8 LoRA step (Tracy, per chip): see `SDPA_LONG_SEQ_KV_REUSE_32k.md` / below.

## Files

* `metal/common/sdpa_grouped_heads.hpp` — gate / cost model.
* `sdpa_fw/device/kernels/compute/sdpa_fw_grouped_utils.hpp`, `sdpa_fw_compute_kernel.cpp`,
  `dataflow/sdpa_fw_{reader,writer}_kernel.cpp`, `sdpa_fw_program_factory.cpp`.
* `sdpa_bw/device/kernels/compute/sdpa_bw_grouped_utils.hpp`, `sdpa_bw_{q,kv}_compute_kernel.cpp`,
  `dataflow/sdpa_bw_{q,kv}_{reader,writer}_kernel.cpp`, `sdpa_bw_{q,kv}_program_factory.cpp`.

## Not addressed here

* `Slice.backward` (`ttml/models/deepseek/autograd_ops.py`) materialises a full-size zero gradient
  with host-side `ttnn.zeros` + `concat` per loss chunk: ~3.3 s/step of device idle at 32k with
  `--loss-chunk 2048`. Use `autograd_split` or a device-side pad instead.
* Further SDPA headroom once DRAM-bound no longer: multi-Q-row items in fw/bw-Q (reuse K/V across
  rows too), and K/V chunking in bw-Q/bw-KV (currently one tile-row per NoC barrier).
