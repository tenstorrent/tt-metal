# indexer_score — data-movement (reader/writer DMA) analysis

How the reader/writer NoC traffic is hidden under compute for `indexer_score`, what the
bottleneck is, what was done, and what is left. Companion to `INDEXER_COMPUTE_CEILING.md`
(the matmul/compute side) and `INDEXER_OP.md` (op design); measurement how-to in
`INDEXER_PROFILING.md`.

All numbers are **GLX shape, sp_rank 7, 110 cores, V=35010 valid tiles, HiFi2, bf16 DEST
half-sync**, device-kernel time from `test_indexer_score_sp7_math_util` (DMA on = full kernel).

## How to attribute the DMA gap

`test_indexer_score_sp7_math_util[<h>_k_<fmt>]` reports both device-kernel ms and math_util.
- `INDEXER_DMA_OFF=1` → compute ceiling (reader+writer skip NoC, still push/pop CBs).
- `INDEXER_DMA_OFF_READER=1` / `INDEXER_DMA_OFF_WRITER=1` → disable one side (isolate it).
- `INDEXER_READ_Q_OFF` / `_K_OFF` / `_W_OFF` → skip just one input's reads (attribute the reader).
- `INDEXER_QC=<n>` → force a specific QC (bypasses the auto-tune gate) for sweeps.

## The bottleneck is the READER, not the writer

heads8 bfp8 sp7, QC=1 baseline:

| measurement                         | device | over ceiling |
|-------------------------------------|--------|--------------|
| compute ceiling (both DMA off)      | 0.361  | —            |
| reader off (compute + writer)       | 0.368  | +0.007       |
| writer off (compute + reader)       | 0.643  | +0.282       |
| full kernel                         | 0.729  | +0.368       |

The writer's full-width **fast-strip path is already hidden** (+0.007). The reader is the whole
gap (+0.282), plus ~0.08 of reader/writer DRAM contention when both run. **Optimize the reader.**

The reader is **bandwidth-bound, not latency-bound**: deepening cb_k (2→3→4 chunks) made it
*worse* (more L1 pressure + DRAM contention, no latency win). The fix is fewer bytes.

## Win 1 — auto-tune QC to cut redundant K reads (committed)

With QC=1 each K tile is re-read once per q-row-group (~Sqt=20× over: total K reads = V·Dt vs
Tt·Dt unique). Raising QC (q-tile-rows per work unit) reuses each K chunk across QC q-rows in the
matmul → K DRAM traffic ~QC-fold less.

The factory auto-tunes QC up from the requested value by **a single divisor step of Sqt** (to
QC=2 for the GLX shape) when the op is **reader-bound**: `k_tile_bytes > 51·HB·fidelity_passes`
(per-tile K bytes vs ~HB-head matmul work; BH-calibrated). The step must fit L1 (footprint vs
`l1_size_per_core() − 320 KB`; the reserve covers the tracy profiler's per-RISC L1). Compute-bound
cases stay at QC=1 so the multi-row-group masking overhead never regresses them.

**Stop at one step — QC=2 is the K-bandwidth knee.** A second doubling (QC 2→4) does NOT help the
reader (already past the knee) and regresses on the higher compute ceiling + larger resident CBs.
Drift-controlled sp7 sweep:

| case               | QC=1   | QC=2 (chosen) | QC=4   | math_util (QC=2) |
|--------------------|--------|---------------|--------|------------------|
| heads8  bfp8       | 0.732  | **0.491**     | 0.523  | ~49%             |
| heads16 bf16       | 1.354  | **0.890**     | 1.357  | 54.1%            |
| heads16 bfp8       | **0.797** (compute-bound, stays QC=1) | 0.831 | — | — |
| heads64 bfp8/bf16  | **~2.89 / 2.93** (compute-bound, QC=1) | — | — | — |

(An earlier, transient measurement had QC=4 ≈ QC=2 for heads8 and the auto-tune was set to
max-fitting QC; the drift-controlled re-measure showed QC=2 clearly best, so the policy is now a
single step.) bf16 k doubles reader bytes, which is why heads16-bf16 *is* reader-bound and bumps
while heads16-bfp8 does not. Accuracy 41/41 (bf16+bfp8, all sp ranks).

## Where the reader gap is after Win 1 (heads8 bfp8, QC=2)

reader-only 0.454 vs ceiling 0.373 → +0.081 ms still exposed. Per-input (READ_*_OFF), full kernel:

| input | full-kernel saving if skipped | shareable? |
|-------|-------------------------------|------------|
| q     | ~0.070 (0.493→0.423)          | **yes** — every core in a q-row-group reads the identical q-block (~22× redundant across cores) |
| w     | +~0.016 (→0.407 with q)       | **yes** — same (read identically per group) |
| k     | the rest                      | no — each core's k-chunk is distinct |

Beyond QC=2 the reader is **no longer K-bandwidth-bound**; the residual is q/w cross-core
redundancy + fixed per-read cost. So a k-stationary rewrite would *not* help — the next lever is
sharing q/w, not cutting K further.

## What's left — q/w sharing (attempted; needs the right transport)

q (and w) for a q-row-group are read identically by all ~22 cores that share it (~22× redundant).
Measured upper bound at QC=2 (full kernel with those reads skipped): Q off → 0.423, **Q+W off →
0.407** — only 0.034 above the 0.373 ceiling. So q+w sharing would take heads8 bfp8 ~0.49 → ~0.41
(another ~17%) and **nearly fully hide the reader**.

**Tried — parallel-pull unicast (env `INDEXER_QW_UNICAST`, reverted):** each group's first core
(injector) reads q/w from DRAM and the group's other cores pull it from the injector's L1. It
**regressed 3.7× (0.49→1.9 ms)**: the injector is `c_lo(g)`, the straddle core that first *receives*
group g−1, so the per-group q-reads serialize into a chain across all groups instead of running in
parallel. **Double-buffering cb_q/cb_w was inert** (0.490 vs 0.492 — the q/w cost is DRAM-bandwidth
redundancy, not a boundary stall).

**Fix for a working version:** the injector must be a core *fully inside* g (e.g. `c_lo(g)+1`),
which has no g−1 work and reads g's q/w at kernel start — so the groups pipeline. And the transport
should be **multicast** (one broadcast) not parallel-pull (which also contends ~22-way on the
injector's L1). The output-stationary flat deal lays a group's cores out as a contiguous row-major
run (≈2 rows on the 11×10 grid) — not a clean rectangle — so mcast needs ≤3 rectangles, or a
deal aligned to whole grid rows.
