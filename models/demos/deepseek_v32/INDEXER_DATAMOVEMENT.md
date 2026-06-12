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

The factory auto-tunes QC up from the requested value to the largest divisor of Sqt whose
resident CBs fit L1 (footprint vs `l1_size_per_core() − 320 KB`; the reserve covers the tracy
profiler's per-RISC L1 so the perf test still builds), **gated on the op being reader-bound**:
`k_tile_bytes > 51·HB·fidelity_passes` (per-tile K bytes vs ~HB-head matmul work; BH-calibrated).
Compute-bound cases stay at QC=1 so the multi-row-group masking overhead never regresses them.

| case               | QC   | before → after        | math_util       |
|--------------------|------|-----------------------|-----------------|
| heads8  bfp8       | 1→4  | 0.729 → **0.523** ms  | 33.1 → 46.1%    |
| heads16 bf16       | 1→2  | 1.324 → **0.892** ms  | 36.5 → 54.1%    |
| heads16 bfp8       | 1    | 0.795 ms (unchanged)  | compute-bound   |
| heads64 bfp8/bf16  | 1    | ~2.89 / 2.93 (unchgd) | compute-bound   |

QC>1 raises the compute ceiling a few % (heads8 0.361→0.386); the reader-bound gate is what keeps
that from regressing the compute-bound cases (QC=2 *regressed* heads16 bfp8 0.795→0.831). bf16 k
doubles reader bytes, which is why heads16-bf16 *is* reader-bound and bumps while heads16-bfp8
does not. Accuracy 41/41 (bf16+bfp8, all sp ranks).

## Where the reader gap is after Win 1 (heads8 bfp8, QC=4)

reader-only 0.498 vs ceiling 0.386 → +0.112 ms still exposed. Per-input (READ_*_OFF):

| input | exposed | shareable? |
|-------|---------|------------|
| q     | 0.059   | **yes** — every core in a q-row-group reads the identical q-block (~22× redundant across cores) |
| w     | 0.020   | **yes** — same (read identically per group) |
| k     | 0.031   | no — each core's k-chunk is distinct |

Note QC2→QC4 barely moved reader-only (0.503→0.498): beyond QC=2 the reader is **no longer
K-bandwidth-bound**; the residual is q/w cross-core redundancy + fixed per-read cost. So a
k-stationary rewrite would *not* help — the next lever is sharing q/w, not cutting K further.

## What's left — q/w multicast (analyzed, not yet implemented)

q (and w) for a q-row-group are read identically by all ~22 cores that share that group, ~22×
redundant (~29 MB q + ~7 MB w vs ~10 MB k at QC=4). Multicasting them (one core reads from DRAM,
broadcasts L1→L1 to the group's cores; receivers wait on a semaphore — the `ring_joint_sdpa`
`chain_link.hpp` idiom) would cut that DRAM traffic ~22×.

Measured upper bound (full kernel with those reads skipped entirely, the best a perfect mcast
could do minus mcast overhead): Q off → 0.449, **Q+W off → 0.420** — only 0.034 above the 0.386
compute ceiling. So q+w sharing would get heads8 bfp8 from 0.524 to ~0.42 ms (a further ~20%) and
**nearly fully hide the reader** (only k's 0.031 + a tiny residual would remain).

Complexity/risk: the output-stationary flat deal lays a group's cores out as a **contiguous
row-major run, not a rectangle**, so per-group mcast needs ≤3 rectangles (head partial row / full
rows / tail partial row) plus boundary cores that belong to two groups — intricate and
deadlock-prone. A mcast-friendly rectangular per-group deal, or a store-and-forward unicast chain
along the contiguous run, are the two viable shapes.
