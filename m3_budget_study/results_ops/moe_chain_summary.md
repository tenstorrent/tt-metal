# P0-B: the MoE chain on real routing, (2,4) stage, EP=8

Inputs:

* `per_op.csv` and the per-chip zones in `profiles/<run>/per_device.json`, from the P0-A zone profiles. These are
  layers 0-6 contiguous with real M3 tokens, 1D fabric, v1 dispatch/combine, `M3_MOE_W_NDSHARD=1` and
  `M3_MOE_HYBRID_THRESHOLD=128`.
* `load_skew.csv` and `load_skew_join.csv` (`M3_MOE_LOAD_STATS=1`).
* `bench/experts.csv`, `bench/experts_fit.txt` and `bench/moe_reduce.csv`.

All numbers are sparse layers 3-6 averaged unless a layer is named. "Single" means one request at h = 139,264
(W = 4096: 2048 tokens per chip; W = 8192: 4096 tokens per chip).

**Mesh mapping.** The P0-B chip order is mesh (row, col) row-major. Its device ids are 0, 4, 28, 24 / 1, 5, 29, 25.
This order was inferred by matching the per-chip routed-token counts in `load_skew.csv` to the per-device
`experts_mm` zone time. Correlation over the 24 (run, layer) points is 0.967; the next-best permutation scores
0.918. Rows are SP ranks (axis 0) and columns are TP ranks (axis 1).

## Per-op chip times (worst / mean / min ms)

| op | W=4096 prose | W=4096 code | W=4096 packed (2 seg) | W=8192 prose | W=8192 code | W=8192 packed (4 seg) |
|---|---|---|---|---|---|---|
| router | 0.188 / 0.187 / 0.185 | 0.189 / 0.187 / 0.185 | 0.188 / 0.186 / 0.185 | 0.353 / 0.349 / 0.346 | 0.354 / 0.349 / 0.346 | 0.353 / 0.349 / 0.346 |
| dispatch | 1.138 / 0.838 / 0.719 | 0.911 / 0.775 / 0.716 | 0.964 / 0.803 / 0.727 | 2.276 / 1.682 / 1.426 | 1.762 / 1.533 / 1.426 | 1.812 / 1.569 / 1.446 |
| experts | 2.759 / 1.865 / 1.444 | 2.378 / 1.934 / 1.628 | 2.387 / 1.964 / 1.658 | 4.238 / 2.563 / 1.764 | 3.529 / 2.590 / 2.039 | 3.342 / 2.612 / 2.087 |
| combine | 2.022 / 0.828 / 0.377 | 1.160 / 0.579 / 0.342 | 1.255 / 0.653 / 0.368 | 3.808 / 1.535 / 0.650 | 2.240 / 1.103 / 0.586 | 2.374 / 1.233 / 0.685 |
| moe_reduce | 2.652 / 2.011 / 0.828 | 1.894 / 1.440 / 0.836 | 1.808 / 1.362 / 0.837 | 5.123 / 3.910 / 1.643 | 3.659 / 2.776 / 1.634 | 3.222 / 2.398 / 1.652 |
| chain, dispatch to moe_reduce: max / min over chips | 5.63 / 5.45 | 4.80 / 4.66 | 4.83 / 4.70 | 9.85 / 9.48 | 8.14 / 7.79 | 7.95 / 7.63 |
| balanced chain estimate | 3.79 | 3.83 | 3.90 | 6.28 | 6.24 | 6.40 |
| imbalance cost of the chain | 1.84 (33%) | 0.97 (20%) | 0.94 (19%) | 3.57 (36%) | 1.90 (23%) | 1.55 (20%) |

* **Chain.** The chain is the per-chip sum of dispatch, experts, combine and moe_reduce. It is nearly the same on
  all 8 chips (a spread of 0.13-0.37 ms): the chain ends at a synchronising collective, so each chip's skew turns
  into waiting inside the ops.
* **Balanced chain estimate.** min dispatch + mean experts + min combine + min moe_reduce.
* **Imbalance cost.** chain max minus the balanced estimate.

## Expert load per chip (routed token-expert assignments, `load_skew.csv`)

Max/mean over the 8 chips:

| case | L3 | L4 | L5 | L6 | mean | hottest expert's share (mean of layers) |
|---|---:|---:|---:|---:|---:|---:|
| (a) single prose, h=0 | 2.03 | 2.70 | 1.87 | 1.61 | 2.05 | 13.3% |
| (a) single prose, h=139k | 2.03 | 2.86 | 1.88 | 1.84 | 2.15 | 14.4% |
| (b) single code, h=0 | 1.36 | 1.31 | 1.37 | 2.39 | 1.61 | 10.0% |
| (b) single code, h=139k | 1.36 | 1.86 | 1.35 | 2.15 | 1.68 | 10.1% |
| (c) packed W=4096, prose@141k + code@0 | 1.63 | 1.90 | 1.29 | 1.65 | 1.62 | 8.7% |
| (c) packed W=8192, 4 segments | 1.64 | 1.61 | 1.30 | 1.57 | 1.53 | 8.8% |

* **Packing.** Packing reduces the skew against prose, from 2.1 to 1.5-1.6, but not against code (1.65).
  Layer 4 on prose is the worst case: one expert takes 22-23% of all assignments, and chip (0,2) gets 2.9x the
  mean.
* **Active experts.** All 16 local experts are active on almost every chip. The worst chip has 13-16 active.

## Anomaly A: combine and moe_reduce, worst chip against mean (up to about 3x)

### Which chip is worst

This is counted over 24 (run, layer) points with load data: W=4096 prose and code at h 0 and 139k, and both
packed runs.

| worst chip of ... | is ... | count |
|---|---|---:|
| experts | the chip with the most routed tokens (the hot chip) | 20 / 24 |
| combine | the hot chip's SP partner (same column, other row) | 21 / 24 |
| combine | the hot chip itself | 0 / 24 |
| moe_reduce | a chip in the hot chip's column | 0 / 24 |
| moe_reduce (min chip) | a chip in the hot chip's column | 22 / 24 |

So the worst combine and moe_reduce chips are not the loaded chips. They are chips that wait.

### Example: layer 4, prose, W=4096, h=139k

Chip (0,2) is hot with 5,862 assignments, against a mean of 2,048. `post_combine_reduce` and the RS are the two
ops inside the moe_reduce zone; ReshapeView adds a further 0.016 ms.

| dev (r,c) | tokens | dispatch | experts | combine | post_combine_reduce | moe_reduce RS | chain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (0,0) | 1234 | 0.705 | 1.811 | 0.341 | 0.362 | 3.187 | 6.42 |
| 4 (0,1) | 3341 | 0.792 | 2.312 | 0.507 | 0.361 | 2.464 | 6.45 |
| **28 (0,2)** | **5862** | 1.566 | **3.126** | 1.036 | 0.369 | **0.471** | 6.58 |
| 24 (0,3) | 1334 | 0.727 | 1.472 | 0.474 | 0.366 | 3.634 | 6.69 |
| 1 (1,0) | 1038 | 0.707 | 1.325 | 0.848 | 0.363 | 3.189 | 6.45 |
| 5 (1,1) | 2007 | 0.801 | 1.844 | 1.039 | 0.365 | 2.486 | 6.55 |
| 29 (1,2) | 870 | 1.555 | 1.270 | **2.913** | 0.365 | **0.472** | 6.59 |
| 25 (1,3) | 698 | 0.703 | 1.655 | 0.289 | 0.363 | 3.637 | 6.66 |

W=8192, same layer: `post_combine_reduce` takes 0.73 ms on every chip. The RS takes 0.90 ms in column 2 and
4.9-7.5 ms elsewhere.

### What the zone time contains

* **What is measured.** A zone is the sum of `DEVICE KERNEL DURATION` over its ops on that chip, from the first
  core's kernel start to the last core's end. There is no host sync between ops, so an op starts on a chip as soon
  as that chip's previous op ends. Any cross-chip semaphore wait inside the kernel is therefore charged to the
  early chip.
* **combine on axis 0** (`combine(..., cluster_axis=0)`) waits in two places:
  * The writer does an init handshake, `noc_semaphore_wait(init_sem_ptr, combine_devices - 1)` in
    `writer_combine.cpp:257`. It waits for its SP peer to enter combine, which that peer does only after its own
    experts finish.
  * An exit handshake, `writer_combine.cpp:381`, follows the data writes.
  * The result: the hot chip's SP partner sits in combine for as long as the hot chip is still in experts
    (2.91 ms against its own 0.3-1.0 ms of work).
* **moe_reduce** is three ops: ReshapeView, `post_combine_reduce` and `reduce_scatter_minimal_async` on
  **axis 1** (`TtMiniMaxReduce`, `cluster_axis=1`, ping-pong/barrier semaphores).
  * `post_combine_reduce` is flat on every chip (0.36 ms at 2048 tokens per chip).
  * The RS is 0.44-0.47 ms on the hot column and 1.2-3.6 ms on the others. It waits for the row peer in the hot
    column, which reaches it late through the hot chip's experts and its partner's combine.
* **dispatch** (axis 0) is higher on both chips of the hot column (1.56 against 0.70-0.80 ms). That column moves
  more tokens. The extra is load-proportional traffic, not a wait: both chips of the pair pay it.

### Comparison with the bench

The bench (`bench/moe_reduce.csv`, tokens replicated, about 25% local slots) measures moe_reduce at 2048 tokens
per chip:

* `post_combine_reduce` (fused) = 0.367 ms
* RS = 0.438-0.493 ms
* fused + RS = 0.805-0.851 ms

In the model at W=4096, `post_combine_reduce` takes 0.36-0.37 ms on every chip. On the hot column, fused + RS
comes to 0.83-0.85 ms, and the moe_reduce min-chip time is 0.83 ms (layers 3-6 mean). That is the bench, exactly.
At W=8192 the min is 1.64 ms, against the bench's 1.61-1.68 ms at 4096 tokens per chip.

The zone's 4.02 ms worst and 2.82 ms mean (layer 4) are therefore 0.85 ms of kernel plus RS wait. On the worst
chip the wait is 3.2 ms; on the mean chip it is 2.0 ms.

### How to attribute it

* **Kernel time** for combine and moe_reduce is the min-chip time, which equals the bench. For W=4096 prose that is
  combine 0.38 ms and moe_reduce 0.83 ms. For W=8192 it is 0.65 and 1.64 ms.
* **Imbalance.** Everything above the min chip is expert-load imbalance, and belongs to routing and experts, not to
  combine or moe_reduce. In a per-layer budget, charge it once as the "imbalance cost of the chain" above: 1.84 ms
  (33% of the chain) for W=4096 prose, 0.94-0.97 ms for code or packed, and 1.55-3.57 ms at W=8192.
* **For the sim** (`CAL.effs`, chip means), the waits are spread over combine, moe_reduce and misc, so the layer
  total is right. But a what-if that improves the moe_reduce kernel saves only the kernel part: at most about
  0.47 ms per layer at W=4096 (0.83 ms against a 0.36 ms target time). The headroom-ms figures in
  `pavlo_table_ours.md` do not apply to the kernel. The 1.8-3.6 ms waits go away only through balance: expert
  replication or placement, capacity-aware routing, or packing.

## experts: fit verdict (`bench/experts_fit.txt`)

The bench covers 16 experts per chip, tokens per expert 16-1024 and 4 / 8 / 16 active experts. Here W is the
expert-weight MB read, T the tokens per chip.

| path | better model | R² additive / max | fit |
|---|---|---|---|
| **hybrid** (deployed, threshold 128) | **additive** (read, then compute) | 0.990 / 0.986 (with +c) | a = 2.34 µs/MB (427 GB/s effective weight BW), b = 0.274 µs/token (about 413 TFLOP/s), c = 68 µs |
| nd (unified only) | **max** (overlapped) | 0.971 / 0.998 (with +c) | a = 3.85 µs/MB (259 GB/s), b = 0.361 µs/token (about 313 TFLOP/s), c = 31 µs |

**Verdict.** On the path we run, Pavlo's model holds: weight read **plus** compute. The ND-only kernel already
overlaps them, but at a lower weight bandwidth. Overlapping reads with compute in the hybrid path is the kernel fix.

At the balanced load, 128 tokens per expert x 16 active experts (T = 2048, the W=4096 mean), the bench measures
1.955 ms for hybrid and 1.939 ms for ND. There is no gain from switching today.

## How much of the worst chip's experts time is imbalance, and how much is kernel

This is `load_skew_join.csv`, averaged over layers 3-6 (and over h for the single-request cases). "Bench model" is
the hybrid additive+c fit at a chip's (tokens, active experts).

| case | worst chip ms | mean chip ms | imbalance (worst - mean) | bench model at mean load | bench-predicted skew | residual (in-model - bench at worst's load) | roofline (imb 1.2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| single prose, W=4096 | 2.70 | 1.90 | **0.80 (30%)** | 1.71 | 0.65 | 0.34 | 1.45 |
| single code, W=4096 | 2.40 | 1.95 | **0.46 (19%)** | 1.79 | 0.37 | 0.25 | 1.45 |
| packed, W=4096 | 2.39 | 1.96 | **0.42 (18%)** | 1.80 | 0.36 | 0.23 | 1.45 |
| packed, W=8192 | 3.34 | 2.61 | **0.73 (22%)** | 2.37 | 0.59 | 0.38 | 1.91 |

**Answer.**

* **The worst chip's experts time** is about 70-82% kernel at the balanced load and 18-30% imbalance.
  * **The kernel part** (1.9-2.6 ms) is at 75-78% of the sum-roofline on the mean chip, which is above Pavlo's
    70% target. Against a max(read, compute) roofline it is well below target. The remaining kernel gain is the
    read/compute overlap.
  * **The imbalance part** is 0.4-0.8 ms: max/mean load of 1.5-2.1 against the sim's assumed 1.2. About 80% of it
    is predicted by the bench model from the routing counts alone. The remainder (0.23-0.38 ms) is the in-model
    kernel running slower at the hot chip's uneven per-expert counts than the bench does with equal counts.
* **The whole MoE chain** pays the imbalance about twice. The imbalance cost of the chain (1.84 ms prose, 0.94 ms
  packed at W=4096) is about 2x the experts worst-minus-mean. The hot column also has the longest dispatch and
  combine, and every other chip waits for it in combine or the RS.

## moe_reduce microbenchmark (`post_combine_reduce` + TP RS)

| tokens per chip | fused ms | bytes moved | GB/s (fraction of 512) | useful bytes, GB/s | cores | RS ms | RS eff (vs 100 GB/s) | fused+RS ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.192 | 62.9 MB | 327 (64%) | 25.2 MB, 131 | 32 | 0.224-0.262 | 36-42% | 0.415-0.454 |
| 2048 | 0.367 | 125.9 MB | 343 (67%) | 50.3 MB, 137 | 64 | 0.438-0.493 | 38-43% | 0.805-0.851 |
| 4096 | 0.750 | 251.7 MB | 336 (66%) | 100.7 MB, 134 | 120 | 0.868-0.942 | 40-43% | 1.607-1.676 |

* **Bytes moved.** Every token reads all 4 top-k slots (4 x 12 KB) and writes 12 KB, plus weights and indices.
  "Useful" counts only the ~25% local slots plus the output. That is Pavlo's roofline, `2·T·topk/tp·E·2 B`.
* **Core grid.** `min(tokens/32, grid)` cores, row-major on the compute grid: 32, 64 and 120 cores. At 4096 tokens
  per chip, 8 cores take 2 chunks. The emb dimension is never split.
* **Per-core loop** (`deepseek_moe_post_combine_reduce_reader.cpp`), `for chunk: for token(32): for slot(4)`:
  * The reader issues one 12 KB `async_read` per slot, followed by `async_read_barrier()`. Reads are serialised,
    one row in flight, with c_0 holding one row of 6 tiles. No slot is skipped.
  * Compute skips non-local slots (`table[idx] == -1`) and does `mul_tiles_bcast` scalar-and-accumulate on the
    rest.
  * So the reader moves 4x the useful bytes at about 340 GB/s.
* **Kernel fix** (the 0.36 to about 0.1 ms part). Let the reader skip non-local slots, pipeline the reads (more
  than one row in flight), and use all 120 cores at 2048 tokens per chip.
* **RS.** It runs at about 40% of its link roofline and scales linearly. It is the same collective as attn_rs and
  the shared RS.
* **Why it matters in the model.** Neither fix touches the 1.2-3.6 ms of RS wait.
