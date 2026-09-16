# `cyclic_sdpa_bw` -- the cyclic attention backward, for users

This is the backward pass of scaled-dot-product attention (dQ, dK, dV from
Q, K, V, dO and the forward's statistics) as one fused kernel over a
*cyclic schedule*: every core owns one block of keys for the whole launch,
and the query blocks travel from core to core in packets, carrying their
running dQ with them. Nothing is recomputed, nothing is read twice from
DRAM, and dQ needs no atomic adds. It exists in three forms:

| entry point | what it is for |
|---|---|
| `ttml::metal::cyclic_sdpa_bw` | one chip, the statistics given (`log_sum_exp`, `D = rowsum(dO . O)`) |
| `ttml::metal::cyclic_sdpa_bw_from_forward` | one chip, from what `sdpa_fw` returns (`attn_output` and `log_sum_exp`); computes `D` |
| `ttml::metal::ring_cyclic_sdpa_bw` | one step of a context-parallel ring backward; drop-in for `ring_sdpa_bw` |

The ring driver `ttml::ops::distributed::ring_attention_sdpa` uses the
third when asked (`backward_kind = Cyclic` or `CyclicInPlace`).

The design and the measurements behind it are in the companion repository
`tt-flash-attn`: `docs/ring-plan.md` (what was built, in order, with the
numbers) and `docs/learnings.md` (what the measurements taught). This page
is only what a caller needs.

## What it computes

For each (batch, head) slice, with `P = softmax(Q K^T / sqrt(d))` masked
causally or not:

    dV = P^T dO
    dP = dO V^T
    dS = P * (dP - D),  D = rowsum(dO * O)
    dQ = dS K / sqrt(d)
    dK = dS^T Q / sqrt(d)

The statistics are *taken as given*: `log_sum_exp` (one value per row, the
forward's normaliser) and `D`. In a ring they must be the global ones --
each step's partial sums are only right against the global normaliser --
which the ring driver arranges.

## Inputs and outputs

| tensor | shape | type | notes |
|---|---|---|---|
| `query`, `key`, `value`, `grad_output` | `(B, H, N, d)` | bfloat16 | tile layout, interleaved, on device; one shape for all four |
| `log_sum_exp`, `row_scalar` (`D`) | `(B, H, N, 1)` or `(B, H, N, 32)` | Float32 | one value per row, read from column 0 of the row's tile |
| `attn_output` (from-forward overload) | `(B, H, N, d)` | bfloat16 | `D` is formed from it in Float32 |
| dQ, dK, dV (returned) | `(B, H, N, d)` | Float32 | see "accumulating" below |

Constraints, each checked with a message naming what failed:

* `d` is a multiple of 32.
* `N` is a multiple of `2 * rows_per_block_tiles * 32` (the schedule has
  `2C` blocks and `C` must be a whole number of cores; see "sizing").
* Masks: `AttentionMaskType::Causal` (the triangle) or `None` (every block
  pair live). There is no mask-tensor path; `Arbitrary` is rejected.
* Operands bfloat16 (the matmul source registers take no Float32),
  statistics Float32.
* Not supported: dropout, grouped-query attention (`K`/`V` with fewer heads
  than `Q`), head dimensions that are not multiples of 32, sharded or
  row-major tensors.

## Accumulating into running sums

Every gradient can start from what its preallocated output holds:

* dQ always does. Pass `preallocated_grad_query` holding a running Float32
  sum and the kernel adds this launch's dQ onto it; pass nothing and it
  starts from zero.
* dK and dV do when `accumulate_into_outputs = true`, which then requires all
  three preallocated outputs. Otherwise their first visit starts from zero.

This is how the ring accumulates on device: `CyclicInPlace` passes its
three accumulators to every step and never adds on the host.

## The knobs

### `rows_per_block_tiles` (called `Bt`)

Height of a block in tiles of 32 rows: 1, 2 or 4. It is the one knob that
matters for speed, and it also sets how many cores a launch needs. Measured
throughput per core, `d = 64`, kernel alone (`DISABLED_BenchRelay`, median
of five):

| `Bt` | per core | when |
|---|---|---|
| 1 | ~0.13 TFLOP/s | short sequences (the relay and dispatch dominate); 1024 rows on 16 cores in ~160 us |
| 2 | ~0.33 TFLOP/s | middle |
| 4 | ~0.55 TFLOP/s (`d = 128`: ~0.77) | long sequences; 4096 rows on 16 cores in ~640 us, 28160 rows on 110 cores in ~4.1 ms |

Taller blocks do more arithmetic per packet hop and per DST register fill;
at `Bt = 4` the matmul pipe is busy about 38% of the time (executed FLOPs
times fidelity phases against the 5.4 TFLOP/s per core that the matrix
engine does at LoFi). Prefer the tallest `Bt` whose core count fits (next
section).

### `mask_type`

`Causal` for the diagonal block of a sequence (or the whole sequence on one
chip). `None` for a dense block pair -- a ring step where the visiting keys
are earlier than the local queries. On the causal schedule the wholly masked
tiles of a diagonal block pair are skipped at `Bt > 1`.

### `sequence_chunks`, `row_chunks`, `col_chunks` (chunk pairs)

Sub-problems within the local sequence. Every tensor's sequence is read as
`sequence_chunks` equal chunks, and sub-problem `p` attends the queries (with
`dO`, statistics and dQ) of chunk `row_chunks[p]` to the keys (with `V`, dK,
dV) of chunk `col_chunks[p]`. All sub-problems run the launch's mask mode
and each is one more slice for the planner. Empty means the whole sequence
against itself. This is what a zigzag ring step needs: a chip holding two
chunks meets two visiting chunks and exactly two of the four pairs are live.

Through `cyclic_sdpa_bw` two pairs of one launch must not share a chunk on
either side (the check says so and asks for separate launches): slices run
independently and would race on the shared rows. The ring-step op, which
plans its own launches, does pass pairs that share the visiting chunk; the
planner then deals the slices pair-major and caps the number of groups to a
divisor of the head count, so each head's pairs run in turn on one group
and accumulate in order -- which is what makes a zigzag dense step one
launch.

### `grad_query_in_tile_transposed`, `grad_query_out_tile_transposed`

Advanced. Between cores the kernels carry dQ with every tile transposed
within itself (tile positions unchanged). Where a row's dQ enters from DRAM
or leaves to it in the natural layout, it is transposed on the way, once per
row and streak -- and at small `Bt` that transpose paces the snake's first
core, where every row enters. A caller that keeps its dQ accumulator in the
tile-transposed form across launches says so with these two flags and the
kernels transpose nothing at that boundary. A zero accumulator is in both
forms at once. The ring sets them for every launch but the last causal one,
which converts every row back. Leave both `false` unless you are that
caller: with `out = true` the returned dQ is not in the natural layout.

### `use_barrier`

Debug. Orders the timesteps with a chip-wide barrier (Algorithm 3) instead
of the per-row endpoint counters (Algorithm 4). Same gradients bit for bit;
if the counters ever deadlock, flipping this says in one run whether the
protocol or something under it is at fault. Slightly slower.

### `max_groups`

Test knob. Caps the number of core groups running slices side by side
(`0` = as many as fit); the groups take the remaining slices in turn either
way. Exists to exercise that loop at sizes where every slice would fit.

### Ring-step arguments (`ring_cyclic_sdpa_bw`)

`ring_size`, `ring_axis`, `step`, `ring_direction` (`Backward` by default)
say which (chip, step) this is; the op skips exactly the (chip, step) pairs
`ring_sdpa_bw` skips, from the same helper, and picks `Causal` on the
diagonal chunk and `None` on an earlier one. `layout` is `Contiguous` (chip
`r` holds chunk `r`) or `Zigzag` (chip `r` holds chunks `r` and
`2 ring_size - 1 - r`, back to back, of a sequence cut into `2 ring_size`
chunks -- balanced work under a causal mask); `zigzag_pair` selects a pair
when a step must be split.

### Ring-driver arguments (`ring_attention_sdpa`)

| argument | values | what it does |
|---|---|---|
| `backward_kind` | `TwoPass` (the original `ring_sdpa_bw`), `Cyclic`, `CyclicInPlace` | `CyclicInPlace` accumulates on device: six dispatches fewer a step. `Cyclic` exists so a measurement can separate driver from kernel; on the zigzag layout it runs as `CyclicInPlace`. |
| `rows_per_block_tiles` | 1, 2, 4 | passed to every launch |
| `shift_transport` | `Fifo`, `Direct` | how the ring shifts move bytes; `Direct` runs at link rate over both links and is the one to use |
| `layout` | `Contiguous`, `Zigzag` | see above; `Zigzag` is for the causal mask and needs the local tensors in zigzag order |

## Sizing: how many cores a launch takes

One schedule of `N` rows at block height `Bt` needs

    C = N / (2 * Bt * 32)

cores in a rectangle of *exactly* `C` cells (the packet path is a serpentine
through it, so every hop is one core away; a rectangle with spare cells would
break that, not merely waste cores). The planner tries every height `h`
dividing `C` with `h <= grid rows` and `C / h <= grid columns`, keeps the
shapes whose serpentine is single-hop, and picks the one that fits the most
groups side by side (wider on ties). If none fits it fails with
`no rectangle of area C = ... fits a ...x... compute grid`. Then

    slices = batch * heads * pairs
    groups = min(slices, how many rectangles fit the grid, max_groups)

groups run side by side and take the remaining slices in turn. Shapes used
in the tests and benchmarks (Blackhole p150, 11x10 compute grid):

| `N` | `Bt` | `C` | rectangle | groups on the grid |
|---|---|---|---|---|
| 1024 | 1 | 16 | 4x4 | 6 (3 across, 2 down) |
| 2048 | 2 | 16 | 4x4 | 6 |
| 4096 | 4 | 16 | 4x4 | 6 |
| 2048 | 4 | 8 | 2x4 | 12 |
| 7040 | 2 | 55 | 11x5 | 2 |
| 7040 | 1 | 110 | 11x10 | 1 |
| 28160 | 4 | 110 | 11x10 | 1 |

To choose: take the tallest `Bt` for which `C` has a fitting rectangle and
the groups cover your slices in few rounds. A short sequence with many heads
wants small `C` and many groups; one long sequence wants `C` near the whole
grid.

## What to expect

Kernel alone (`DISABLED_BenchRelay`), Blackhole, `d = 64` unless stated,
median of five, September 2026:

| shape | time |
|---|---|
| 16 cores, `Bt = 1`, 1024 rows | 168 us |
| 16 cores, `Bt = 2`, 2048 rows | 268 us |
| 16 cores, `Bt = 4`, 4096 rows | 646 us |
| 16 cores, `Bt = 4`, 4096 rows, `d = 128` | 896 us |
| 110 cores, `Bt = 4`, 28160 rows | 4.04 ms |

Ring backward on 4 chips, zigzag, `CyclicInPlace`, direct shifts, against
the original two-pass backward on its better layout, milliseconds:

| rows per chip, `Bt` | two-pass | cyclic | change |
|---|---|---|---|
| 1024, 1 | 4.00 | 2.69 | -33% |
| 2048, 2 | 6.55 | 3.69 | -44% |
| 4096, 4 | 15.02 | 6.75 | -55% |
| 8192, 4 | 42.65 | 13.10 | -69% |
| 10 heads, 5632, 4 | 51.79 | 13.17 | -75% |

At 4096 rows per chip and above the ring step is bound by the shifts, not
the kernel; below 1024 by dispatch.

### Utilisation against the repository's two-pass backward

Same problem, same 110 cores, causal, `d = 64`, both kernels timed on one
chip (`DISABLED_CompareWithTheRepositorysBackward`, median of five). MFU
is useful FLOPs (five matmuls over the causal triangle) over the LoFi
matrix-engine peak of the 110 cores, 594 TFLOP/s; "busy" is executed FLOPs
times fidelity phases over the same peak (the two-pass kernel recomputes
the score stage in both passes, 7 matmuls at HiFi4; the cyclic runs 5 at
3.6 phases on average). The two-pass kernel's DRAM bytes are counted from
its readers (it re-reads K and V for every query tile and Q, dO and the
statistics for every key tile, so its traffic grows with the square of the
sequence); the cyclic kernel's are the lower bound of every operand once
(its relay never re-reads). The card's DRAM peak is 512 GB/s.

| rows x heads, `Bt` | two-pass: TFLOP/s, MFU, busy, GB/s | cyclic: TFLOP/s, MFU, busy, GB/s | speed-up |
|---|---|---|---|
| 7040 x 1, 1 | 9.1, 1.5%, 9%, 289 | 16.4, 2.8%, 10%, 15 | 1.8x |
| 7040 x 2, 2 | 10.1, 1.7%, 10%, 322 | 37.5, 6.3%, 23%, 34 | 3.7x |
| 14080 x 1, 2 | 10.2, 1.7%, 10%, 323 | 37.6, 6.3%, 23%, 17 | 3.7x |
| 14080 x 2, 4 | 10.4, 1.8%, 10%, 328 | 60.1, 10.1%, 36%, 27 | 5.8x |
| 28160 x 1, 4 | 10.6, 1.8%, 10%, 332 | 63.0, 10.6%, 38%, 14 | 6.0x |

Bytes moved for the 28160-row problem: about 8 GB against 58 MB, a factor
of 138. The two-pass kernel sits at roughly 65% of DRAM bandwidth with its
matrix pipe a tenth busy; the cyclic kernel uses 3% of the bandwidth and
is bound by its own pipeline (the vector unit's DST traffic in the score
pass, then the matmuls of the gradient updates), not by DRAM.

## Accuracy

Outputs are Float32 with Float32 accumulation throughout. Against a Float64
host reference on random inputs the relative RMS error is 1e-4 to 3e-4 for
all three gradients at every block height (`d = 64` and `128`, causal and
dense), with the systematic part (a least-squares scale bias) at or below
2e-4. Where the arithmetic is exact it is documented as such: the dQ seed
and the dK/dV accumulators are never rounded through a source register, the
probabilities are Float32 to within 3e-6 relative before they meet the
matmuls, and the statistics are carried to 20 bits. What is not exact is
what the matmul source registers keep of a Float32 operand (19 bits, rounded
by the packer, not truncated) and the bfloat16 inputs themselves.

## Testing and measuring

Correctness: `ttml_tests --gtest_filter='CyclicSdpaBw*'` (84 tests: the
relay, groups, the op, dense mode, chunk pairs, the endpoint protocol,
against the real forward). Ring correctness on a loudbox:
`LoudboxRingSDPATest.*`.

Measurement tests are `DISABLED_*` and take environment knobs:

| test | knobs |
|---|---|
| `CyclicSdpaBwTimingTest.DISABLED_BenchRelay` | the kernel alone over the shapes above; prints us, TFLOP/s per core, DRAM GB/s |
| `CyclicSdpaBwTimingTest.DISABLED_ProfileRingLaunch` | one launch shaped like a ring step: `RING_LAUNCH_ROWS`, `RING_LAUNCH_HEADS`, `RING_LAUNCH_BT`, `RING_LAUNCH_KIND=dense|diagonal`, `RING_LAUNCH_DQ_TRANSPOSED=1` (dQ in the kernels' form on both sides, as inside the ring) |
| `CyclicSdpaBwOpTest.DISABLED_ReportErrorAcrossBlockHeights` | the accuracy report: relative error, RMS and scale bias per gradient over block heights, masks and input distributions |
| `LoudboxRingSDPATest.DISABLED_CompareLayouts` | the ring table above; `TTML_LOUDBOX_RING8=1` for eight chips, `TTML_LOUDBOX_MIN_ROWS` to skip small cases |
| `LoudboxRingSDPATest.DISABLED_ProfileOneBackward` | per-phase host profile of one ring backward: `TTML_LOUDBOX_PROFILE_ROWS`, `_BT`, `_LAYOUT` |

With `TT_METAL_DEVICE_PROFILER=1` the kernels emit device zones per
timestep (`T-STEP`, `SCORES`, `UPDATE-DQ/DK/DV`, `WAIT-PACKET` on the
compute threads; `RELAY-READER-STEP`, `SEND-IMM`, `PREFETCH-IMM` on the
reader; `STAT-ROWS`, `STAT-GATHER`, `STAT-EXPAND` on the writer). Zones
measure the RISC's issue time: a wait on a Tensix semaphore reads as
nothing and surfaces as back-pressure in the next zone.

## Where things are

    tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/        the op: wrapper, device operation, program factory
      device/kernels/compute/cyclic_sdpa_bw_compute.cpp    the fused compute kernel
      device/kernels/dataflow/cyclic_sdpa_bw_relay_*.cpp   the relay reader and writer (column-resident variant)
      device/kernels/dataflow/cyclic_dataflow_utils.hpp    packets, statistics, soft-float
      device/cyclic_schedule.hpp                           the schedule (pairs, producers, consumers) as constexpr
    tt-train/sources/ttml/metal/ops/ring_cyclic_sdpa_bw/   the ring-step op
    tt-train/sources/ttml/ops/distributed/ring_attention_sdpa.cpp   the ring driver
    tt-train/tests/ops/cyclic_sdpa_bw_device_test.cpp      the tests and benchmarks
    tt-train/tests/ops/distributed/ring_sdpa_loudbox_test.cpp   the ring tests

Branch `bklockiewicz/cyclic-kernel-opt` (kernel work, off
`bklockiewicz/ring-zigzag`). The original two-pass `ring_sdpa_bw` is
untouched and remains the reference the comparisons are taken against.
