# MiniMax-H3 on Wormhole Galaxy: the ring joint SDPA — baseline, zones, experiments, results

The attention op of the transformer block, `RingJointSDPADeviceOperation` (the "normal" ring joint SDPA), and the
work done on it between 2026-09-17 and 2026-09-21 on the 4x8 Wormhole Galaxy `UF-EV-B12-GWH02`, branch
`jameslee/exp_ring_sdpa_wh` (off `jameslee/bringup_h3_wh_galaxy` at `1361ebabb85`). This merges the SDPA sections of
the block perf write-up, the exp ring SDPA handoff and its optimization log (2026-09-18) into one document ordered as
baseline -> zone breakdown -> experiments per zone -> results. Block-level context: [README.md](README.md).

Commits: `7274624d524` (exp ring op bring-up on Wormhole: two hang fixes, 2-or-4-link MUX layout, model/test knobs,
first A/B), `3559b70d163` (sequential passes, `TT_EXP_SDPA_Q_GROUPS`), `f9aa767a918` (bottom-row MUX placement, 64
SDPA cores, chunk-size sweep at 15 s), `b4736d61242` (inner loop: blocked pack at width 4, phase-zone diagnostic,
dst_full_sync and exp-approx A/Bs, profiler tools). Blackhole reference for the exp op: `42986a68fe0`.

## 1. The op and the baseline

Target: 15 s / 768p / 16:9, TP=4 / SP=8, 13664 rows/device, 14 heads/device, head dim 128, HiFi2 bf16, ring of 8
devices on 4 fabric links. The block runs the normal ring joint SDPA at `q_chunk=256 / k_chunk=512` on 63 compute
cores (7x9; the profiler's `CORE COUNT` reads 71 because it counts the fused CCL workers, placed in the reserved
last column by `ccl_core_grid_offset=(7, 0)` with `use_column_major_ccl=True`, `attention_minimax_h3.py:572-573`).

| measurement | value | source |
|---|---|---|
| in the block, 2026-09-17 baseline (`bc1d99d05f6`) | **174.56 ms** fsdp1 / 174.57 fsdp0 = **70.7%** of the 246.92 ms block | README Part 2, Tracy breakdown |
| `PM FPU UTIL` | 48.3% (47.5-48.7 across the 32 devices) | same profile |
| roofline | 10.71 TFLOP per device per layer; **83.0 ms** at HiFi2 peak on 63 cores; measured / ideal = 48% | README Part 2, cross-check table |
| isolated op, `create_perf_table[minimax_h3_15s_768p]` at 13664 rows | 175.269 ms (run 1) / 171.694 ms (run 2), 47.3-48.1% FPU, 35.1-35.6% math | §2.1 |
| in the block, re-measured 2026-09-21 (`d4fca5e3f23` + working tree, 6 runs) | **172.16 +- 0.03 ms**; device-to-device spread 0.1 ms | README Part 2, variance section |
| padded-shard reference for the exp-op A/Bs (14336 rows/device, `create_perf_table[minimax_h3_15s_768p_pad14336]`) | 192.9 ms; **191.6 ms** after the pack-4 inner-loop change | §3.4 |
| SP=32-equivalent shard (3424 rows/device) | 15.74 ms isolated, 14.76 ms in the `sp_sim4` block | §3.2 |

The utilisation figures in the sources are one fact seen through different instruments: 48.3% is Tracy's
`PM FPU UTIL` (the perf-model ideal divided by measured time, not a hardware counter), 48% is the roofline
cross-check, 47.3-48.1% the isolated sweep's reading of the same column, and "50% of the step" is the inner-loop
zone arithmetic (§2.2: 33 us of FPU matmul in a 66 us step). Likewise the L1 budget appears as 1,499,136 B (the
part's maximum), 1.31 / 1.34 MB (usable after reserved regions, as the op's allocator and the CB check see it) and
1.23 MB (the exp op's single-pass footprint at q256/k512).

## 2. Baseline breakdown by zone

Two granularities exist for this op. At the op level the "zones" are the chunk shape (how many inner-loop steps
each core runs, and whether the CBs fit L1). Inside the kernel the inner-loop step has per-phase device zones on
each compute thread.

### 2.1 Chunk shape and the L1 envelope (op level)

Sweeping only q >= 256 with k <= 512 bounds the search on the wrong axis. From the CB allocation in
`ring_joint_sdpa_program_factory.cpp:1296-1308`:

```
q = 8*Sq    k = 8*Sk    v = 8*Sk    mask = Sq*Sk    qk = Sq*Sk
out_im = 4*Sq    out0 = 4*Sq    stats = Sq
```

`Sq*Sk` dominates, but Sq carries the heavier linear term — q, out_im, out0 and the statistics FIFO
all scale with it, against K/V's two buffers on Sk. That is why `(512, 256)` is L1-infeasible while
`(256, 512)` fits at the same product. The untried direction is therefore **smaller q with larger
k**, which also halves the ring K-loop iterations — the property that made k=512 beat k=256.

That prediction was **wrong**, and the widened sweep closes the question. q in {192, 256} x k in
{512, 640, 768, 1024} (q=128 excluded: it hung the op twice on 2026-09-17 on another Wormhole galaxy
at seq_local 13632 / k=512, did not reproduce on `UF-EV-B12-GWH02` the same day -- 6/6 completed on
the same code, under the profiler and the watcher, at 13632 and 13664 -- and is slower than q=192 at
every feasible k regardless):

The widened sweep was run twice on the normal op at 13664 rows/device (`test_ring_joint_attention_create_perf_table
[minimax_h3_15s_768p]`, run plain: it self-shells `run_device_profiler`, so wrapping it in `--profile` nests
profilers). The two runs rank the shipped `(256, 512)` first both times and disagree by 2% on its absolute time,
which is the day-to-day spread of this test, not a change in the op.

| rank | q_chunk | k_chunk | run 1 | run 2 | iters/core | pad waste | slot waste | FPU util (run 1 / run 2) | math util |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **256** | **512** | **175.269 ms** | **171.694 ms** | 2592 | 2.3% | 0.0% | 47.3-47.7% / 48.1% | 35.1% / 35.6% |
| 2 | 192 | 640 | 184.930 ms | 193.960 ms | 2816 | 4.1% | 0.0% | 44.9-46.1% / 42.6-42.9% | 33.2% / 31.5% |
| 3 | 192 | 512 | 185.212 ms | 190.572 ms | 3456 | 2.3% | 0.0% | 44.8-45.7% / 43.3-43.7% | 33.2% / 32.1% |
| — | 192 | 768 / 1024 | L1 infeasible | | | | | | |
| — | 256 | 640 / 768 / 1024 | L1 infeasible | | | | | | |

`(256, 512)` is what `measured_sdpa_chunk_sizes[13664]` ships, and it wins by 5.5%. The five
infeasible points fail with `Statically allocated circular buffers on core range [0-0 - 6-8] grow to
N B which is beyond max L1 size of 1499136 B` — Wormhole's 1.5 MB/core, on the 7x9 = 63 compute grid
— at N = 1,602,880 (192/768), 1,639,744 (256/640), 1,836,352 (256/768), 1,963,328 (192/1024) and
2,229,568 (256/1024), each matching the L1 envelope calibrated below.

Slot waste is zero at both feasible q: 13664 rows give 54 Q chunks at q=256 (54 x 14 heads = 756 =
12 x 63) and 72 at q=192 (1008 = 16 x 63), so the ranking is decided by per-core efficiency, not
scheduling — and larger q wins, FPU utilization 47.5% against 45.5%. The harness reports
"63 compute + 9 CCL = 72 total cores" and measures SDPA at 175.269 ms against 174.56 ms in-block,
0.4% apart.

So chunk-size tuning at 15 s is exhausted: 3 feasible points, 5 ruled out by L1, and the shipped
config is the best of them. The ~48% FPU / 35% math utilization is **inherent to the ring joint SDPA
kernel at this shape**. At 71% of the block it is the only thing worth attacking, but the work is in
the kernel. Note the contrast with 5 s, where `q=320` wastes 16.7% of the 63 slots and chunk
tuning *does* have headroom.

Smaller q *does* unlock a larger k — `(192, 640)` builds where `(256, 640)` does not, the first
k > 512 point run on this shape — but it is **13% slower**. Two reasons, both visible above:

  * Larger q chunks are more efficient per core. FPU utilization drops 48.1% -> 42.6% and math
    utilization 35.6% -> 31.5% going from q=256 to q=192.
  * "Larger k halves the ring K-loop" ignores that shrinking q *multiplies* the Q-chunk count.
    iters/core goes 2592 at (256, 512) to 2816 at (192, 640) — more iterations, not fewer. The two
    effects oppose each other and q dominates.

So `(256, 512)`, which `measured_sdpa_chunk_sizes[13664]` already ships, is optimal. **Chunk-size
tuning at 15 s is exhausted**: 3 feasible points measured, 5 ruled out by L1. The ~48% FPU / 35.6%
math utilization is inherent to the ring joint SDPA kernel at this shape.

#### The L1 envelope, calibrated

The four L1 failures carry exact byte counts, which fit the footprint exactly (Sq = q/32,
Sk = k/32):

```
bytes = 2048*Sq*Sk + 67584*Sq + 32768*Sk + 116032        (Wormhole max: 1,499,136)
```

Per unit that is 1 tile for `Sq*Sk`, **33 tiles for Sq and 16 for Sk** — so Sq is about twice as
expensive as Sk, which is why `(512, 256)` fails while `(256, 512)` fits at the same product. The
`Sq*Sk` coefficient being one tile rather than two also says the mask CB is not allocated here,
consistent with `is_causal=False`. Observed: `(6,24)` 1,602,880 B; `(8,20)` 1,639,744 B; `(8,24)`
1,836,352 B; `(6,32)` 1,963,328 B. Reusable for any future chunk question on this part.

### 2.2 The inner-loop step (kernel zones)

Both ring ops run `sdpa_inner_loop_step` (`compute_streaming.hpp`) once per (Q chunk, K chunk):
at q256 / k512 that is 2912 steps per core per call at 15 s, 66 µs each (192 ms). The kernel
already carries per-phase device zones behind a `profiling_enabled` template flag; the env knob
`TT_EXP_SDPA_PROFILE_INNER=1` (exp op) compiles them in. The zones were taken on the exp op, but the step is the
same function in both ring ops, so the table is the baseline op's inner loop too. The L1 profiler buffer holds ~125 zones per
RISC per launch, so the log covers the first step and a half of each core, which is what the table
uses (`tools/sdpa_phase_zones.py` on the report's `profile_log_device.csv`). Step 0 on the math
thread includes a 15 µs wait for the first K chunk; the steady-state step is ~66 µs.

Per step, one core (device 0, core (1,1)), pack-4 build:

| thread | matmul zones (QK + PV) | softmax zones | outside all leaf zones |
|---|---|---|---|
| unpack (TRISC_0) | 16.6 + 21.1 µs | SUB 8.7, reduce 3.1 | 17.7 µs |
| math (TRISC_1) | 34.3 (19 steady) + 21.2 µs | SUB 10.1, init 1.9, reduce 0.8 | 10.9 µs |
| pack (TRISC_2) | 22.2 + 2.2 µs | EXP 14.4, PACK SUB_EXP 14.0, reduce 1.7 | 24.9 µs |

Reading. The pure FPU work is 1024 tile-matmuls per step (QK 8x16x4, PV 8x4x16) at 32 cycles each
for HiFi2 = 33 µs, i.e. **50% of the step**, which is the 48% "FPU util" the roofline reported.
Inside their zones the matmul blocks run at ~80% of that rate. The other half of the step is the
softmax and the thread handshakes: on the pack thread the exp (SFPU, `exp_packthread_tile`) and the
in-place pack plus the row-sum accumulate pack (every probability tile is packed twice, the second
time with packer L1-accumulate into the row-sum tile) cost 28 µs per step; on the math thread the
broadcast subtract of the row max is 10 µs (128 tiles at ~80 cycles) and 11 µs sit between zones in
`tile_regs_acquire`/`wait` handshakes and CB waits. Nothing waits on DRAM or the fabric after the
first chunk: memory and fabric are off the critical path, the core is bound by its own non-matmul
work. That is also what the chunk sweep said (time follows steps, not FLOPs).

**Which resource binds.** Three checks, cheapest first:

1. Roofline arithmetic from the op time. Bytes each resource moves per call divided by the time,
   against its peak. Here: DRAM ~0.9 GB per device per call (4.6 GB/s, under 2% of peak); fabric 84 MB
   per link per direction (0.43 GB/s, under 5%); FPU 1024 tile-matmuls per step at 32 cycles (HiFi2) =
   33 µs of a 66 µs step, 50%. So neither memory nor fabric bandwidth binds, and the FPU is only half busy.
2. A/B a knob that touches one resource. 2 vs 4 links moved 3.4 ms with the fabric at 5% utilization:
   a latency stall on the per-chunk semaphore, not bandwidth. Dropping math fidelity would test the FPU
   the same way.
3. Look inside the kernel with the phase zones. If the math thread's matmul zones filled the step the
   FPU would bind; if `cb_wait_front` time dominated, data would. Neither: half the step is softmax and
   thread handshakes (below). Tracy's hardware-counter capture (`--profiler-capture-perf-counters`)
   would give the direct FPU number but deadlocks on this box (its inner `python -m tracy` waits on a
   UMD chip lock its parent holds).

## 3. Experiments, by zone

### 3.1 Chunk shape (experiments 3, 3b, 4 of the README index)

All three closed the same way: the shipped `(256, 512)` is optimal at 15 s.

- **q in {256, 384, 512} x k in {256, 512}** (experiment 3): larger q is L1-infeasible; `(256, 512)` wins. Done.
- **`q_chunk = 128`** (3b): slower than q=192 at every feasible k, so not a perf path; it also hung the op twice on
  2026-09-17 on another Wormhole galaxy at seq_local 13632 / k=512, did not reproduce here (6/6 clean under the
  profiler and the watcher). Done, rejected.
- **small q / large k, q in {192, 256} x k in {512, 640, 768, 1024}** (4): the hypothesis of §2.1 was wrong;
  `(192, 640)` is the first k > 512 point that builds and it is 13% slower. Done, rejected; the L1 envelope above is
  the by-product. Chunk-size tuning at 15 s is exhausted (3 feasible points, 5 ruled out by L1). Note the contrast
  with 5 s, where `q=320` wastes 16.7% of the 63 slots and chunk tuning does have headroom.

### 3.2 Core count and placement: the exp ring op (experiment 7, 2026-09-18)

`exp_ring_joint_scaled_dot_product_attention` is the fused ring-attention kernel that measured 21% faster
than `RingJointSDPADeviceOperation` on Blackhole at the H3 shard `[1, 14, 3424, 128]`, ring 8 (`42986a68fe0`).
This subsection records what it took to run it on this galaxy and what it measures against the normal op. All
numbers are max over the 32 devices, `DEVICE KERNEL DURATION`, from Tracy CSVs under
`generated/profiler/reports/2026_09_18_01_4*`–`02_0*`.

#### What blocked it, and the fixes

| blocker | where | fix |
|---|---|---|
| model gate `is_blackhole() and sp == 32` | `attention_minimax_h3.py` | `MINIMAX_H3_EXP_RING_SDPA=1/0` forces it on/off; unset keeps the Blackhole rule |
| SDPA rows must be even (backward/forward MUX-client halves) | `exp_ring_joint_sdpa_program_factory.cpp` "SDPA grid rows must be even" | program grid `(8, 8)`: **7x8 = 56 SDPA cores** (the normal op has 63); `num_workers_per_link = 4` |
| `num_links == 2` `TT_FATAL`; one MUX-client column per link | `exp_ring_joint_sdpa_device_operation.cpp` | the model passes 2 for this op (`MINIMAX_H3_EXP_RING_NUM_LINKS`); the factory now also lays out 4 client columns / 8 MUX kernels for `num_links=4` |
| **fabric packet-header pool**: the AG writer allocates 8 scatter + 2 unicast + 1 atomic-inc headers per RISC; Wormhole's pool is `NUM_PACKET_HEADERS / 2 = 8` per RISC (Blackhole 12), and `PacketHeaderPool::allocate_header` spins forever on exhaustion | `exp_ring_joint_writer.cpp`; `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h:149` | rotation sized from the budget: 4 scatter headers on Wormhole, 8 on Blackhole. This was the first hang: every fabric writer on all 32 chips parked in `allocate_header` |
| reader's per-link semaphore array was `[2]` | `exp_ring_joint_reader.cpp` | `[4]` with an assert; with 4 links the overflow corrupted the reader's stack and every reader exited without work (second hang) |
| 8 KB fabric payload illegal on WH (cap 7616 B) | `fabric_context.cpp` | the H3 WH mesh already runs 4 KB; the op derives packet size from the fabric |
| `kMaxPasses = 3` | factory + device op | raised to 4 (the CB budget check is what bounds passes) |

#### L1 decides the shape, not the gates

The op keeps every pass's Q chunk and flash state resident for the whole op, so per-core L1 scales with
`rows_per_device x heads_per_device / SDPA cores`. Using the model's own `_exp_sdpa_l1_bytes`: **no
(cols, segs, q, k) fits 5 s, 10 s or 15 s on the 7x8 grid** (minimum 2.45 MB at 15 s against a 1.31 MB
budget, even with Q streamed and the pass cap lifted). The only H3 shard that fits is the SP=32-equivalent
`3424 rows/device` — the `sp_sim4` shard — at q512/k128 (2 passes, streamed Q) or q256/k256 (4 passes,
streamed Q). Everything below is measured there; **the real 15 s pipeline shard cannot run the exp op as
designed**. Making it fit means sequential passes (pass-outer, ring-inner) with passes ≥ 1 reading the
gathered K/V from DRAM instead of the fabric — kernel work across reader, writer, compute and factory.

#### Single op, `[1, 14, 3424, 128]`, ring 8, HiFi2 bf16, PCC 0.99975 on every exp point

| op | config | cores | per call |
|---|---|---|---|
| normal `RingJointSDPA` (sweep best of 7, `create_perf_table[minimax_h3_15s_768p_sim32]`) | q256 / k512 | 63 | **15.74 ms** |
| normal | q160 / k256 | 63 | 16.11 ms |
| exp, 2 links | q256 / k256, 4 passes | 56 | 16.65 ms |
| exp, 2 links | q512 / k128, 2 passes | 56 | 20.26 ms |
| exp, 4 links | q512 / k128, 2 passes | 56 | 20.33 ms (no change vs 2 links: the K/V gather is not on the critical path) |

#### In the block (`test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim4-15s_768p-…_is_fsdp1]`)

| SDPA variant | SDPA row | block device-only |
|---|---|---|
| normal ring op (model fallback q256 / k512) | **14.76 ms** | **44.39 ms** |
| exp, q256 / k256, 4 passes (`MINIMAX_H3_EXP_RING_MAX_PASSES=4`) | 16.09 ms | 45.12 ms |
| exp, q512 / k128, 2 passes | 19.88 ms | 49.13 ms |

The normal op is **~9% faster on the SDPA row** at the one shard the exp op can hold, and the exp op runs
on 56 cores where the normal op has 63. The Blackhole 21% did not transfer: there the exp op fits Q and
state resident on 110 cores at q160/k512, here L1 forces streamed Q and k=128–256 chunks whose per-chunk
overhead costs more than the removed DRAM save/restore saves. Both ops share the same inner loop
(`sdpa_inner_loop_step`), so the Wormhole utilization gap (48% vs Blackhole's ~70% on the same kernel) is
untouched by either.

Reproduce: unit test `test_exp_ring_joint_attention.py::…[wormhole_b0-4x8_wh_h3_sim32{,_p4,_nl4}-ring]`
(PCC + timing under `--profile`), normal-op table `test_ring_joint_sdpa.py::…create_perf_table[minimax_h3_15s_768p_sim32]`,
block A/B with `MINIMAX_H3_EXP_RING_SDPA={1,0}` and `MINIMAX_H3_EXP_RING_MAX_PASSES={3,4}`.

#### Sequential passes: the exp op on the 15 s shard (2026-09-18, `3559b70d163`)

The L1 wall above comes from the lockstep schedule: every pass's Q chunk and flash state stay resident
because all passes advance together per ring iteration. `TT_EXP_SDPA_Q_GROUPS=G` (factory + all three
kernels, one setting per process) switches the op to **pass-outer / ring-inner**: one pass runs all
ring iterations before the next starts, so one Q chunk and one flash state are live per core (the
normal op's `q_per_core == 1` scratch path, no L1 state FIFO), and a head-segment's Q chunks are split
into G groups of one chunk per column walked as extra passes. Only group 0 of a segment forwards K/V
over the fabric; later groups re-read the gathered K/V the first group landed in DRAM. Per-core L1 is
then the single-pass footprint (q256/k512: 599 tiles = 1.23 MB) at any shard size.

Constraint: `num_q_chunks % (columns x G) == 0`. 13664 rows give 54 chunks of q=256, which no
7-column layout divides, so the shard is padded to **14336 rows/device** (56 chunks = 7 x 4 x 2
segments, +4.9% work). The normal op was measured at the same 14336 rows for the comparison.

| op | layout | links | per call |
|---|---|---|---|
| normal `RingJointSDPA` (`create_perf_table[minimax_h3_15s_768p_pad14336]`) | q256 / k512, 63 cores | 4 | **192.9 ms** |
| exp, sequential, G=4 | segs=2 (pair dedup on), 4 segment-passes on rows 0-3 and 3 on rows 4-7, 56 cores | 2 | 238.0 ms |
| exp, sequential, G=4 | same | 4 | 238.2 ms |
| exp, sequential, G=2 | segs=4 (7 balanced passes per row, no dedup: every row forwards) | 2 | 212.6 ms |
| exp, sequential, G=2 | same | 4 | **206.7 ms** |

Numerics: sequential mode PCC 0.99975 at the 3424-row shard (`4x8_wh_h3_sim32_seq`, G=2), identical to the
lockstep schedule; the 14336-row runs are timing-only (the torch reference at 114688 tokens does not fit
host memory) and validated end-to-end only through that smaller PCC.

Reading: the exp op **now runs the 15 s shard on Wormhole**, best at 206.7 ms against the normal op's
192.9 ms (+7%). Per core it is ahead: 206.7 ms x 56 cores = 11.6 core-s against 192.9 x 63 = 12.2 for the
normal op, i.e. ~5% less core time for the same work. The remaining gap is exactly the 7 cores the
even-row MUX-client constraint costs on a 9-row grid; a 9-row layout (asymmetric backward/forward
client halves) was proposed here to close it. *Superseded the same day by the bottom-row MUX placement below, which
recovers 64 cores host-side; and the prediction that the cores alone would make the exp op the faster kernel did not
hold (64 cores: 196.2 vs 192.9 ms).* Row balance matters more than fabric
duplication: segs=4 forwards every head 4x yet beats segs=2 with pair dedup by 11%, and only at segs=4
do 4 links help (212.6 -> 206.7 ms). Reproduce with
`TT_EXP_SDPA_Q_GROUPS={2,4} … test_exp_ring_joint_attention.py::…[wormhole_b0-4x8_wh_h3_15s_seq{,_nl4}-ring]`
under `--profile`; the normal-op row is `create_perf_table[minimax_h3_15s_768p_pad14336]`.

#### Bottom-row MUX placement: 64 SDPA cores instead of 56 (2026-09-18, `f9aa767a918`)

The 7 lost cores above come from the reserved MUX *column*: the SDPA grid is 7 wide, and the 9-row
height must round down to 8 for the equal backward/forward MUX-client halves. The op already had a
Blackhole placement experiment, `TT_EXP_SDPA_MUX_BOTTOM_ROW`, that puts the MUX kernels on the bottom
*row* instead and gave up two rows to keep the count even. On the 8x9 Wormhole grid one row is enough:
SDPA keeps all 8 columns and rows 0-7 (8x8 = **64 cores**), the 8 MUX kernels of 4 links fill row 8
exactly. The change is host-side only: the grid helper (`exp_sdpa_grid_for_user_grid`) drops one row
plus an idle row only when the remainder is odd, and the bottom-row MUX list is generalized from two
hard-coded pairs to `num_links` columns per direction (the 9-row asymmetric-halves layout proposed above would give 63 cores for far more surgery, so it is not needed).

With 8 columns the 56 chunks of q=256 divide as 8 x 7, so a segment is 7 chunks wide and the sequential
mode runs G=1: 98 segments on 8 rows -> 13 passes on rows 0-1, 12 on rows 2-7 (ideal 12.25). Every pass
forwards its head's K/V (no groups), so a row forwards 13 shards per op against 7 in the segs=4/G=2 layout.

| op | layout | links | per call |
|---|---|---|---|
| normal `RingJointSDPA` | q256 / k512, 63 cores | 4 | **192.9 ms** |
| exp, sequential, G=2 (previous best) | segs=4, 7x8 = 56 cores, reserved-column MUX | 4 | 206.7 ms |
| exp, sequential, G=1 | q256 / k512, segs=7, **8x8 = 64 cores**, bottom-row MUX | 2 | 199.6 ms |
| exp, sequential, G=1 | same | 4 | **196.2 ms** |
| exp, sequential, G=1 | q448 / k256, segs=4 (7 passes of 14 tile-rows), 64 cores | 4 | 200.9 ms |
| exp, sequential, G=1 | q192 / k512 at 13824 rows/device (72 chunks = 8 x 9; 16 passes of 6 tile-rows) | 4 | 220.2 ms |
| exp, sequential, G=1 | q128 / k512, segs=14 (25 passes of 4 tile-rows), 64 cores | 4 | 336.2 ms |
| exp, sequential, G=1 | q448 / k512 | 4 | does not build: CBs need 1.71 MB of 1.34 MB |

Numerics: PCC 0.99975 at the 3424-row shard on the bottom-row layout with 2 links (`4x8_wh_h3_sim32_seq`,
7x8, unchanged from the reserved-column number) and 0.99972 on the 64-core grid with 4 links at a
4096-row shard (`4x8_wh_h3_4096_bot_nl4`, 16 chunks = 8 x 2). The 15 s rows are timing-only as before.

Reading: 64 cores take the exp op from 206.7 to **196.2 ms**, 1.7% behind the normal op (192.9); both numbers are before the pack-4 inner-loop change of §3.4 (193.7 vs 191.6 after it). The
per-core work model predicted 206.7 x 13/14 = 192 ms; the missing 4 ms is fabric traffic (13 forwards
per row instead of 7), visible as 2 links -> 4 links = 199.6 -> 196.2 and as a 3 ms spread across
devices that the 56-core layout did not have (206.80 / 206.84). The chunk sweep says the op's time
follows the number of inner-loop steps (passes x K chunks), not Q tile-rows: q=192 does 8% fewer
tile-rows per core but 19% more steps and is 12% slower; q=128 nearly doubles the steps and is 71%
slower; q=448 cuts the steps but only fits with k=256, and the halved K chunk costs more than the
larger Q chunk saves. q256 / k512 stays the shape. What is left between 196.2 and a win: the
13-vs-12.25 pass imbalance (6%, inherent to 98 segments on 8 rows at q=256) and the duplicate
forwarding (~2%, the 2-vs-4-link gap). With segs=7 a pass holds 8 segments of 2-3 distinct heads, so
generalizing the pair dedup to "one forwarder per (pass, head)" would cut each row's forwards from 13
to about 4 and take the traffic off the critical path even on 2 links. Reproduce with
`TT_EXP_SDPA_MUX_BOTTOM_ROW=1 TT_EXP_SDPA_Q_GROUPS=1 … [wormhole_b0-4x8_wh_h3_15s_seq{,_nl4}-ring]`
under `--profile`; the q128 / q192 / q448 rows are the `4x8_wh_h3_15s_q{128,192,448}_nl4` cases.

### 3.3 Fabric forwarding (proposed, and one structural limit)

Both items come out of the 64-core result: 4 links beat 2 by 3.4 ms (199.6 -> 196.2) with the fabric at ~5%
utilisation, so the cost is a latency stall on the per-chunk semaphore, not bandwidth.

#### One forwarder per (pass, head) — the remaining ~2% and the 2-link path (proposed, ranked next)

At segs=7 on 8 rows each row forwards its head's K/V on all 13 passes; only 2-3 distinct heads are
live per pass, so 8 rows send 8 copies of 2-3 shards. Cost today: 4 links beat 2 links by 3.4 ms
(199.6 -> 196.2) and the result sits 4 ms above the 13/14 work model, plus a 3 ms device spread the
56-core layout did not show. Generalize the split-head pair dedup (`row_dedup_role`,
`row_buddy_injector`, the reader's `dedup_role` gate) from fixed row pairs to per-pass groups: for each
pass, the rows whose segment belongs to the same head elect a leader (lowest row in the same direction
half) that forwards, followers gate on the leader's relayed semaphore. The per-row role is per pass
now, so it becomes a per-pass runtime-arg table (or a compact `(head -> leader row)` rule both
neighbors compute identically: `head = (p * rows + y) / segs`, leader = smallest `y` in the half with
that head). Follower rows must still connect to the MUX (they forward on other passes), so the
channel-reclaim trick of the pair version does not apply; keep one channel per row. Measure 2 vs 4
links again: if 2 links then match 4, traffic is off the critical path.

#### The pass imbalance (6%, structural)

98 segments on 8 rows -> 13 vs 12.25. Nothing in the chunk sweep fixes it: q=192/128 reduce the
tile-row imbalance but add inner-loop steps and lose 12-71%; q=448 fits only with k=256 and loses 2%.
The only balanced q=256 layouts need a row count dividing 98 (7 or 14 rows) or a segment count
dividing 8 x k: e.g. a 7-row SDPA grid (56 cores, 14 passes exactly, worse) or 16 heads. Accept it, or
treat the two 13-pass rows as the place to put the joint (text) chunks if the pipeline has any.

### 3.4 The inner loop (experiments A-D, `b4736d61242`, shared with the normal op)

Exp op on 64 cores at 15 s, 196.2 ms base; normal op on the padded shard, 192.9 ms base; max over 32 devices.

| change | exp op | normal op | verdict |
|---|---|---|---|
| A. `MIN_BLOCKED_PACK_TILES` 8 -> 4 on Wormhole: one pack per 4-wide subblock row instead of four single-tile packs | 196.2 -> **193.7 ms** | 192.9 -> **191.6 ms** | kept |
| B. full-sync 16-tile DST (`dst_full_sync_en`; the exp factory now forwards it so the host's subblock search and the kernel agree) | 262.8 ms | | rejected: the half-sync ping-pong that overlaps math and pack is worth far more than larger subblocks |
| C. approximate SFPU exp (`TT_EXP_SDPA_TEST_EXP_APPROX=1`) | 192.7 ms, PCC unchanged | | 0.5%: the EXP zone is not SFPU-op bound; the model keeps exact exp |
| D. hardware counters via `SAFE_PYTEST_TRACY_OPTS="--profiler-capture-perf-counters=fpu,pack,unpack"` | no data | | Tracy multi-pass capture deadlocks on this box |

Also checked and closed: the broadcast subtract's fidelity template parameter is unused by the custom
LLK, so LoFi would not speed it up.

Proposed for the inner loop, by size: the double pack of the probabilities (14 µs on the pack
thread; computing the row sum with the FPU reduce instead of the packer accumulate would trade
pack time for math time, and math has ~11 µs of handshake slack), the SFPU exp (14 µs; approx mode
proved it is not SFPU-op bound, so the cost is the pack-thread scheduling around it), and the
broadcast subtract (10 µs on math; a fused "exp(x - m)" on the SFPU would remove it, the current
custom LLK ignores its fidelity parameter so LoFi does not help). Each is a kernel change of a day
or more and applies to both ring ops.

## 4. Results

| stage | exp ring op | normal ring op (what the block runs) | gap |
|---|---|---|---|
| start (56 cores; lockstep does not fit L1; sequential passes segs=4 / G=2) | 206.7 ms | 192.9 ms | +7.2% |
| 64 cores via bottom-row MUX placement (segs=7 / G=1, 4 links) | 196.2 ms | 192.9 ms | +1.7% |
| + blocked pack at width 4 in the shared inner loop (**landed**) | **193.7 ms** | **191.6 ms** | +1.1% |
| + approximate exp (knob, not adopted) | 192.7 ms | | |

All on the 14336-row padded shard, max over the 32 devices. Numerics on every PCC-checkable point: 0.99975 at 3424
rows/device (both schedules, both MUX placements), 0.99972 at 4096 rows on the 64-core grid with 4 links (threshold
0.9993). The 14336-row runs are timing-only; the torch reference at 114688 tokens does not fit host memory.

In the block the normal op keeps running, and the only landed change is the inner-loop pack-4, which took the SDPA row
from 174.56 ms (2026-09-17) to 172.16 +- 0.03 ms (2026-09-21, six runs): -2.4 ms, 13 run-to-run standard deviations,
1.0% of the block. The exp ring op is not adopted: on this part it ends 1.1% behind the normal op at 15 s, because
L1 forces streamed Q and the sequential schedule, and both ops share the inner loop that holds the FPU at 48-50%.

### What was changed in the op and the model (`7274624d524`, `3559b70d163`, `f9aa767a918`, `b4736d61242`)

#### Op (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/`)

* `kernels/dataflow/exp_ring_joint_writer.cpp`: fabric packet-header rotation sized from the per-arch
  pool (`NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType`: 8 on Wormhole, 12 on Blackhole). The op
  used 11; `PacketHeaderPool::allocate_header` spins forever on exhaustion. This was the first hang.
* `kernels/dataflow/exp_ring_joint_reader.cpp`: per-link semaphore array `[2]` -> `[4]`. With 4
  links the overflow corrupted the reader's stack and every reader exited without work (second hang).
* `exp_ring_joint_sdpa_program_factory.cpp` / `_device_operation.cpp`: MUX-client layout generalized
  from two hard-coded columns to `num_links` columns (2 or 4), including the plain-writer range,
  termination masters, client ranges and the cache-hit runtime-arg patch; MUX kernels fill the
  reserved column for `num_links > 2`. `kMaxPasses` 3 -> 4.
* **Bottom-row MUX on Wormhole** (`TT_EXP_SDPA_MUX_BOTTOM_ROW=1`, pre-existing Blackhole placement
  experiment): `exp_sdpa_grid_for_user_grid` in the header is the single grid derivation for the
  factory build, the cache-hit patch and validation. It gives up the bottom row to the MUX kernels and
  an extra idle row only when the remainder is odd (8x9 -> 8x8 on Wormhole; 13x10 -> 13x8 on Blackhole,
  unchanged). The bottom-row MUX list is generalized to `num_links` cores per direction (backward at
  columns `[0, num_links)`, forward at `[num_links, 2*num_links)`); 2 links keep the measured
  placement. The 2-link-only TT_FATAL is gone; the reserved-row width check replaces it.
* **Sequential passes** (`TT_EXP_SDPA_Q_GROUPS=G`, read in `exp_ring_joint_sdpa_program_factory.hpp`):
  all three kernels map `(outer, inner)` onto `(ring_iter, pass)` in either order; in sequential mode
  one pass runs every ring iteration before the next starts, so one Q chunk and one flash state are
  live (the normal op's `q_per_core == 1` scratch path, no L1 state FIFO). A head-segment's Q chunks
  are split into G groups of one chunk per column, walked as extra passes; only group 0 forwards K/V
  over the fabric, later groups re-read the gathered K/V from DRAM. The pass cap does not apply in
  this mode. Constraint: `num_q_chunks % (columns x G) == 0`.

#### Model and tests

* `attention_minimax_h3.py`: `MINIMAX_H3_EXP_RING_SDPA=1/0` replaces the Blackhole-only enable rule;
  rows rounded to even; `num_workers_per_link = rows / 2`; `MINIMAX_H3_EXP_RING_NUM_LINKS` (default 2)
  and `MINIMAX_H3_EXP_RING_MAX_PASSES` (default 3) knobs. **The model does not yet know about
  sequential mode or the 14336-row padding** (see §5, pipeline integration).
* `test_exp_ring_joint_attention.py`: Wormhole cases `4x8_wh_h3_sim32{,_nl4,_p4,_seq}` (3424 rows,
  PCC-checked), `4x8_wh_h3_4096_bot_nl4` (4096 rows, 16 chunks = 8 x 2, PCC-checked on the 64-core
  grid), `4x8_wh_h3_15s_seq{,_nl4}` (14336 rows, timing-only) and the chunk-sweep cases
  `4x8_wh_h3_15s_q{128,192,448}_nl4`; grid rows rounded to even; columns chosen as the widest divisor
  of the chunk count; with `TT_EXP_SDPA_MUX_BOTTOM_ROW` set the helper sizes the user grid as
  `(cols, rows + 1)` with all `full_grid.x` columns available and logs the resulting grid; 4 KB fabric
  payload on Wormhole.
* `test_ring_joint_sdpa.py`: normal-op configs `minimax_h3_15s_768p_sim32` (3424) and
  `minimax_h3_15s_768p_pad14336`.
* `test_performance_minimax_h3.py`: `sp_sim4` runs on Wormhole; `MINIMAX_H3_EXP_RING_SDPA=0` keeps
  the normal op there for the in-block A/B.

## 5. What is left, ranked

1. **The softmax half of the inner loop (shared by both ring ops; the only route past ~50% FPU).** Compute the row
   sum with an FPU reduce instead of the second pack of every probability tile (14 us on the pack thread; math has
   ~11 us of handshake slack); fuse the row-max subtract into the SFPU exp (removes 10 us on math and a DST round
   trip; the custom LLK ignores its fidelity parameter so LoFi is not a shortcut); interleave the exp of
   column-subblock k with the pack of k-1 (approx mode showed the EXP zone is scheduling, not throughput). Each is
   a day or more in `compute_streaming.hpp`.
2. **One forwarder per (pass, head)** (§3.3): ~2% plus the device spread; makes 2 links as good as 4.
3. **The pass imbalance** (§3.3): 6%, structural at q=256 on 8 rows; accept, or place the joint (text) chunks on
   the two 13-pass rows.
4. **Pipeline integration**, only if the exp op is ever adopted:

* `_build_exp_sdpa_program_config` in `attention_minimax_h3.py` still models the lockstep L1 footprint
  (`_exp_sdpa_l1_bytes` with `passes` resident) and returns `None` for the 15 s shard. Add a sequential
  branch: footprint is the single-pass one, choose `(columns, G, segs)` with
  `num_q_chunks % (columns x G) == 0`; on Wormhole use the bottom-row grid `(full.x, full.y)` with
  `full.x` SDPA columns and `(full.y - 1)` rounded to even rows. Prefer the layout with the fewest
  `ceil(B x NH x segs / rows) x G` chunk-passes per core (13 at 15 s), then the fewest forwards.
* 13664 rows/device give 54 chunks of q=256, which neither 7 nor 8 columns divide. Pad the packed
  sequence to 14336 rows/device in `packing.py:padded_sequence_length` (+4.9% attention work, and every
  `M=13664`-keyed matmul table entry must be re-keyed; the pipeline runs 13664 rows/device at 15 s). Padding to
  14336 is what the numbers above assume. (q=192 at 13824 rows divides but measured 12% slower.)
* Turn `TT_EXP_SDPA_Q_GROUPS` and `TT_EXP_SDPA_MUX_BOTTOM_ROW` into op parameters (program config or
  `ExpRingJointSDPAParams`) so they are in the program-cache key; same for the `num_workers_per_link`
  derivation. Keep `kMaxPasses` semantics.
* Then the same-host pipeline A/B (README Part 4, *Optimization target*): `test_pipeline_minimax_h3.py -k 4x8nl4`,
  15 s / 16:9 only, exp on vs off, ms/fwd and CLIP (bar 33.0; 36.31 on the other host's baseline, 35.88 on this host's later run).

5. **Correctness at 15 s.** The torch reference is infeasible at 114688 tokens. Add an on-device check: run the normal ring op on the same inputs and compare outputs (PCC on device or via `to_torch` of both). The 3424-row PCC covers the kernel logic; this covers the 14336-row schedule (56 chunks, 7 segment-passes, group forwarding).
6. **Blackhole regression.** The lockstep path is unchanged in intent but the three kernels were restructured (loop mapping) and the writer's header count is now derived. The bottom-row grid helper reproduces the old 13x8 on the 13x10 Blackhole grid, and the 2-link bottom-row MUX placement is untouched, but `TT_EXP_SDPA_MUX_BOTTOM_ROW` should be re-run there once. Run the Blackhole cases of `test_exp_ring_joint_attention.py` and the `test_exp_ring_joint_sdpa.py` perf gate (65.3% / 65.6%) on a Blackhole galaxy before merging.
7. **Hygiene:** `dst_full_sync_en` is now forwarded by the exp factory so the host's subblock search and the kernel
   agree; the normal ring factory should forward it too before anyone sets it there.

## 6. Tooling and recipes

### How to run things

Every device run goes through `scripts/run_safe_pytest.sh` (device lock, 5 s dispatch timeout, triage
and reset on hang). Pass node ids, not `-k`. For `--profile` runs wrap the node id in embedded single
quotes so the Tracy re-shell does not glob the brackets; for plain runs do not.

```bash
UT=models/tt_dit/tests/unit/test_exp_ring_joint_attention.py::test_exp_ring_joint_sdpa_dit_bh_glx_custom

# correctness, 3424-row shard, lockstep schedule
scripts/run_safe_pytest.sh "${UT}[wormhole_b0-4x8_wh_h3_sim32-ring]" --timeout 3600
# correctness, sequential schedule
TT_EXP_SDPA_Q_GROUPS=2 scripts/run_safe_pytest.sh "${UT}[wormhole_b0-4x8_wh_h3_sim32_seq-ring]" --timeout 3600
# correctness on the 64-core bottom-row grid, 4 links
TT_EXP_SDPA_MUX_BOTTOM_ROW=1 TT_EXP_SDPA_Q_GROUPS=1 scripts/run_safe_pytest.sh "${UT}[wormhole_b0-4x8_wh_h3_4096_bot_nl4-ring]" --timeout 3600
# 15 s shard timing, best layout (64 cores, segs=7, 4 links): 196.2 ms before pack-4, 193.7 ms with it
TT_EXP_SDPA_MUX_BOTTOM_ROW=1 TT_EXP_SDPA_Q_GROUPS=1 scripts/run_safe_pytest.sh --profile "'${UT}[wormhole_b0-4x8_wh_h3_15s_seq_nl4-ring]'" --timeout 3600
# previous best on the reserved-column 56-core grid (segs=4, G=2, 4 links): 206.7 ms
TT_EXP_SDPA_Q_GROUPS=2 scripts/run_safe_pytest.sh --profile "'${UT}[wormhole_b0-4x8_wh_h3_15s_seq_nl4-ring]'" --timeout 3600
# normal-op baseline at the same rows (self-profiling; do NOT wrap in --profile)
scripts/run_safe_pytest.sh "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table[minimax_h3_15s_768p_pad14336]" -s --timeout 3600
# in-block A/B at the simulated shard: exp (passes 3 or 4) vs normal
MINIMAX_H3_EXP_RING_SDPA=1 MINIMAX_H3_EXP_RING_MAX_PASSES=4 scripts/run_safe_pytest.sh --profile \
  "'models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim4-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]'" -s --timeout 3600
MINIMAX_H3_EXP_RING_SDPA=0 scripts/run_safe_pytest.sh --profile "'…same node id…'" -s --timeout 3600
```

Profiled runs print `SAFE_PYTEST: PROFILER CSV: <path>`; the op's time is the max over devices of
`DEVICE KERNEL DURATION [ns]` for `ExpRingJointSDPADeviceOperation` (the same statistic
`tools/project_block_perf.py` uses). Hangs leave a callstack dump in `generated/tt-triage/triage.csv`
(`dump_callstacks.py` section); aggregating the top frame per kernel is how both hangs were found.

Rebuilding the host library after factory changes: `cmake --build build_Release --target install`
(about 30 s; do not run while a test holds the device). Kernel sources are JIT-compiled per run. Do
not run `build_metal.sh`: it recreates `python_env` and removes the pinned `diffusers` fork.

The env knobs (`TT_EXP_SDPA_Q_GROUPS`, `TT_EXP_SDPA_MUX_*`) are **not part of the program-cache key**:
one setting per process. Per-call op time from a profiled run: max over devices of `DEVICE KERNEL
DURATION [ns]` for `ExpRingJointSDPADeviceOperation`, matched by per-device call position
(`models/tt_dit/tests/models/minimax_h3/tools/op_time_from_profiler_csv.py <PROFILER CSV> ExpRingJoint`; `GLOBAL CALL
COUNT` is per-device and must not be used as the join key).

### Where a step goes

`TT_EXP_SDPA_PROFILE_INNER=1` + `--profile` on the exp op compiles the per-phase zones of `sdpa_inner_loop_step` in;
the report's `profile_log_device.csv` then holds start and end timestamps per phase, per core and per compute thread.
The L1 profiler buffer holds about 125 zones per thread per launch, so only the first step and a half of each core
is recorded, which is enough for a per-step split because every step does the same work.
`models/tt_dit/tests/models/minimax_h3/tools/sdpa_phase_zones.py` prints the per-thread table for one core (§2.2).

### Gotchas that cost time

* `run_safe_pytest.sh --profile` masks pytest's exit code; read the PASSED/FAILED line.
* Wormhole's fabric payload cap is 7616 B; `create_fabric_router_config(8192)` throws. The unit test
  uses 4 KB on Wormhole.
* A `for (pass)` -> `(outer, inner)` restructure moved the lockstep-mode Q pop inside the inner loop
  once; it over-popped without failing PCC. The fixed version is committed; keep the regression case
  (`4x8_wh_h3_sim32`) in any future kernel change.
* Do not grep a monitor for `L1`: the JIT kernel compile lines contain it and look like errors.
* Do not edit `scripts/run_safe_pytest.sh` while an instance is running: bash reads the script by byte
  offset, so the running instance resumes mid-line after the child exits (one run lost its result line).
* `SAFE_PYTEST_TRACY_OPTS="--profiler-capture-perf-counters=..."` deadlocks: Tracy's multi-pass
  capture spawns an inner `python -m tracy` that waits on a UMD chip lock held by its parent. Kill the
  parent PID; the wrapper then releases the device lock.
* Full-sync DST (`dst_full_sync_en=True`) is a 36% loss on this kernel; do not retry it as a "free"
  bigger-subblock knob.
* `q_chunk < 256` is not a lever in sequential mode: time follows passes x K chunks (inner-loop
  steps), so q=192 lost 12% and q=128 lost 71% despite fewer tile-rows per core.
* Pre-commit's `black` targets a newer Python than `python_env`'s; it reformats
  `test_ring_joint_sdpa.py` on commit, so expect one hook retry.
* Git identity is unset on this box; commit with `GIT_AUTHOR_*` / `GIT_COMMITTER_*` env vars.
