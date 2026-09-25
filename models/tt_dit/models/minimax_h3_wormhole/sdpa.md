# MiniMax-H3 on Wormhole Galaxy: the ring joint SDPA — baseline, zones, experiments, results

The attention op of the transformer block, `RingJointSDPADeviceOperation` (the ring joint SDPA), and the work done
on it between 2026-09-17 and 2026-09-21 on the 4x8 Wormhole Galaxy `UF-EV-B12-GWH02`, branch
`minimax_h3_wh_optimizations`. Ordered as baseline -> zone breakdown -> experiments per zone -> results.
Block-level context: [README.md](README.md).

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

The utilisation figures in the sources are one fact seen through different instruments: 48.3% is Tracy's
`PM FPU UTIL` (the perf-model ideal divided by measured time, not a hardware counter), 48% is the roofline
cross-check, 47.3-48.1% the isolated sweep's reading of the same column, and "50% of the step" is the inner-loop
zone arithmetic (§2.2: 33 us of FPU matmul in a 66 us step). Likewise the L1 budget appears as 1,499,136 B (the
part's maximum), 1.31 / 1.34 MB (usable after reserved regions, as the op's allocator and the CB check see it).

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
carries per-phase device zones behind a `profiling_enabled` template flag; the table below was taken with that
flag compiled in on an instrumented build. The L1 profiler buffer holds ~125 zones per RISC per launch, so the
log covers the first step and a half of each core, which is what the table uses (start and end timestamps per
phase, per core and per compute thread in the report's `profile_log_device.csv`). Step 0 on the math
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
2. A/B a knob that touches one resource: dropping math fidelity would test the FPU that way.
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

### 3.2 The inner loop (shared kernel)

The op on the 14336-row padded shard, 192.9 ms base; max over 32 devices.

| change | result | verdict |
|---|---|---|
| A. `MIN_BLOCKED_PACK_TILES` 8 -> 4 on Wormhole: one pack per 4-wide subblock row instead of four single-tile packs | 192.9 -> **191.6 ms** | kept |
| B. hardware counters via `SAFE_PYTEST_TRACY_OPTS="--profiler-capture-perf-counters=fpu,pack,unpack"` | no data | Tracy multi-pass capture deadlocks on this box |

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

The one landed change is the inner-loop blocked pack at width 4 (§3.2 A). Isolated, on a 14336-row padded shard (max over
the 32 devices), the op went 192.9 -> **191.6 ms**.
In the block it took the SDPA row from 174.56 ms (2026-09-17) to **172.16 +- 0.03 ms** (2026-09-21, six runs):
-2.4 ms, 13 run-to-run standard deviations, 1.0% of the block. Chunk-size tuning is exhausted (§3.1) and the
inner loop holds the FPU at 48-50% (§2.2), so what is left is kernel work inside `sdpa_inner_loop_step`.

## 5. What is left, ranked

1. **The softmax half of the inner loop (shared by both ring ops; the only route past ~50% FPU).** Compute the row
   sum with an FPU reduce instead of the second pack of every probability tile (14 us on the pack thread; math has
   ~11 us of handshake slack); fuse the row-max subtract into the SFPU exp (removes 10 us on math and a DST round
   trip; the custom LLK ignores its fidelity parameter so LoFi is not a shortcut); interleave the exp of
   column-subblock k with the pack of k-1 (approx mode showed the EXP zone is scheduling, not throughput). Each is
   a day or more in `compute_streaming.hpp`.
2. **Hygiene:** the ring factory does not forward `dst_full_sync_en`, so the host's subblock search and the kernel
   can disagree if anyone sets it; forward it before that happens.

## 6. Tooling and recipes

### How to run things

Every device run goes through `scripts/run_safe_pytest.sh` (device lock, 5 s dispatch timeout, triage
and reset on hang). Pass node ids, not `-k`. For `--profile` runs wrap the node id in embedded single
quotes so the Tracy re-shell does not glob the brackets; for plain runs do not.

```bash
# isolated op at the pipeline's 15 s rows (self-profiling; do NOT wrap in --profile)
scripts/run_safe_pytest.sh "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table[minimax_h3_15s_768p]" -s --timeout 3600
# in the block (one duration; prints "SAFE_PYTEST: PROFILER CSV: <csv>")
scripts/run_safe_pytest.sh --profile \
  "'models/tt_dit/tests/models/minimax_h3/test_transformer_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim1-test_prompt_text_tokens-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]'" -s --timeout 3600
```

Profiled runs print `SAFE_PYTEST: PROFILER CSV: <path>`; the op's time is the max over devices of
`DEVICE KERNEL DURATION [ns]` for `RingJointSDPADeviceOperation` (the same statistic
`tools/project_block_perf.py` uses), matched by per-device call position
(`models/tt_dit/tests/models/minimax_h3/tools/op_time_from_profiler_csv.py <PROFILER CSV> RingJoint`; `GLOBAL CALL
COUNT` is per-device and must not be used as the join key). Hangs leave a callstack dump in
`generated/tt-triage/triage.csv` (`dump_callstacks.py` section); aggregating the top frame per kernel is how hangs
are found.

Rebuilding the host library after factory changes: `cmake --build build_Release --target install`
(about 30 s; do not run while a test holds the device). Kernel sources are JIT-compiled per run. Do
not run `build_metal.sh`: it recreates `python_env` and removes the pinned `diffusers` fork.

### Gotchas that cost time

* `run_safe_pytest.sh --profile` masks pytest's exit code; read the PASSED/FAILED line.
* Wormhole's fabric payload cap is 7616 B; `create_fabric_router_config(8192)` throws. The 4x8_WH mesh row
  opens with 4 KB.
* Do not grep a monitor for `L1`: the JIT kernel compile lines contain it and look like errors.
* Do not edit `scripts/run_safe_pytest.sh` while an instance is running: bash reads the script by byte
  offset, so the running instance resumes mid-line after the child exits (one run lost its result line).
* `SAFE_PYTEST_TRACY_OPTS="--profiler-capture-perf-counters=..."` deadlocks: Tracy's multi-pass
  capture spawns an inner `python -m tracy` that waits on a UMD chip lock held by its parent. Kill the
  parent PID; the wrapper then releases the device lock.
* Pre-commit's `black` targets a newer Python than `python_env`'s; it reformats
  `test_ring_joint_sdpa.py` on commit, so expect one hook retry.
* Git identity is unset on this box; commit with `GIT_AUTHOR_*` / `GIT_COMMITTER_*` env vars.
