# Optimizing the exp ring joint SDPA for MiniMax-H3 on the Wormhole 4x8 Galaxy

Written 2026-09-18 on `UF-EV-B12-GWH02`, branch `jameslee/exp_ring_sdpa_wh` (commits `7274624d524`,
`3559b70d163`, `f9aa767a918`, `b4736d61242`). This is the profiling-and-optimization log for
`exp_ring_joint_scaled_dot_product_attention` at the H3 target shape: what was measured, how, what each
change did, and what is left. The operational handoff is `MiniMaxH3_wormhole_exp_ring_sdpa_handoff.md`;
the block-level context is `MiniMaxH3_wormhole_perf.md`.

Target: 15 s / 768p / 16:9, TP=4 / SP=8, 13664 rows/device padded to 14336, 14 heads/device, head dim
128, HiFi2 bf16, ring of 8 devices on 4 fabric links. Every number below is one op call, max over the
32 devices, from the Tracy device profiler.

## 1. Result in one table

| stage | exp ring op | normal ring op | gap |
|---|---|---|---|
| start of day (56 cores, lockstep does not fit L1; sequential passes segs=4 / G=2) | 206.7 ms | 192.9 ms | +7.2% |
| 64 cores via bottom-row MUX placement (segs=7 / G=1, 4 links) | 196.2 ms | 192.9 ms | +1.7% |
| + blocked pack at width 4 in the shared inner loop | **193.7 ms** | **191.6 ms** | +1.1% |
| + approximate exp (knob, not adopted) | 192.7 ms | | |

Numerics on every PCC-checkable point: 0.99975 at 3424 rows/device, 0.99972 at 4096 rows on the 64-core
grid (threshold 0.9993). The 14336-row runs are timing-only; the torch reference at 114688 tokens does
not fit host memory.

## 2. How to measure

**Op time.** Run the unit test under the device profiler and take the max over devices of
`DEVICE KERNEL DURATION [ns]`, matched by per-device call position (`GLOBAL CALL COUNT` is per-device
and must not be used as the join key). `models/tt_dit/tests/models/minimax_h3/tools/op_time_from_profiler_csv.py`
does this; the block projector uses the same statistic.

```bash
UT=models/tt_dit/tests/unit/test_exp_ring_joint_attention.py::test_exp_ring_joint_sdpa_dit_bh_glx_custom
# best layout: 64 cores (bottom-row MUX), sequential passes with G=1, 4 links, 15 s shard, timing only
TT_EXP_SDPA_MUX_BOTTOM_ROW=1 TT_EXP_SDPA_Q_GROUPS=1 scripts/run_safe_pytest.sh --profile \
  "'${UT}[wormhole_b0-4x8_wh_h3_15s_seq_nl4-ring]'" --timeout 3600
python models/tt_dit/tests/models/minimax_h3/tools/op_time_from_profiler_csv.py <PROFILER CSV> ExpRingJoint
# PCC on the 64-core grid (4096 rows/device, 16 chunks = 8 columns x 2 segments)
TT_EXP_SDPA_MUX_BOTTOM_ROW=1 TT_EXP_SDPA_Q_GROUPS=1 scripts/run_safe_pytest.sh \
  "${UT}[wormhole_b0-4x8_wh_h3_4096_bot_nl4-ring]" --timeout 3600
# normal-op reference at the same rows (self-profiling; no --profile)
scripts/run_safe_pytest.sh "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table[minimax_h3_15s_768p_pad14336]" -s --timeout 3600
```

**Where a step goes.** The compute kernel's inner loop (`sdpa_inner_loop_step` in
`compute_streaming.hpp`, shared by both ring ops) has per-phase device zones behind a template flag.
`TT_EXP_SDPA_PROFILE_INNER=1` compiles them in for the exp op; the report's `profile_log_device.csv`
then holds, per core and per compute thread (unpack / math / pack), start and end timestamps for each
phase. The L1 profiler buffer holds about 125 zones per thread per launch, so only the first step and a
half of each core is recorded, which is enough for a per-step split because every step does the same
work. `tools/sdpa_phase_zones.py` prints the per-thread table for one core.

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

## 3. The inner-loop step, measured

Per step (one Q chunk of 256 rows against one K chunk of 512 rows), core (1,1) of device 0, with
blocked pack at width 4:

| thread | matmul zones (QK + PV) | softmax zones | outside all leaf zones |
|---|---|---|---|
| unpack (TRISC_0) | 16.6 + 21.1 µs | SUB 8.7, reduce 3.1 | 17.7 µs |
| math (TRISC_1) | ~19 (steady) + 21.2 µs | SUB 10.1, init 1.9, reduce 0.8 | 10.9 µs |
| pack (TRISC_2) | 22.2 + 2.2 µs | EXP 14.4, PACK SUB_EXP 14.0, reduce 1.7 | 24.9 µs |

Reading. 2912 steps per core per call at 66 µs is the whole 192 ms; nothing is hidden in waits after
the first chunk. The pure matmul is 33 µs (50%). The other half: on the pack thread the SFPU exp and
the packing of every probability tile twice (in place, then with packer L1-accumulate into the row-sum
tile) cost 28 µs; on the math thread the broadcast subtract of the row max is 10 µs (128 tiles at ~80
cycles) and 11 µs sit between zones in `tile_regs_acquire` / `tile_regs_wait` handshakes. The chunk sweep
agrees: time follows the number of steps (passes x K chunks), not FLOPs or bytes.

## 4. What was tried, in order

### 4.1 Sequential passes so the shard fits L1 (earlier commit `3559b70d163`)

The lockstep schedule keeps every pass's Q chunk and flash state resident; at 13664 rows/device no
configuration fits Wormhole's 1.31 MB. `TT_EXP_SDPA_Q_GROUPS=G` runs pass-outer / ring-inner so one Q
chunk and one state are live, and splits a head-segment's chunks into G groups (only group 0 forwards
K/V; later groups re-read the gathered K/V from DRAM). Constraint `num_q_chunks % (columns x G) == 0`
forces the 13664 rows to be padded to 14336 (56 chunks of q=256). Result on 56 cores: 206.7 ms
(segs=4, G=2, 4 links); segs=2 with pair dedup was 238 ms (row imbalance costs more than duplicate
forwarding saves).

### 4.2 64 cores: bottom-row MUX placement (`f9aa767a918`). Kept. 206.7 -> 196.2 ms

The reserved MUX column leaves 7 columns, and the 9 rows round down to 8 for the equal backward /
forward MUX-client halves: 56 cores. The existing `TT_EXP_SDPA_MUX_BOTTOM_ROW` experiment (Blackhole)
puts the MUX kernels on the bottom row and gave up two rows to stay even; on Wormhole one row suffices,
so SDPA keeps all 8 columns and rows 0-7 (64 cores) and the 8 MUX kernels of 4 links fill row 8.
Host-side only: one shared grid helper for build / validation / cache-hit patch, and the bottom-row MUX
list generalized to `num_links` cores per direction. With 8 columns the 56 chunks split 8 x 7, so G=1
and segs=7: 98 segments on 8 rows -> 13 passes on two rows, 12 on the rest (ideal 12.25). Every pass
forwards, so 13 forwards per row against 7 before, which is why 4 links beat 2 (196.2 vs 199.6) and
why the result sits 4 ms above the pure work model (206.7 x 13/14 = 192).

### 4.3 Chunk-size sweep on 64 cores. Nothing beat q256 / k512

| shape | per call | why |
|---|---|---|
| q448 / k256 (7 passes of 14 tile-rows) | 200.9 ms | the halved K chunk costs more than the larger Q chunk saves |
| q192 / k512 at 13824 rows (72 chunks = 8 x 9, 3.6% less padding) | 220.2 ms | 8% fewer tile-rows but 19% more steps |
| q128 / k512 | 336.2 ms | steps nearly double |
| q448 / k512 | does not build | CBs need 1.71 MB of 1.34 MB usable L1 |

### 4.4 Inner-loop experiments (`b4736d61242`), shared with the normal op

| change | exp op | normal op | verdict |
|---|---|---|---|
| A. `MIN_BLOCKED_PACK_TILES` 8 -> 4 on Wormhole: one pack per 4-wide subblock row instead of four single-tile packs | 196.2 -> **193.7 ms** | 192.9 -> **191.6 ms** | kept |
| B. full-sync 16-tile DST (`dst_full_sync_en`; the exp factory now forwards it so the host's subblock search and the kernel agree) | 262.8 ms | | rejected: the half-sync ping-pong that overlaps math and pack is worth far more than larger subblocks |
| C. approximate SFPU exp (`TT_EXP_SDPA_TEST_EXP_APPROX=1`) | 192.7 ms, PCC unchanged | | 0.5%: the EXP zone is not SFPU-op bound; the model keeps exact exp |
| D. hardware counters via `SAFE_PYTEST_TRACY_OPTS="--profiler-capture-perf-counters=fpu,pack,unpack"` | no data | | Tracy multi-pass capture deadlocks on this box |

Also checked and closed: the broadcast subtract's fidelity template parameter is unused by the custom
LLK, so LoFi would not speed it up.

## 5. What is left, ranked

1. **Softmax half of the inner loop (shared, the only route past 50% FPU).** Compute the row sum with
   an FPU reduce instead of the second pack of every probability tile (14 µs on the pack thread, math has
   ~11 µs of handshake slack); fuse the row-max subtract into the SFPU exp (removes 10 µs on math and a
   DST round trip); interleave the exp of column-subblock k with the pack of k-1 (approx mode showed the
   EXP zone is scheduling, not throughput). Each is a day or more in `compute_streaming.hpp` and moves
   both ring ops.
2. **One forwarder per (pass, head).** At segs=7 each row forwards its head's K/V on all 13 passes
   while only 2-3 heads are live per pass. A per-pass leader election (lowest row in the direction half
   with that head) cuts forwards per row from 13 to about 4 and takes the fabric off the critical path
   even on 2 links. Worth ~2% (the 2-vs-4-link gap) plus the device spread.
3. **Pass imbalance (6%, structural).** 98 segments on 8 rows is 13 vs 12.25 at q=256; no chunk shape
   fixes it without adding steps. Accept it, or place the joint (text) chunks on the two 13-pass rows if
   the pipeline has any.
4. **Pipeline integration.** The model's grid search must emit the bottom-row `(8, 9)` grid with 8 SDPA
   columns and `num_workers_per_link=4`, pad to 14336 rows/device (re-keying the `M=13664` matmul
   tables), and the env knobs (`TT_EXP_SDPA_Q_GROUPS`, `TT_EXP_SDPA_MUX_BOTTOM_ROW`) must become op
   parameters so they enter the program-cache key. Then the same-host pipeline A/B (ms/fwd and CLIP).

## 6. Gotchas that cost time today

* Do not grep a log monitor for `L1`: the JIT kernel compile lines contain it.
* Do not edit `scripts/run_safe_pytest.sh` while an instance is running; bash reads it by byte offset.
* Full-sync DST is a 36% loss on this kernel. Q chunks below 256 are a 12-71% loss.
* Tracy's counter capture deadlocks here; kill the parent `python -m tracy` PID to free the device lock.
* Env knobs are not in the program-cache key: one setting per process.
