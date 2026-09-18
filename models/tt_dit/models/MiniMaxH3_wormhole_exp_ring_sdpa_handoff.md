# Handoff: exp ring joint SDPA for MiniMax-H3 on the Wormhole 4x8 Galaxy

Written 2026-09-18 on `UF-EV-B12-GWH02`, updated the same day. Branch **`jameslee/exp_ring_sdpa_wh`**
(off `jameslee/bringup_h3_wh_galaxy` at `1361ebabb85`), pushed to origin:

| commit | what |
|---|---|
| `7274624d524` | brings `exp_ring_joint_scaled_dot_product_attention` up on Wormhole: two hang fixes, 2-or-4-link MUX layout, model/test knobs, first A/B |
| `3559b70d163` | sequential passes (`TT_EXP_SDPA_Q_GROUPS`) so the 15 s shard fits L1; measured against the normal op |
| (this commit) | bottom-row MUX placement on Wormhole: 8x8 = 64 SDPA cores; chunk-size sweep at 15 s |

Measurements and the reasoning behind each step are in `MiniMaxH3_wormhole_perf.md`, sections
*Exp ring joint SDPA on Wormhole*, *Sequential passes* and *Bottom-row MUX placement*. This document is
the operational summary and the ranked list of what to do next.

## 1. Where things stand

Target: 15 s / 768p / 16:9, TP=4 / SP=8, 13664 rows/device, 14 heads/device, HiFi2 bf16. The normal
`RingJointSDPADeviceOperation` takes 174.6 ms of a 246.9 ms transformer block (71%).

| op | shard | layout | per call (max over 32 devices) |
|---|---|---|---|
| normal ring op | 14336 rows (15 s padded) | q256 / k512, 63 cores | **192.9 ms** |
| exp, sequential, bottom-row MUX | 14336 rows | q256 / k512, segs=7, G=1, 4 links, **64 cores** | **196.2 ms** |
| exp, sequential, bottom-row MUX | 14336 rows | same, 2 links | 199.6 ms |
| exp, sequential, bottom-row MUX | 14336 rows | q448 / k256, segs=4, G=1, 4 links, 64 cores | 200.9 ms |
| exp, sequential | 14336 rows | q256 / k512, segs=4, G=2, 4 links, 56 cores (reserved-column MUX) | 206.7 ms |
| exp, sequential | 14336 rows | segs=4, G=2, 2 links, 56 cores | 212.6 ms |
| exp, sequential, bottom-row MUX | 13824 rows | q192 / k512, segs=9, 64 cores | 220.2 ms |
| exp, sequential | 14336 rows | segs=2 (pair dedup), G=4, 2 or 4 links, 56 cores | 238.0 / 238.2 ms |
| exp, sequential, bottom-row MUX | 14336 rows | q128 / k512, segs=14, 64 cores | 336.2 ms |
| normal ring op | 3424 rows (SP=32-equivalent) | q256 / k512 | 15.74 ms (14.76 in-block) |
| exp, lockstep | 3424 rows | q256 / k256, 4 passes, 2 links | 16.65 ms (16.09 in-block) |

Numerics: PCC 0.99975 against torch on every exp point that has a feasible reference (3424 rows, both
schedules, both MUX placements) and 0.99972 on the 64-core grid at a 4096-row shard with 4 links. The
14336-row runs are timing-only; the torch reference at 114688 tokens does not fit host memory.

**Reading.** The exp op runs the 15 s shard on Wormhole at 196.2 ms, 1.7% behind the normal op. The
64-core layout (`TT_EXP_SDPA_MUX_BOTTOM_ROW=1`) recovered most of the 7% gap the reserved MUX column
cost: with 8 columns the 56 chunks split as 8 x 7 segments, 98 segments on 8 rows -> 13 passes on rows
0-1 and 12 on the rest (ideal 12.25), so per-core work drops 14 -> 13 chunk-passes. Every pass forwards
its head's K/V (13 forwards per row vs 7 before), which is why 4 links beat 2 by 3.4 ms and why the
result is 4 ms above the pure work model (192 ms). The chunk sweep shows the time follows inner-loop
steps (passes x K chunks), not Q tile-rows, so q256 / k512 stays. Both ops share the same inner loop
(`sdpa_inner_loop_step` in `compute_streaming.hpp`), so neither touches the Wormhole utilization
ceiling (48% FPU at 15 s, vs about 70% for the same kernel on Blackhole).

## 2. What was changed

### Op (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/`)

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

### Model and tests

* `attention_minimax_h3.py`: `MINIMAX_H3_EXP_RING_SDPA=1/0` replaces the Blackhole-only enable rule;
  rows rounded to even; `num_workers_per_link = rows / 2`; `MINIMAX_H3_EXP_RING_NUM_LINKS` (default 2)
  and `MINIMAX_H3_EXP_RING_MAX_PASSES` (default 3) knobs. **The model does not yet know about
  sequential mode or the 14336-row padding** (see §4.3).
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

## 3. How to run things

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
# 15 s shard timing, best layout so far (64 cores, segs=7, 4 links): 196.2 ms
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
DURATION [ns]` for `ExpRingJointSDPADeviceOperation`, matched by per-device call position (a 20-line
CSV script; `GLOBAL CALL COUNT` is per-device and must not be used as the join key).

## 4. What to do next, ranked

### 4.1 Done: the cores are back (bottom-row MUX, 64 cores)

The 9-row asymmetric-halves layout this section proposed is superseded: `TT_EXP_SDPA_MUX_BOTTOM_ROW=1`
gives 8x8 = 64 symmetric cores with host-only changes (see §2). Result 206.7 -> 196.2 ms. The
bottom-row layout should become the Wormhole default when the op is wired into the model (§4.3), which
also means the model's grid search must produce `(8, 9)` with 8 SDPA columns and `num_workers_per_link=4`.

### 4.2 One forwarder per (pass, head) — the remaining ~2% and the 2-link path (largest, next)

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

### 4.2b The pass imbalance (6%, structural)

98 segments on 8 rows -> 13 vs 12.25. Nothing in the chunk sweep fixes it: q=192/128 reduce the
tile-row imbalance but add inner-loop steps and lose 12-71%; q=448 fits only with k=256 and loses 2%.
The only balanced q=256 layouts need a row count dividing 98 (7 or 14 rows) or a segment count
dividing 8 x k: e.g. a 7-row SDPA grid (56 cores, 14 passes exactly, worse) or 16 heads. Accept it, or
treat the two 13-pass rows as the place to put the joint (text) chunks if the pipeline has any.

### 4.3 Pipeline integration

Nothing runs the exp op in the pipeline yet.

* `_build_exp_sdpa_program_config` in `attention_minimax_h3.py` still models the lockstep L1 footprint
  (`_exp_sdpa_l1_bytes` with `passes` resident) and returns `None` for the 15 s shard. Add a sequential
  branch: footprint is the single-pass one, choose `(columns, G, segs)` with
  `num_q_chunks % (columns x G) == 0`; on Wormhole use the bottom-row grid `(full.x, full.y)` with
  `full.x` SDPA columns and `(full.y - 1)` rounded to even rows. Prefer the layout with the fewest
  `ceil(B x NH x segs / rows) x G` chunk-passes per core (13 at 15 s), then the fewest forwards.
* 13664 rows/device give 54 chunks of q=256, which neither 7 nor 8 columns divide. Pad the packed
  sequence to 14336 rows/device in `packing.py:padded_sequence_length` (+4.9% attention work, and every
  `M=13664`-keyed matmul table entry must be re-keyed; see the doc's *rows-per-device* notes). Padding to
  14336 is what the numbers above assume. (q=192 at 13824 rows divides but measured 12% slower.)
* Turn `TT_EXP_SDPA_Q_GROUPS` and `TT_EXP_SDPA_MUX_BOTTOM_ROW` into op parameters (program config or
  `ExpRingJointSDPAParams`) so they are in the program-cache key; same for the `num_workers_per_link`
  derivation. Keep `kMaxPasses` semantics.
* Then the same-host pipeline A/B the perf doc prescribes: `test_pipeline_minimax_h3.py -k 4x8nl4`,
  15 s / 16:9 only, exp on vs off, ms/fwd and CLIP (bar 33.0, baseline 36.31).

### 4.4 Correctness at 15 s

The torch reference is infeasible at 114688 tokens. Add an on-device check: run the normal ring op on the
same inputs and compare outputs (PCC on device or via `to_torch` of both). The 3424-row PCC covers the
kernel logic; this covers the 14336-row schedule (56 chunks, 7 segment-passes, group forwarding).

### 4.5 Blackhole regression

The lockstep path is unchanged in intent but the three kernels were restructured (loop mapping) and the
writer's header count is now derived. The bottom-row grid helper reproduces the old 13x8 on the 13x10
Blackhole grid, and the 2-link bottom-row MUX placement is untouched, but `TT_EXP_SDPA_MUX_BOTTOM_ROW`
should be re-run there once. Run the Blackhole cases of `test_exp_ring_joint_attention.py` and
the `test_exp_ring_joint_sdpa.py` perf gate (65.3% / 65.6%) on a Blackhole galaxy before merging.

### 4.6 The inner loop (shared with the normal op; the only route past ~50% FPU)

Independent of which op wins, the Wormhole inner loop is at 48% FPU while Blackhole runs the same code
at ~70%. Untested, cheap diagnostics first:

* Hardware perf counters: `python -m tracy -r --profiler-capture-perf-counters=fpu,pack,unpack -m
  "pytest …"` gives real FPU/SFPU util, packer efficiency and unpacker stalls (the `PM FPU UTIL` column
  in the ops CSV is a perf-model ratio, not a counter).
* `MIN_BLOCKED_PACK_TILES` is 8 on Wormhole vs 4 on Blackhole (`compute_streaming.hpp:170-177`); with
  width-4 subblocks every pack takes the per-tile path. One-line experiment.
* `dst_full_sync_en` is not forwarded to the ring kernels (`ComputeConfigDescriptor` in the factories);
  16-tile DST would allow (4,4) or (2,8) subblocks.
* The PV matmul accumulates over 4 L1-accumulate passes (`compute_streaming.hpp:1580-1638`); the
  latent-V path shows the single-pass DST-accumulated alternative.

## 5. Gotchas that cost time

* `run_safe_pytest.sh --profile` masks pytest's exit code; read the PASSED/FAILED line.
* Wormhole's fabric payload cap is 7616 B; `create_fabric_router_config(8192)` throws. The unit test
  uses 4 KB on Wormhole.
* A `for (pass)` -> `(outer, inner)` restructure moved the lockstep-mode Q pop inside the inner loop
  once; it over-popped without failing PCC. The fixed version is committed; keep the regression case
  (`4x8_wh_h3_sim32`) in any future kernel change.
* Do not grep a monitor for `L1`: the JIT kernel compile lines contain it and look like errors.
* `q_chunk < 256` is not a lever in sequential mode: time follows passes x K chunks (inner-loop
  steps), so q=192 lost 12% and q=128 lost 71% despite fewer tile-rows per core.
* Pre-commit's `black` targets a newer Python than `python_env`'s; it reformats
  `test_ring_joint_sdpa.py` on commit, so expect one hook retry.
* Git identity is unset on this box; commit with `GIT_AUTHOR_*` / `GIT_COMMITTER_*` env vars.
