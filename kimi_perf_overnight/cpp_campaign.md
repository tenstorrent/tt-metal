# C++ dispatch perf campaign (started 2026-06-14 ~14:20)

GOAL: reduce the per-op device-idle dispatch stall (Kimi prefill, 8x4 BH). Pinned cause: the ~69KB TENSIX
kernel-config L1 ring byte-wraps ~every op (111/120 syncs) forcing worker-completion stalls. Secondary:
launch-msg ring (62) + table (fixed: kernel_config_entry_count 8->32, committed a8239c53ea2, perf-neutral).
Metric: 1-layer warm chunk wall (baseline 24.7ms) + standalone 61-layer per-chunk (~1878ms). Correctness:
61-layer KV PCC >= 0.88 (must hold).

KNOB MAP (from L1-layout investigation):
- kernel-config ring size = worker_l1_unreserved_start - KERNEL_CONFIG_base. Enlarge via
  bh_hal_tensix.cpp:43 default_l1_kernel_config_size (69*1024). +64KB -> ring 133KB (model L1 1.40MB);
  +128KB -> ring 197KB (model L1 1.33MB). Steals 1:1 from model L1 (DEFAULT_UNRESERVED) -> PCC/OOM catches.
- launch_msg_buffer_num_entries (dev_msgs.h:399, 8) -> 16: deepens launch ring (must stay power-of-2);
  costs 8*sizeof(launch_msg_t) of mailbox L1 (raise MEM_MAILBOX_SIZE if static_assert at bh_hal_tensix.cpp:150 fires).
- smarter free policy in worker_config_buffer.cpp reserve() (free multiple entries/sync) — maybe.

## Experiment log
| # | change | build | byte-wraps | 1L warm ms | 61L per-chunk ms | PCC | verdict |
|---|--------|-------|-----------|-----------|-----------------|-----|---------|
| baseline | (committed kce=32) | - | 111 idx0 + 62 idx3 | 24.7 | 1878 | 0.965 | - |
| E1 | default_l1_kernel_config_size 69->133KB (+64KB) | building | ? | ? | ? | ? | running |

## Plan (iterate overnight; one experiment per cycle)
E1: ring 133KB. If PCC ok + faster -> E2 ring 197KB; if OOM -> ring 101KB (+32KB).
E2: ring 197KB (if E1 fit). Find max ring that fits + best perf.
E3: best ring + launch_msg_buffer_num_entries 8->16 (combined).
E4: if ring helps, smarter free policy / other dispatch sizing.
Keep the best PCC-passing perf win committed; revert dead ends.

### E1 result (ring 133KB, +64KB): config-ring byte-wraps ELIMINATED, bottleneck shifted to launch ring
idx0 BYTE-WRAP 111->0 (config ring no longer wraps; no OOM, model fits 1.40MB). BUT idx3 launch-msg ring
(size 7) byte-wraps 62->154 (now SOLE binding) -> warm wall 24.94ms ~= baseline (perf-neutral; bottleneck
shifted to launch ring). => Need to fix BOTH. E3-combined: keep ring 133KB + launch_msg_buffer_num_entries 8->16.

### E3 build FAILED: launch=16 overflows mailbox (sizeof(mailboxes_t)>12912, BH+WH). Reverted launch edits.
YELLOW FLAG: E1 dropped syncs 173->154 but 1-layer wall flat (24.94 vs 24.7) — sync count may not track
wall. DECISIVE CHECK before risky mailbox surgery: measure E1 (ring 133KB, current lib) on REAL 61-layer
per-chunk (vs 1878ms standalone). If E1 helps 61L -> pursue launch-ring + mailbox surgery; if flat -> sync
theory wrong, don't do the surgery.

### ROOT CAUSE of the 30-min hangs / profiler timeout=416 (2026-06-14 ~16:05): SELF-INFLICTED
`git checkout dev_msgs.h dev_msgs.hpp` at 15:11 FAILED SILENTLY — the 2nd pathspec (dev_msgs.hpp,
which doesn't exist) made git abort and revert NOTHING. So dev_msgs.h stayed at launch=16 (E3 edit) while
the host lib (built 14:56) was launch=8. Every run since 15:12 JIT-compiled device firmware/kernels with
the launch=16 MAILBOX layout vs launch=8 host lib => host/device mailbox MISMATCH: profiler sync marker
landed at wrong offset (timeout=416, complete=0) AND launch-msg ring mismatch corrupted dispatch (embedding
distribution hung 30+ min). TWO GLX resets did NOT fix it (it's a host/fw mismatch, not device state).
FIX: `git checkout tt_metal/hw/inc/hostdev/dev_msgs.h` ALONE -> launch=8. Smoke 1L after: profiler
complete=32 timeout=0, chunk 15.75ms, KV PCC 0.999922 PASS. LESSON: always verify `git checkout` of a
single file SUCCEEDED (check exit + re-grep the constant); never batch a real path with a bogus one.
NOTE: the firmware cache is hash-keyed by header content, so reverting is sufficient (no rebuild needed).

## CAMPAIGN RESULT (2026-06-14 16:15): dispatch-sync hypothesis FALSIFIED — abandon ring/launch levers
E1 (config ring 133KB) on REAL 61-layer standalone: **20501ms/11 = 1863.7 ms/chunk** vs baseline ~1878
=> PERF-NEUTRAL (<1%, noise). KV PCC 0.965424 PASS (== baseline). Matches 1-layer proxy (24.94 vs 24.7).
NUMERICAL CEILING (computed post-hoc): all ~173 per-op syncs x ~185us ~= 32ms = <2% of the 1864ms chunk.
So eliminating worker-completion syncs CANNOT explain or fix a meaningful fraction of the chunk time. The
whole kernel-config/launch-ring byte-wrap angle (E1/E2/E3 + mailbox surgery) is a DEAD END for perf.
ACTIONS: revert E1 (bh_hal_tensix.cpp -> 69KB). Do NOT pursue E3/mailbox surgery. kernel_config_entry_count
=32 stays committed (harmless, removes table-full syncs, PCC-neutral). 
Note: the runner's +1.4s/chunk vs this 1.86s standalone is the SEPARATE H2D-service tax (already root-caused);
the 1.86s standalone itself ~= the "test" speed, so within-chunk dispatch micro-opt has little headroom.
NEXT: per user, "go-signal optimization" — but MEASURE the go-signal per-op contribution FIRST before
optimizing, given the <2% ceiling lesson. If go-signal is also sub-% of the chunk, the only wholesale lever
for per-op dispatch overhead is Metal Trace (kept at user's distance).

## DEFINITIVE ATTRIBUTION (2026-06-14 16:25) — from existing 3L device profile (dev 30, warm, last chunk)
op2op (dispatch gap) per layer:  layer0 dense=9.64ms | layer1 MoE=3.45ms | layer2 MoE=-0.06ms (~0).
=> The op-to-op gaps are a ONE-TIME PIPELINE-FILL transient (front-loaded in layers 0-1); by layer2 the
dispatcher has run ahead and ops are BACK-TO-BACK (op2op ~= 0). The "ridiculous" gaps (RingJointSDPA
2020us, AllGather 1901us, ReduceScatter 707us, RoPE ~500us) are ALL in the first 1-2 layers = pipeline fill.
Across the real 61-layer chunk (1 dense + 60 MoE): op2op ~= 9.6 + 3.4 + ~0*59 ~= 13ms out of 1864ms = <1%.
The chunk is COMPUTE-BOUND: kernel_sum=91ms/3layers; x~20 ~= the full 1864ms chunk.
=> CONCLUSION: per-op dispatch micro-opts (worker-completion sync [E1, falsified], go-signal mcast wait)
target a <1% steady-state cost and CANNOT move the chunk. go-signal optimization is a confirmed dead end
for the standalone chunk (would be neutral, exactly as E1 was). NOT pursuing it.
LEVERS THAT REMAIN for the 1864ms standalone chunk: (a) faster KERNELS (matmul/SDPA/comm ops are the time);
(b) Metal Trace removes only the <1% dispatch overhead -> won't help a compute-bound chunk. 
The user's headline gap (runner 3.3s vs test 1.94s) = the SEPARATE +1.4s H2D-service tax (already root-caused);
the 1.86s standalone ALREADY ~= the 1.94s "test" speed. There is no within-chunk dispatch win to be had.

## 9-LAYER PER-LAYER op2op (2026-06-15) — DEEP LAYERS CONFIRMED gap-free (no-service)
Method: profiled NUM_LAYERS=9 in ONE clean run. Solved the profiler limits: (1) DRAM marker buffer overflow
-> TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=3000 (env, no rebuild); (2) ~1.2GB/layer raw blowup on the
99%-full / -> redirect TT_METAL_PROFILER_DIR=/dev/shm/ttprof (284G tmpfs). No drains (READ_EVERY=0) so data
is unperturbed (op2op == op2op_excl1). Slowest dev 30, warm kv=51200:
  layer0 dense: kernel 17.5ms  op2op 8.53ms  (31.6% of wall)
  layer1..8 MoE: kernel ~37ms each, op2op ~ -0.07ms (~0, perfectly pipelined; boundary first-op ~0.5us)
=> The op2op dispatch gap is ENTIRELY a one-time PIPELINE-FILL in layer 0. Deep layers (incl. 6,7,8) have
~0 gap. Total kernel 314ms / op2op 7.96ms (~all in L0) = 2.5% of the 9-layer wall. Standalone chunk is
COMPUTE-BOUND, confirmed across depth (settles the earlier 3L-only doubt). For 61 layers op2op ~= 8ms (L0)
+ 0*60 = ~0.4% of chunk. Tools: profile_NL.sh (PSC + /dev/shm + ITERS/PROF params), parse_NL.py.

## 9-LAYER WITH-SERVICE (2026-06-15) — the H2D service tax IS per-layer-recurring at all depths
Slowest dev 2, warm kv=51200. op2op per layer: L0 14.4ms, L1 10.5, L2 14.8, L3 21.4, L4 15.8, L5 17.4,
L6 18.0, L7 16.2, L8 16.1 => total 146.9ms op2op vs 302ms kernel (33% of wall!). CONTRAST with no-service
(8ms total, ALL in L0). Per-op (layer4): the gaps land on the small ops BEFORE RingJointSDPA (Matmul 694us,
ReduceScatter 823us, RoPE 1019/1235us, ...); after SDPA (op25+) they collapse to ~0.5us, then re-starve next
layer. MECHANISM: dispatcher run-ahead eliminates gaps (no-svc: fills once at L0 SDPA, stays ahead forever).
The H2D service's per-op dispatch coordination DISRUPTS run-ahead every layer -> pre-SDPA small ops re-starve
each layer -> ~16ms/layer x 61 ~= ~1s = the +1.4s/chunk service tax. Detail logs:
ops_slowest_device_9L_{noservice,service}.log; per-layer: ops_9L_shm_{noservice,service}.perlayer.log.
DELIVERABLE for user: per-op + per-layer op2op for all 9 layers, both modes (the format requested).
