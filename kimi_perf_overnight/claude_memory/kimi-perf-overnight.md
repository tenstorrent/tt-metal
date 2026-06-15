---
name: kimi-perf-overnight
description: Overnight perf investigation harness for the Kimi prefill runner-vs-test 1.4s/chunk gap (tmux + queue)
metadata: 
  node_type: memory
  type: project
  originSessionId: 73e2e6db-60ce-49ca-9a6e-bf0b6caab1ac
---

Investigating why the Kimi K2.6 chunked-prefill RUNNER is ~3.3 s/chunk vs the no-PCC transformer TEST
~1.94 s/chunk (~1.4 s constant additive per-chunk gap). Started 2026-06-13 (evening).

Durable harness lives at **`/home/ppopovic/kimi_perf_overnight/`** (NOT in git, local to bh-glx-d04u02):
- Runs in tmux session **`kimi_perf`** via `orchestrator.sh` (queue-driven, no Claude in the loop).
- `RESULTS.md` = findings; `HYPOTHESES.md` = ranked theories; `SESSION_CONTEXT.md` = resume snapshot;
  `RECOVERY.md` = attach/inspect/stop/add-experiments.
- Instrumentation committed to branch `ppopovic/investigation` (commit d0abf44ed8f): new
  `models/demos/deepseek_v3_d_p/tt/perf_probe.py` + edits to tt_prefill_transformer.py,
  tt_prefill_block.py, mla/mla.py. Env-gated: PREFILL_SECTION_TIMING, PREFILL_DUMP_CONSTRUCTION,
  PREFILL_SKIP_ACK_SYNC. All default OFF, pure Python (no C++ rebuild).

RESOLVED 2026-06-14 ~03:46. **Root cause = the H2D stream service running in the request-loop process.**
Both paths call the same forward_chunk; only the service's PRESENCE matters. Standalone-chunked & the
no-PCC test run ~1878ms/chunk; request-loop ~3090ms. CONFIRMED two ways: (1) by elimination — mla_seq_len
(sweep 56320→102400 flat), ack-sync (PREFILL_SKIP_ACK_SYNC/DISABLE_LAYER_ACK), construction (identical),
and the request-only clear_loaded_sub_device_manager (PREFILL_FORCE_PRECLEAR standalone stayed ~1879)
all REFUTED; (2) directly — PREFILL_FORCE_BUILD_SERVICE built the UNUSED service in the standalone path
and it slowed ~1878→~3177ms/chunk. Fix is RUNNER/tt-metal-side; model is already at test speed. DIAGNOSIS COMPLETE via tracy (NUM_LAYERS=1):
on WARM (post-compile) ops the on-device DEVICE FW time is IDENTICAL with/without the service (~55us
median) but OP-TO-OP LATENCY is ~3x (135us->429us median) => the H2D service imposes a per-op-launch
DISPATCH/host-coordination tax, NOT on-device contention and NOT compile. Two Python fixes FAILED: service
worker-core relocation off the model grid (PREFILL_H2D_WORKER_COL=11, ~3190 vs ~3201ms) and 2 command
queues (PREFILL_NUM_CQ=2, ~3241ms). => Fix requires H2DStreamService (C++): give its resident receiver
program its own dispatch domain/sub-device (excluded from other programs' fast-dispatch launch
coordination) or idle it during forward_chunk. Hand to a tt-metal dispatch engineer with evidence
kimi_perf_overnight/devperf_{noservice,service}.csv. Diagnostic env knobs committed (default-safe):
PREFILL_FORCE_BUILD_SERVICE, PREFILL_H2D_WORKER_COL/ROW, PREFILL_NUM_CQ, PREFILL_STANDALONE_CHUNKED_ITERS.
Commits: 89283177b4d, 284cb228743, cdb2823298a (+repo RUNNER_PERF_INVESTIGATION.md). CAUTION: box disk hit
100% from tracy raw logs (~4.7GB/run on an already-near-full /); cleaned mine; avoid repeated tracy here.

Full writeup: /home/ppopovic/kimi_perf_overnight/RESULTS.md (FINAL SUMMARY) + repo
models/demos/deepseek_v3_d_p/tt/runners/RUNNER_PERF_INVESTIGATION.md (commit c708cb07a4d). Probes on
branch ppopovic/investigation: d0abf44ed8f, 3b63b010766, 0900424df25. Orchestrator stopped; all 11 exps
done. Operational lessons: never kill a runner mid-init (chip-lock/ethernet wedge → needs tt-smi
-glx_reset_auto, but NOT while a co-user has device procs); cleanup must wait for FULL process reap (a
zombie still holds sysmem) and rm stale /dev/shm/tt_prefill_layer_acks_* after SIGKILL; ETH_LIVE_STATUS
0x0 is normal, not a wedge signal. Related: [[kimi-prefill-env-vars]], [[kimi-chunked-prefill-work-state]].

3-LAYER per-op profiles (warm, slowest device, kimi 1 dense + 2 MoE; logs in kimi_perf_overnight/ops_slowest_device_3L*[svc]_{first,last}.log): RingJointSDPA (attention) is the only prefix-scaling op (~2.9ms@5120 -> ~14ms@56320, all layers); MoE FFN prefix-independent (UnifiedRoutedExpertFfn x12 ~8-10ms, Combine ~3-4ms, Tilize ~2ms, Dispatch ~2ms, ReduceScatter ~3ms, gate MoeGroupedTopk ~0.07ms). MoE layers ~2x dense kernel. WITH H2D service: kernel times unchanged, WARM op2op dispatch gaps grow +5..+25ms/LAYER (MoE expert ops worst: ~7us->~2500us op2op) — the per-op-launch dispatch tax confirmed warm + per-layer, the mechanism of the ~1.2s/chunk runner-vs-test gap over 61 layers.

ROOT CAUSE of op2op gaps (pinned 2026-06-14): DISPATCH-PIPELINE-bound, not host-CPU and not device-compute.
Per-op host/device timing (ops_perf_results HOST START/END TS): layer0 host CPU work=1.9ms but host span=16.5ms
(~=device time) => host idle ~88%, enqueues each op in ~1-2us then idles ~370us. Small ops (3-160us kernel)
wait ~100-1900us for command-processing + go-signal multicast across the 8x4 mesh; device idles. A heavy op
(RingJointSDPA ~14ms) lets the prefetcher/dispatcher run ahead so following ops' go-signals overlap -> gaps
collapse to ~0.5us (why gaps appear only before the first heavy op per layer). H2D service adds per-dispatch
sub-device/worker bookkeeping -> ~3x warm gaps. FIX = Metal Trace (begin/end_trace_capture + execute_trace,
ttnn/cpp/ttnn/operations/trace.cpp): replay command buffer via one add_prefetch_exec_buf/device, eliminates
per-op dispatch -> op2op->~0 (caveat: static shapes, 1 trace per logical_n or pad). Secondary: 2nd CQ, check
prefetch_q 256(DRAM-backed) vs 1534, reduce sub-device churn. To pin exact stall: re-profile --profile-dispatch-cores
(DISPATCH GO SEND WAIT TIME).

DISPATCH GAP deep-dive CONCLUSION (2026-06-14): the ~370us/op op2op gap is REAL, not profiler observer-effect
(profiler OFF wall ~24.8ms == ON ~24.5ms for a 1-layer warm chunk; ~9ms/37% is genuine device idle, standalone
no-service). Host CPU idle ~88% (no enable_async toggle; fast dispatch already async) => per-op multi-chip
go-signal+completion dispatch cost on the 8x4 mesh (~10x single-chip ~10-50us). Gaps hit small ops before the
heavy SDPA; collapse after it (dispatcher runs ahead during the 14ms op). NUM_CQ=2 does NOT help (~25ms). No
config knob fixes it. THE FIX = Metal Trace (purpose-built for small-op dispatch starvation; replay = one
add_prefetch_exec_buf/device, collapses per-op dispatch AND the H2D dispatch tax) OR op-fusion OR a tt-metal
mesh go-signal C++ optimization (fd_mesh_command_queue.cpp go-signal path + cq_dispatch.cpp worker-completion
gate). Diagnostic knobs committed (PROFILE_KV + per-pass timing, 7968b89092d). Tools: dispatch_observer.sh.

REFINEMENT/CORRECTION 2026-06-14 ~16:25 (SUPERSEDES the "per-layer recurring gap / Metal Trace fixes the
standalone" framing below): re-read the NO-SERVICE 3L profile (ops_slowest_device_3L_last.log, dev30, warm,
logical_n=56320): op2op gap per layer = layer0 dense 9.64ms, layer1 MoE 3.45ms, layer2 MoE -0.06ms (~0). So
in the STANDALONE path the op2op gaps are a ONE-TIME PIPELINE-FILL transient (front-loaded layers 0-1; by
layer2 the dispatcher runs ahead and ops are back-to-back). Across 61 layers that's ~13ms/1864ms = <1%. The
standalone chunk (1.86s = the "test" speed) is COMPUTE-BOUND (kernel ~91ms/3layers). => within-chunk per-op
dispatch micro-opts are DEAD ENDS on the standalone path: E1 (kernel-config ring 69->133KB) FALSIFIED —
perf-neutral 1864 vs 1878ms/chunk on real 61-layer, PCC 0.965 (idx0 byte-wraps 111->0 but launch-ring idx3
became binding; net syncs are <2% of the chunk by arithmetic). go-signal mcast micro-opt would likewise be
neutral on standalone. The ONLY meaningful dispatch lever is the SEPARATE +1.4s H2D-SERVICE per-op tax (the
WITH-service profile's +5..25ms/layer op2op): the resident receiver program taxes EVERY model op's go-signal
mcast + worker-completion coordination. Real fix = isolate the service into its own dispatch domain/sub-device
(a go-signal/completion-domain C++ change, runner path) — NOT optimizing the mcast primitive, NOT E1/ring, NOT
Metal Trace (Trace only removes the <1% standalone dispatch). Also pinned a self-inflicted infra trap, see
[[firmware-host-mailbox-mismatch]] (unreverted dev_msgs.h launch=16 vs launch=8 host lib -> profiler
timeout=416 + 30-min embedding hang, survives GLX reset).

OPTION B (device dispatch optimization) CONCLUDED 2026-06-14: per-op ~370us gap pinned via env-gated probe
(TT_DISPATCH_SYNC_DEBUG in worker_config_buffer.cpp reserve()): dispatcher forced to sync on worker-completion
~every op because the ~69KB TENSIX kernel-config L1 ring byte-wraps (245 syncs/120ops: 111 ring byte-wrap + 72
table-full + 62 launch-msg-ring). FIX TRIED: kernel_config_entry_count 8->32 (committed a8239c53ea2) — removes
72 table-full syncs, PCC 0.965>=0.88 (correct), but PERF-NEUTRAL (~1863 vs 1878ms/chunk) since the ring
byte-wrap stalls ~every op regardless. Host-fan-out parallelization (earlier) REGRESSED (reverted). 2nd CQ:
no help. Launch-ring device bump: non-binding while ring byte-wrap dominates (not attempted). REAL FIX (human
decision, not done autonomously): enlarge the kernel-config L1 ring (reduce model per-op L1 footprint or HAL
KERNEL_CONFIG/worker_l1 layout — deep, risky) OR Metal Trace (eliminates per-op dispatch; recommended). Whole
gap = two per-op dispatch overheads (H2D service tax + kernel-config-ring stalls), both killed by Trace.
