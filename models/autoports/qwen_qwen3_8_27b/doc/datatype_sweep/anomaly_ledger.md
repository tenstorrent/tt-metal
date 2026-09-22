# Stage 8 anomaly ledger

## Interrupted BFP4 KV startup

- Observed anomaly: device 1 realtime-profiler synchronization timed out before
  layer 0; loading later stalled after layer 63, before `MODEL_LOADED`.
- Evidence: `kv_bfp4_evaluated.log`, `triage_kv_startup/host_snapshot.json`,
  `host_gdb.log`, `host_gdb_main.log`, `recovery_actions.json`.
- Affected path: mesh health and blocking pinned weight/embedding/rope upload,
  before any full-model KV allocation or measured decode.
- Control: same-policy `kv_bfp4_recovered.json` passes full64 accuracy and traces.
- Likely subsystem: native device/completion-queue infrastructure. Exact root
  cause is unproven; host input bfloat16 is not proof of device tensor dtype.
- Investigation: bounded live triage timed out empty; host GDB captured the main
  blocking upload and CQ workers. Only the owned model process was terminated.
  Bounded list failed, one reset passed, all four p300c devices reappeared and the
  exact Ring1x4 mesh opened/closed. Fresh subagents inspected source and recovery.
- Resolution: controlled recovery, supported by `AUTOTRIAGE_kv_startup.md` and
  `AUTOFIX_kv_startup.md`. No numerical result was assigned to the interrupted
  attempt. The retry retained the same runtime/native sources and selected dtype;
  added timeout diagnostics are disclosed as a possible timing confound.

## Initial launcher failure

- Observed anomaly: first `head_bfp8_lofi` model report completed, but its shell
  launcher returned 2 after being edited while running.
- Evidence: `head_bfp8_lofi.{json,log,exit_status}`, `work_log.md`.
- Affected path: Bash file read offset and experiment bookkeeping.
- Control: `head_bfp8_lofi_evaluated` clean exit 0, full64 accuracy/trace results.
- Likely subsystem: experiment launcher, not TT model execution.
- Investigation: model JSON completed; shell error followed launcher mutation.
  Launcher changes were completed before subsequent jobs were launched.
- Resolution: fixed run procedure and clean rerun; initial attempt excluded from
  ranking and retained in `failed_attempts.json`.

## Active-trace allocation warning

- Observed anomaly: generic allocator warning says buffers allocated while a
  trace exists may be corrupted on replay.
- Evidence: full-model logs; `check_prefill_tracing.py` capture/lifetime guards.
- Affected path: prefill/decode/sampling persistent and temporary device buffers.
- Control: selected full64 non-aligned/output-lifetime check and separate selected
  real-layer 0/3 trace-allocation-tracked watcher check.
- Likely subsystem: trace allocator lifetime rules.
- Investigation: selected_non_aligned exit0 covers all seven required full64
  lengths, 50 guarded captures, retained outputs and physical-page remapping.
  selected_watcher exit0 uses allocation tracking plus watcher on real layers0/3,
  Ethernet checks enabled and no device profiler; 21 guarded captures pass.
- Resolution: controlled by successful selected-policy retained-output and
  separately instrumented lifetime/synchronization tests.

## L1 semaphore placement warning

- Observed anomaly: multicast gather warns that semaphores use ordinary L1,
  potentially fragmenting it when L1_SMALL has not been reserved.
- Evidence: `all_gather_multicast_factory.cpp:34-48`, candidate and final logs.
- Affected path: inherited CCL semaphore allocation and available L1 headroom.
- Control: identical mesh/allocator settings in refreshed baseline and all 14
  passing full-model candidates; no L1 allocation failure in those runs.
- Likely subsystem: allocator placement, not a numerical corruption report.
- Investigation: source explicitly selects normal L1 when L1_SMALL size is zero.
  Selected head precision is smaller than the baseline; geometry is unchanged.
  Final batch/context and separate watcher checks additionally exercise the
  selected allocation and synchronization behavior.
- Resolution: controlled in measured candidate and final full-context/B32
  workloads, plus separate watcher. No claim of unlimited L1 headroom is made.

## Motherboard discovery fallback

- Observed anomaly: unknown `B850M-C` uses PCI bus ID as tray ID.
- Evidence: `tt_metal/fabric/physical_system_discovery.cpp:103-122`, startup and
  post-recovery device list/mesh logs, all completed candidate logs.
- Affected path: physical discovery tray metadata for this host motherboard.
- Control: all four expected p300c devices are listed and the exact TP4 Ring
  opens, executes full64 collectives and closes successfully.
- Likely subsystem: motherboard-name lookup, with an explicit source fallback.
- Investigation: source returns the discovered bus ID as tray ID when the board
  name lacks a table entry; it does not remove a chip or change weight precision.
- Resolution: controlled by actual mesh/list and full-model communication results.

## Recovery owner-snapshot label correction

- Observed anomaly: recovery_actions.json field `owners_after_termination` lists
  device 1 PID 602116, while the work log claimed empty owners after termination.
- Evidence: `triage_kv_startup/ownership_timeline_correction.json` preserves the
  narrow original tool-call/output sequence, including the empty 22:58:20 snapshot.
- Affected path: recovery evidence labeling and timing, not a new device action.
- Control: terminated model exit 143; all four owner files printed empty at
  22:58:20; list started at 22:58:26 and completed at 22:59:00; reset started at
  22:59:14, before the JSON owner snapshot was taken at 22:59:15.
- Likely subsystem: the report captured owner state during reset and labeled it
  as the earlier post-termination state. PID-to-command mapping was not saved;
  attributing PID 602116 to reset is a timing inference, not direct evidence.
- Investigation: inspected original coordinator tool transcript and preserved
  only relevant recovery calls/outputs; raw recovery_actions.json left intact.
- Resolution: fixed documentation, with actual snapshot timing and uncertainty.

## HF greedy token versus readiness top-1 ordering

- Observed anomaly: selected native outputs match 98/100 HF greedy feedback
  tokens, while the standard readiness top-1 result is 99/100.
- Evidence: `reference_topk_order_audit.json`, the pinned refpt,
  `tests/hf_reference.py:80-81`, and `models/common/readiness_check/teacher_forcing.py`.
- Affected path: reference scoring convention, not native sampled-token delivery.
- Control: the reference differs only at position 15: `scores.argmax()` saved
  token 13, while `scores.topk(100)` saved token 15060 first and token 13 second.
  These operations can order tied maxima differently.
- Likely subsystem: HF top-k tie ordering versus lowest-index greedy argmax.
- Investigation: a host audit recomputed top-1/top-5/top-100 from every available
  candidate native-token array against the actual stored top-k reference. Every
  result exactly matches the standard readiness report. Selected confirmation
  matches top-k first entry at position 15; its sole readiness miss is position
  17, where its prediction is second in the HF top-k list.
- Resolution: controlled reference convention. Metrics and thresholds are
  unchanged; no sampler/runtime divergence was found by this comparison.

## Haiku meter under selected head quantization

- Observed anomaly: selected prompt0 final lines scan5/7/6, versus conventional
  5/7/5 in the exact HF and previous-stage controls. The prompt asks for a haiku
  without an explicit syllable-count constraint; the six-syllable line is retained
  as a specific quality limitation, not declared conventionally correct.
- Evidence: AUTODEBUG_haiku.md, AUTOFIX_haiku.md, haiku_selected_retry.json,
  haiku_baseline.json, haiku_head_hifi2.json and their native/host output artifacts.
- Affected path: BFP4 LM-head quantization and greedy creative-output trajectory.
- Controls: S60/G1024, cache1088, physical page table0..33, history1023 and seed0
  fixed. Selected native/native/host-greedy are identical347-token streams.
  Baseline BFP8/HiFi2 native repeats match the Stage7 418-token 5/7/5 stream.
  BFP4/HiFi2 native repeats match the selected347-token 5/7/6 stream.
- Likely subsystem: precision-sensitive model generation; no source-proven
  runtime bug. First branch divergence is token31, absolute position90/page2
  offset26, away from cache/page/chunk boundaries.
- Investigation: fixed-shape precision A/B, repeated trace replay, actual device
  page-table checks, and host CPU argmax control. Final norm remains HiFi2.
  The initial probe-only optional-host-mirror dereference was corrected; original
  attempt is preserved in failed_attempts.json and all final controls exit0.
- Resolution: controlled quality limitation, not repaired. A fidelity-only fix is
  refuted. The user-specified full-model accuracy thresholds and fastest traced
  performance remain the selection rule; no universal instruction-quality claim
  is made for the six-prompt suite.

## Noncanonical layer-exception keys

- Observed anomaly: integer0 or string00 keys could pass validation but fail
  canonical string lookup, silently leaving a requested exception unapplied.
- Evidence: reviewer host reproduction; precision.py guard and
  check_datatype_evidence.py regression cases; policy_validation_equivalence.json.
- Affected path: unmeasured malformed override inputs, before model allocation.
- Control: canonical string0 applies the expected dtype; all15 measured configs
  and the selected artifact resolve identically under old and new validation.
- Investigation: host-only old/new policy resolution and explicit rejection tests.
- Resolution: fixed validation. Runtime numerical policies and measured kernels
  are unchanged for every accepted measured config.
