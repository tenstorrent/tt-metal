# Stage Review

Verdict: more-work-needed

Inspection-only follow-up for the Stage 10 batched recurrent-snapshot candidate,
2026-10-25 13:42 UTC. This is an interim candidate review, not final stage
acceptance. No concrete runtime defect was found in the inspected batching,
kernel ABI, ownership or warmup path. The required remaining work is final
integration and performance evidence on the expanded source. This report does
not replace the fresh reviewer required for final closure.

## Required Work

- P1: Complete acceptance on the expanded source and keep superseded runs separate.
  Evidence: the primitive and watcher probes pass, but the final B16 performance
  result is not yet present. `final_b16_command.json` contains the new kernel and
  matching current source hashes. Earlier `after_*` results cover terminal-only
  suppression. `superseded_b1_status.json` correctly excludes the run whose
  source changed while its process was live.
  Why this matters: a small DMA probe does not establish model-state correctness,
  prefix-cache behavior or trace-allocation safety across the complete serving
  workload. Kernel-count reduction does not establish a latency improvement.
  Required next step: finish the already planned comparable B16 measurement;
  retain the candidate only with a justified measured result. If retained, finish
  the planned changed-source B32 mixed/agentic lifecycle with tracking enabled
  and program-cache allocations included, B8 full sampling, and B1
  headline/qualitative runs. Keep final performance diagnostics disabled and
  preserve source manifests. Compare outputs/metrics with the appropriate
  terminal-only or prior-stage controls. All other closure gates in the initial
  review still apply.

- P2: Finalize the new persistent-metadata accounting and its evidence pointers.
  Evidence: `doc/context_contract.json` correctly records the payload delta
  and labels allocator checkpoints pending, but its Stage 10 evidence field
  still points to the terminal-only `after_b32_lifecycle.json`.
  Why this matters: the new metadata allocation is small but persistent, and
  old lifecycle evidence cannot validate it against the final live traces.
  Required next step: reconcile the final preparation/allocation checkpoints and
  update the context/agentic ledger and evidence references to the final run.
  Preserve the distinction between metadata payload and measured allocator use;
  do not double-count it inside the recurrent snapshot payload.

## Other Concerns

- The existing pool class assumes that all selected linear layers have the same
  batch size. This is true for the actual model construction. The new batching
  also assumes scheduler-provided transfer keys are unique valid integer slots,
  as the adapter constructs them. No broader arbitrary-mapping API claim is
  established by this review.
- The pinned plugin always calls `warmup_model_prefill` before decode warmup
  (`model_runner.py:4543`), and that adapter callback prepares snapshots before
  `Generator.prepare_decode`. This order is required: the independent adapter
  decode-warmup callback does not itself warm the snapshot pool. Current serving
  satisfies the order; this is not evidence that arbitrary external callback
  orders are supported.
- The byte-copy kernel changes dispatch count, not total snapshot payload.
  For 48 linear layers, each nonempty save/restore group now launches 96 copy
  programs and one metadata upload instead of 96 programs per row. The kernel
  still copies every selected convolution/recurrent row. B16/B32 metadata and
  host dispatch savings must not be described as reduced cache capacity or
  reduced snapshot bytes.

## Hard-Check Gaps

- The new primitive probe verifies restored active rows and all untouched
  active peers across all four replicas. It does not directly inspect the
  entire pool to assert that unselected/older snapshots remain unchanged after
  batched saves, nor does it exercise one shared snapshot restored to several
  live slots in one call. The source has correct bounded destination indexing
  and never writes the pool on restore, so this absence alone is not a runtime
  defect or a demand for another broad hardware sweep. If extending the small
  probe, these are the two focused ownership cases to add.
- The 13 host tests cover existing boundary ownership and dormant rows, but
  the snapshot metadata host fixture has only one active boundary saver. They
  do not alone prove a multi-row adapter batch. The planned full-model B16
  prefill and B32 mixed-prefix run provide the relevant integration coverage;
  a short host-only multi-row mapping test would make that contract easier to
  inspect if needed.
- The primitive trace is a copy into preallocated scratch, not a full-model
  decode/sampling trace. Its tracking result establishes that varying the row
  count does not introduce unsafe persistent allocations or new program-cache
  entries in that process. Full-model live-trace interaction remains the role
  of the final tracking-enabled serving run.

## Anomaly Ledger

- Observed anomaly: terminal-only B16 serving TTFT exceeds the full-model control.
  Evidence: `after_b16_vllm_result.json`: 16 completed, 0 failed, 65536 input and
  2048 output tokens, median TTFT 14400.397095 ms, mean TPOT 73.873524 ms.
  Control or comparison: selected-policy autoregressive B16 control TTFT 7305.29 ms.
  Affected path: 4096-token prefill boundary snapshots across 16 active rows.
  Likely subsystem: repeated snapshot dispatch; this remains a hypothesis until
  the comparable serving measurement completes.
  Investigation performed: verified the raw result and the old per-row versus
  new per-group dispatch structure. No profiler or adapter timing was run.
  Resolution: more-work-needed; final B16 measurement is pending.

- Observed anomaly: source/kernel edits overlapped the terminal-only B1 process.
  Evidence: `superseded_b1_status.json` and work log.
  Affected path: provenance of the in-flight terminal-only run.
  Control or comparison: final B16 manifest includes the distinct new kernel
  filename and matches current source; old kernel is byte-for-byte unchanged
  from HEAD.
  Likely subsystem: artifact provenance and JIT source lifetime.
  Investigation performed: compared old kernel bytes with `git show HEAD`,
  hashed current runtime files and inspected the new manifest.
  Resolution: controlled by exclusion; final B1 must be rerun as already planned.

- Observed anomaly: watcher coverage excludes Ethernet.
  Evidence: `snapshot_batch_watcher.log` explicitly reports disabled feature ETH.
  Affected path: watcher instrumentation, not the worker copy kernel.
  Control or comparison: both primitive JSON files pass the same cases; the
  watcher log closes the mesh normally without a worker assertion.
  Likely subsystem: inherited ACTIVE_ETH watcher capacity limit.
  Investigation performed: inspected the log and exact-output artifact.
  Resolution: controlled scoped coverage; do not claim Ethernet watcher coverage.

## Scope Inspected

- Goal/skill paths: same Stage 10 contract and installed stage-review,
  vllm-integration, optimize, agentic-serving, tt-enable-tracing and
  tt-device-usage requirements as the initial review; this follow-up targets
  the new snapshot candidate only.
- Code: `tt/serving_state.py`, new
  `tt/kernels/serving_state_copy_batched.cpp`, unchanged original copy kernel,
  adapter batched saves/restores, generator preparation/replay context;
  GenericOp hashing/factory and framework descriptor runtime-argument patching;
  pinned plugin warmup order. Inspected the primitive probe and relevant
  adapter host tests.
- Artifacts: `snapshot_batch_contract.json/.log`,
  `snapshot_batch_watcher.json/.log`, `final_host_contract.log`, work log,
  `superseded_b1_status.json`, context-contract diff, B16 baseline raw result,
  final B16 command manifest.
- Commands: read-only git diff/show, rg, cat, sed, nl, tail, and small
  standard-library Python hash/JSON/arithmetic checks. No imports of TTNN,
  model execution, tests, devices, server launches, hardware tools or profiling.
  The only write is this requested report.

Verified implementation properties:

- Metadata is allocated before warmup/capture at `[linear_layers * B, 8]`
  UINT32 entries. Each 32-byte row contains source/destination indices;
  `layer_row = layer_ordinal * B` and runtime `count_rows` select only the
  initialized prefix for that layer. Empty transfers return before upload or
  dispatch; all transfers validate before any device mutation.
- For convolution, each row comprises three 5120-byte pages. For recurrence,
  each row comprises 192 exact 4096-byte FP32 tiles. Three/eight cores,
  respectively, own disjoint page ordinals. Each read and write completes
  before its staging buffer is reused. Distinct live and pool allocations
  preclude source/destination alias cycles; restore writes only live state.
- Per-core runtime arguments include all addresses, metadata base row, page
  ordinal and row count. The new kernel filename isolates the changed ABI.
  GenericOp hashing excludes runtime values; its factory returns the current
  descriptor, and the framework applies current runtime args on cache hits.
  Thus row count is not frozen by the single-row warmup.
- Snapshot warmup compiles both directions before model captures and preserves
  the live row via a save/restore pair. No new allocation acknowledgment,
  corruptible scope, trace invalidation or request-time recapture was added.
- Adapter prefill uses input row coordinates to obtain layer block IDs and
  fixed scheduler slots for active-state coordinates, exactly as before.
  Saves occur after all selected rows have completed the same piece. Decode
  gathers only active rows that reached a block boundary. Snapshot batching
  does not change decode-state advancement, sampling or asynchronous output
  ownership.
- Probe sequence is 1, 3, 32, then 2 rows, including reversed order and sparse
  edge slots. All four replicas match expected active state; the small-after-max
  case detects stale extra metadata consumption. Empty calls and invalid slot 32
  leave counters unchanged. Six program-cache entries remain throughout; one
  original trace is retained; tracking includes program-cache allocations.
- Full-model metadata payload delta is `48 * (B - 1) * 32` bytes/device:
  B1=0, B8=10752, B16=23040, B32=47616. No KV/recurrent-state capacity reduction.

Source SHA256 at inspection:

- `tt/serving_state.py`:
  `c1339583bd8de5e841c73842f42e57098badb31bb35ee542aa7f4cb1636545b1`
- `tt/kernels/serving_state_copy_batched.cpp`:
  `55841378c887c03d5ccb8b39a3bdc51abcf48be0cd1d38a6ef12e26c91ab92c9`
- `tt/generator_vllm.py`:
  `b7f6950e016f4d6fd3b8a9a2295e0a024b733dea34b069ee9ad8944519806a3d`
- `tt/generator.py`:
  `91c8ad90268bcbb42185e9c3ad4cb143d798917316c93ba7b464410812ccacbb`
- `tests/probe_batched_snapshots.py`:
  `493b19a1a39418b7df031d800a8037897b8dfe075de1c06333608546f89fd8c9`

## Residual Risk

- This is source inspection plus existing focused evidence. Final full-model
  serving behavior, end-to-end speedup and final cleanup are not established
  by this candidate review.
- The live worktree may change after these hashes. Earlier terminal-only
  acceptance cannot be relabeled as validation of the new kernel.
- No additional broad sweep, serving profiler or long soak is requested.
  Complete the planned final-source checks and obtain the fresh final review.
