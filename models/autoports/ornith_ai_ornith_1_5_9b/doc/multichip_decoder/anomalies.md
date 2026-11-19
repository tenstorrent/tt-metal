# Anomaly ledger

## Raw mesh binding
Observed anomaly: initial harness passed physical_device_ids=None.
Evidence: tp4_smoke.log.
Affected path: fixture before mesh open.
Control: mesh_smoke.log opened the same topology successfully.
Investigation: binding signature requires a sequence; omitted default is [].
Resolution: fixed to []; tp4_smoke2 and both tracked contracts passed.

## Fused all-gather geometry
Observed anomaly: sender axis8 did not equal one link times four workers.
Evidence: initial perf_gather_0.log (see work_log; later successful rerun supersedes raw log).
Affected path: candidate fused AG+MM setup.
Control: unfused distributed norm + AG + local matmul passed.
Investigation: adapted workers-per-link to8, retaining local packing and sharded residual.
Resolution: fixed; perf_gather_{0,3}.json pass PCC but are slower than controls.

## Fused scatter API
Observed anomaly: wrong compute descriptor type on new fused operation.
Evidence: initial scatter binding failure recorded in work_log.
Affected path: candidate matmul_reduce_scatter_prefill invocation.
Control: separate local matmul+RS passed.
Investigation: read current operation binding and validation. ComputeConfigDescriptor
and negative scatter_dim=-1 express the same fidelity and hidden scatter contract.
Resolution: fixed, padded logical M internally, both fused scatter and combined AG/RS
families measured successfully. Full valid-length gates must cover selected path.

## Geometry stall
Observed anomaly: isolated local down-projection failed to finish.
Evidence: geometry_0.json/log, triage_geometry/tt-triage.txt, AUTOTRIAGE_INPUT.md.
Affected path: experimental 64-core activation shard with K3072, one DRAM reader.
Control: exact-coverage32-core input shard and earlier reader configurations passed.
Investigation: isolated AutoTriage proved a truncated sender-count calculation;
current factory/kernel source and live readiness semaphores agree. Initial parent
attribution to32-core/eight-reader was corrected from completed JSON records.
Resolution: model-local exact-K input planner, independently sized output layout.
Repair probe passes48-input/64-output cores, readers1, blocks1/2, all-chip PCC
above0.9999999 and warmed trace. See AUTOFIX.md; both resumed 156-case sweeps passed.

## Computed matmul output grid
Observed anomaly: TTNN warns that its compact computed grid differs from the
requested rectangular L1 output grid.
Evidence: all baseline and candidate logs; inherited optimized baseline README.
Affected path: DRAM-sharded matmul output memory config.
Control: baseline and multichip output/state PCC and device-only guards pass.
Investigation: operation chooses its worker grid from readers/bank and output width.
The layer consumes the returned actual layout and converts before CCL/head boundaries.
Resolution: controlled correctness; performance impact is included in whole-layer
measurements and final profiler rows; actual weight dtype/fidelity checks pass.

## Motherboard discovery warning
Observed anomaly: B850M-C has no explicit motherboard mapping; UMD uses bus IDs.
Evidence: mesh_smoke.log and all mesh logs.
Affected path: topology discovery.
Control: expected four devices open, CCL output PCC and per-device cache ownership pass.
Investigation: no missing chip or fabric initialization error; repeated mesh open/close
and serialized real CCL work successful. No change to runtime discovery.
Resolution: controlled for this measured machine; hardware identity recorded.

## Final fixture metadata error
Observed anomaly: final_contracts_0 failed before layer construction with an
UnboundLocalError. Evidence: final_contracts_0_harness_error.log.
Cause: a test metadata field was accidentally inserted into constructor kwargs.
Resolution: removed that insertion; host AST check confirms no premature layer
reference. Devices closed cleanly; prescribed reset/list/smoke completed before
retry. Decoder source and numerical implementation were unchanged.

## Resident fused-prefill watcher assertion
Observed anomaly: BRISC NKFW read-counter assertion at logical(1,1), during the
first padded1024-row fused projection. Evidence: watcher_triage/original_console.log
and original_run/. Prepared triage after abort retained health but no usable live
operation counters; that limitation is explicit in the independent report.
Diagnosis: NCRISC issued W reads on NoC1 under dedicated ownership while BRISC
compared hardware responses to its own unchanged private counter. TRID completion
barriers do not reconcile another RISC's counters. See watcher_triage/AUTOTRIAGE.md.
Resolution: host planner enumeration and64/96-KiB constrained fused candidates;
96-KiB focused watcher passed. Both constraints are slower than separate operations.
Default uses measured minimal MM+RS+AG, with native L1 decode layout. The optional
fusion has the source-proven64-KiB constraint. Both final two-link watcher suites passed; all logical lengths and refreshed
request-state replay are covered. No shared code or watcher
suppression was used. Original resident fusion is not claimed repaired.
