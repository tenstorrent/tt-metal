# Register-resident GDN experiment and compact qualification snapshot

Captured October 10, 2026 at 20:46 UTC. These receipts record completed tests,
failed simulator executions, and a waiting physical experiment separately.
No model policy selects the new register-resident compute path.

## Current measured model

The completed before/compact/after comparison gives B16/32K/TP4
16.583725 / **20.034955** / 16.584061 tokens/s/user. The compact step is
49.912766 ms, a 20.81% throughput improvement. At 16K the three arms give
18.405926 / **22.750585** / 18.406055 tokens/s/user. All output hashes agree;
control drift is 0.00202% at 32K and 0.000702% at 16K. Raw arms and the
recomputed comparison are retained under `compact-gdn-v3/`.

The eight-replica G0 test and serving API checks passed. GPQA began at
20:19:34 UTC with all 198 questions and a 65,536-token output budget. The
captured progress is **159 correct of 167 completed, zero truncations**;
31 questions remain. This is a partial score, biased toward earlier finishing
answers, not accuracy qualification. Protocol, source hashes, G0/API receipts,
and the partial evaluator log are retained. Dataset questions and private raw
responses are not published here. Eight times TP4 throughput is a projection,
not a measured whole-Galaxy client rate.

## Register-resident candidate

`tt/gdn_step/compute_resident.cpp` retains four FP32 state tiles in eight-tile
full-DEST mode through decay, delta update and the output reduction. It removes
eight state-tile reloads, four repeated decay multiplies, and the delta
pack/unpack per value-column work item. Reader/writer ownership, precision,
state traffic and the ordered multiply/add reduction remain unchanged.

`resident_state=True` is allowed only for four value partitions and previously
normalized Q/K; the default is false. The benefit target is **0.5-1.5 ms per
full-model step**, approximately 1-3% from the measured compact baseline.
This is unmeasured. Full-DEST synchronization and later state writeback may
offset the saved operations by reducing math/pack and writer overlap.

Both simulator attempts compiled the executed kernel, then failed with:

```text
UnsupportedFunctionality: tensix_setdvalid: interaction between SETDVALID
and implied src format is ill-specified (use UNPACR_NOP instead)
```

The first attempt executed the existing control first; the second executed
the new candidate first. Neither produced an arithmetic comparison. Both
systemd units exited 1. The hard simulator exit left each probe JSON at
`running`; `units.json` and the raw compressed logs establish the failed
terminal state. Simulator compilation with SFPLOADMACRO disabled is not
physical-hardware validation or a performance result.

## Persistent physical test

`qwen38-gdn-resident-hardware-v1-20261010.service` was live, waiting with
`hardware_started=false`, at capture. It follows the exact existing compact
long-horizon invocation, after GPQA/profile, projection tuning and prefill
attention. It does not reorder or interrupt those jobs.

- Frozen preflight: **579 passed, 1 skipped, 77 subtests passed**. Physical test
  collected, launcher shell syntax passed. Unit checks do not establish device
  correctness.
- Physical cases: four independent rank inputs; B16/B32 for 64 changing-input
  updates; B1 for 4,096 updates; simultaneous allocations and rebinding;
  cancellation-sensitive first steps; all-rank immutable inputs.
- Accuracy requires bit-identical control/candidate output and recurrent state,
  plus independent dense-reference PCC at least 0.999 and per-head relative RMS
  error at most 0.005.
- B16/B32 before/candidate/after timing: five samples of 100 trace replays,
  identical state reset after capture, eight warmups, and identical final
  state/output hashes after 508 updates. Control drift must be at most 3%.
- Physical timeout 1,800 seconds; test timeout 1,620 seconds; 16 GiB host RAM,
  eight CPU equivalents, and the shared device lock. The waiting controller
  has a 28-hour lifetime and survives client disconnect, not host reboot.
- Clean teardown, strict report coverage and frozen source verification are
  required. Passing this test does not promote the candidate into the model.

The live source is frozen under
`/home/ttuser/qwen38-artifacts-20261007/gdn-resident-hardware-source-v1`.
Its eight candidate/harness files match this publication. The inherited
convolution-reader formatting discrepancy is preserved in the source manifest;
the earlier compact source-provenance receipt documents that whitespace-only
difference. Native installs, firmware, NFS and the active GPQA source are
unchanged.

`capture.json` lists remote paths and hashes of the uncompressed receipts.
Logs and XML are gzip-compressed without changing their content. Systemd
`Result=success` on a still-running unit is not a completed-test result.
