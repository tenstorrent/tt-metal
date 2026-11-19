# Round 2 diff-only review brief

Stage 4 multichip decoder, Qwen/Qwen3.6-35B-A3B. Same live main worktree,
starting HEAD 2901d8a506198dc69a9e99714db3a9c590b6ff95; no commits yet.
Original goal and skills: `review_contract.md`. No full-model or vLLM work;
no pushes. This is an independent follow-up, not a whole-stage re-review.

## Previous findings (verbatim)

## Required Work

- **P2: Supply the required trace-allocation positive control.**
  Evidence: the selected skill's `tt-enable-tracing/SKILL.md:205,262`
  explicitly requires a deliberately live unsafe device buffer to make replay
  raise, in the same environment, with output saved at
  `doc/<stage>/trace_alloc_tracking/positive_control.log`. That file and directory
  are absent here. A search including ignored files under the model's `doc/`
  found no positive-control artifact, and the stage manifest/work log contain
  no such result. `natural_l{0,3}_b32`, both stress runs and the mixed-stack
  provenance do correctly record tracking=1 and skip-program-cache=0, but
  `tests/summarize_multichip_validation.py:120-124` verifies only those environment
  settings; its 26 gates never verify a rejection control. The runner acknowledges
  exact final outputs at `tests/run_multichip_decoder.py:263-268` and then runs
  clean replays, without attempting the required negative case.
  Why this matters: clean tracked execution is silent. The required control
  distinguishes an active allocation safety gate from settings recorded in
  provenance without an effective checker. This is a missing required gate
  artifact, not evidence that the decoder is numerically wrong.
  Required next step: run one small, bounded control on the same TTNN build and
  target mesh, with `TT_METAL_TRACE_ALLOC_TRACKING=1` and
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` set before importing TTNN. Capture a
  prepared trace, retain a non-acknowledged device allocation made after capture,
  and assert that replay raises the expected unsafe-allocation RuntimeError.
  Save the command, environment and observed exception at the prescribed path,
  link it from the evidence manifest, and make the acceptance check require the
  passing control. A clean replay after releasing the injected allocation is a
  useful recovery check. The existing decoder correctness/timing/profile runs do
  not need to be repeated merely to add this control.

## Other Concerns

- The constructor census quoted in `mesh_plan.md:139-140` is stale:
  465,528,832 / 450,160,640 bytes versus the selected mixed-stack artifact and
  capacity plan's 465,569,792 / 450,185,216 bytes. The actual capacity calculation
  uses the latter and still has 4,135,717,888 bytes of planned headroom. Correct
  the prose; this small discrepancy does not invalidate the capacity result.
- Stage 5 has concrete performance work to start from: per-request attention at
  B32, approximately 200 us of output splitting, and unmeasured B32 shared/grouped
  geometry. The mesh's multirow projections currently use automatic programs,
  whereas the optimized baseline explicitly tuned prefill programs; adapting
  those to the local shapes is a useful candidate. The supplied profiles identify
  these costs and do not claim those unmeasured families were rejected. These are
  performance follow-ups, not additional correctness blockers for this review.

## Hard-Check Gaps

- The missing positive control is the required gap above. The 26/26 result
  otherwise agrees with the saved artifacts; every listed evidence path exists.
- The gate compares current decoder identity but does not require the measured
  runner's identity to be current. I investigated the mismatch rather than
  treating it as stale evidence: removing only the optional `prefill_states`
  export blocks and its CLI flag from the current paired runner reproduces the
  exact recorded hash `8e1e0437b146aadce7aa518dd0f820e39bef16aed70f8d8296d4b561cd084f7e`.
  That option is disabled in the earlier acceptance runs, so their execution
  remains applicable. No rerun is needed for this historical runner change.
- The profiler summary and checker share summarization helpers. Independently
  streaming the CSVs in this review verified all selected device-row timings
  against the original C++ timing CSV and recomputed each rank's mean; they
  matched. This is a check-design observation, not a request for another gate.


## Changed since round 1

- Added `tests/run_multichip_trace_control.py`: bounded 1x4 control, tracking=1,
  skip-program-cache=0 before TTNN import, exact capture-output acknowledgement,
  unacknowledged live post-capture buffer forces RuntimeError, release then exact
  recovery output on all four devices.
- `tests/summarize_multichip_validation.py`: new mandatory positive-control gate
  (27 total), current runner source identity, environment, actual buffer-specific
  exception in log, recovery and four output replicas.
- `trace_alloc_tracking/positive_control.{json,log}` and JSON provenance sidecar.
- `gate_summary.json`: full updated CPU acceptance check.
- `mesh_plan.md`: corrected census to465569792/450185216 bytes; unchanged capacity.
- `README.md`, `work_log.md`, `artifact_manifest.md`, `AUTOFIX.md`: control commands,
  evidence, review status and stage5 performance follow-ups.
- `failures/trace_control_watcher_zero/`: initial control-launch evidence.
  Watcher=0 enables zero-interval polling; initial open timed out at180 seconds.
  Live stack/triage captured before timeout. Reset12 + trivial-op smoke passed;
  corrected launch unsets watcher and completes. New runner rejects present
  watcher before TTNN import. Logs `logs/reset_12{,_smoke}.log`.
- `stage_review_round1.md`: the prior report.
- Decoder source unchanged: SHA256
  97792b176b8fdc2f41d0b1c9c6ab7e98f08dd7bb94d87fac08b008c3e27fcaa3.

## New evidence commands

After sourcing python_env/bin/activate and ../lane-env.sh:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 timeout 180 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_trace_control \
 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/trace_alloc_tracking/positive_control.json
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_validation
```

Control exit0, expected rejection buffer30, acknowledged output23, recovered
exact four outputs. Gate result/log `gate_summary.json`, `logs/final_gate_review2.log`.
Formatting/precommit and diff checks performed before review. Existing numerical,
latency and profiler data unchanged. Only inspect these changes and close/open
each previous finding. No TT devices or long tests; report read-only except
writing `stage_review_round2.md`. See stage-review skill for required verdict.
