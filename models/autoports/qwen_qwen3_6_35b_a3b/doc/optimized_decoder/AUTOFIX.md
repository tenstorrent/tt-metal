# AutoFix: continuation gate and diagnostic evidence

## Contract and diagnosis

The owner requires PCC>=.995 whole-output and explicitly makes per-row PCC
diagnostic. The copied stress harness instead asserted every row and every
row's residual-subtracted contribution>=.995. It stopped on layer34 synthetic
continuation position276 even though aggregate decode/continuation both pass.

`AUTODEBUG.md` independently identified the same failure in the unchanged fused
control and proposed inspecting routing. Its recommendation to require the old
per-row gate is not an acceptance authority: that recommendation conflicts with
the owner's explicit whole-output contract. Its source findings and mechanism
were tested, not accepted as proof without measurements.

Controls in `stress34_controls.json` exclude expert dtype/fidelity, shared
precision and prefill config as necessary causes. `diagnostic34_fused.json`
reproduces the row with PCC .990553 versus optimized .990702.

`continuation_router_probe_fixed.json` localizes the row276 effect:

- HF versus TT postnorm input PCC .9999992645.
- HF on its own input chooses eighth expert101; TT chooses254.
- HF on the captured TT input also chooses254. MoE output PCC between TT and
  HF on identical TT input is .999799595; HF-own-input versus TT MoE PCC is
  .966754897. Tiny input changes around a routing cutoff explain the larger
  localized output difference; expert/shared arithmetic is not the cause.
- At row257, the inherited FP32 router logits versus HF's BF16 boundary produce
  another cutoff difference; it does not fail the complete decoder's per-row
  .995 screen. That inherited precision-contract difference is documented for
  later end-to-end precision validation, not silently attributed to BFP8.

The probe is explicitly diagnostic-only: device-only auditing is disabled to
capture intermediate tensors; no latency or acceptance claim uses its path.
The first probe accidentally recaptured HF inputs during its same-input control.
It is marked `diagnostic_valid=false`; the repaired probe freezes the first HF
capture and is the only probe used for these conclusions.

Actual HF-propagated natural-text activations pass every strict row at the same
128/128/33 boundary (`natural_stress_34.json`, min PCC .999873). This does not
prove general model quality and is not described as a same-settings HF-versus-HF
known limitation. No precision fallback or production arithmetic change is made
on the basis of synthetic-only per-row evidence.

## Minimal changes

- `stress_fused_decoder.py` retains its original strict default. An explicit
  whole-output mode computes actual concatenated decode/continuation PCC and
  whole contribution PCC, and retains all row diagnostics. Failure tensors now
  include continuation input/reference/output, which the old artifact omitted.
- `stress_optimized_decoder.py` explicitly selects that whole-output mode, the
  owner's acceptance contract, while keeping tracking, exact repeats, state
  restore and cache continuation checks unchanged.
- `check_optimized_evidence.py` checks every aggregate and validates that no
  diagnostic row was hidden. Earlier strict per-row passing artifacts remain
  valid stronger evidence. Failed per-row artifacts are not retroactively
  relabeled; `initial_stress_34.json` is preserved and new gates rerun.

## Verification

Source compiles and Black passes. Exact device commands and results are in
`logs/stress34_controls.log`, `logs/diagnostic34_fused.log`,
`logs/natural_stress_34.log`, `logs/continuation_router_probe_fixed.log` and the
corresponding JSON/provenance. Current-source128/1024 layer34 aggregate gates
run anew in `logs/stress_34_whole_output.log` and
`logs/stress_extended_34_whole_output.log`; final status is checked by
`hard_gates.json`, never inferred from this document.

Residual risk: near-tied expert selection is discontinuous under small upstream
rounding differences. Whole-output PCC is the specified decoder gate; stage8
and full-model accuracy must measure model-visible effects. No full-model
accuracy, generation quality or universal per-row .995 claim is made here.
