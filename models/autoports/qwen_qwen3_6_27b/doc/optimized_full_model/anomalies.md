# Investigated observations

## Small and noisy timing differences

Observed: same-harness readback decode median moves30.97637→30.95694t/s/u;
parameter caching changes TTFT86.730679→86.746810ms. Evidence: `baseline.json`,
`readback_final.json`, `performance_final.json`. Affected path: full-model host
request/readback boundaries. Control: decode repeat ranges overlap; caching
A/B returns the same token and verifies1→0 redundant parameter uploads. At
non128 lengths both prefill A/B branches execute the same eager path; their
variation is not an optimization effect. Resolution: controlled; report exact
numbers and no significant speedup, retain the device-return contract and
upload elimination. Optional tuning stops under the recorded <3% rule.

## Profiler drain gaps and worker annotation

Observed: generic op-to-op annotations exceed the entire prefill signpost
interval; DRAM-core annotation does not describe actual compute workers.
Evidence: `perf/analysis.md`, raw CSV and `perf/runtime_matmul_policy.csv`.
Affected path: performance accounting only. Control: separate per-rank kernel
sums, observed raw program configs and wall timing. Investigation: exclude
cross-drain gaps and generic utilization percentages; never sum TP rank times.
Resolution: controlled, with explicit reduced-model extrapolation limits.

## Equal-logit tie ordering

Observed: split TopK chooses70001 versus PyTorch/force-argmax70000 on65 equal
maxima. Evidence: `sampler_split.json`, `sampler_argmax.json` and the prior
stage6 tie evidence. Affected path: unchanged common sampler tie order.
Control: both select an actual maximum in every row; unique-winner changed-input
cases agree exactly. Investigation: correct physical logits/padding and split
semantics checked; selected split0.512781ms versus argmax2.725168ms.
Resolution: controlled under the existing valid-maximum greedy contract; no
claim of lowest-index tie parity. See `sampling_decision.md`.

## Generic active-trace allocation warning

Observed: allocator warns that allocations while traces exist may be unsafe.
Evidence: final run logs, `full_trace_tracking.command.json`,
`trace_contract.command.json`, `trace_positive_control.log`. Affected path:
trace setup and prepared eager helpers. Investigation: strict repeated replay
with tracking1 and program-cache exclusion0, changed prompt/page data,
nonaligned eager lengths, all64 graphs and a positive control that raises.
No acknowledgments/scopes were added. Resolution: controlled by passing strict
tracking and stable trace/program/buffer identities; warning is not suppressed.

Quality/resume observations from the completed fresh runs follow below.

## Standalone readiness import

Observed: first prefill runner exits before evaluation with relative-import
ImportError. Evidence: `import_failure/prefill.log` and command receipt.
Affected path: standalone readiness module construction; package-mode probes
already resolve the helper correctly. Investigation: exact one-line source
diff and standalone-loader/helper identity proof in
`import_resolution_proof.json`. Fix: use the existing absolute autoport import
convention. Readiness gates rerun; no model graph or policy changed. CPU proof
and hooks pass. Resolution: fixed; the full-model standalone prefill rerun
passes100/100 at top1/top5/top100 (`prefill.log` and new command receipt).

## Resumed-prefill partition arithmetic

Observed: resumed versus uninterrupted logits are not universally bit-exact;
minimum individual PCC0.985754633. Evidence: `prefill_resume.json` and
`../full_model/prefill_resume_native_receipt.json`. Affected path: unchanged
bounded-prefill partition arithmetic. Control/comparison: fresh results reproduce
stage6 exactly, minimum case-mean PCC0.999431966, top1 agreement0.992217898,
all final tokens equal; the runner's individual-PCC floor is0.98. Both no-history
and fresh-suffix negative controls fail in both eligible windows. Resolution:
controlled within the previously accepted explicit runner contract; no new
numerical regression or lowered threshold.

## Short qualitative windows and matched model wording

Observed: shared-suite128-token windows mostly remain in default thinking; story
prompt2 begins with thinking thinking. Evidence: `qualitative_verdict.json`,
current TT and reused matched HF suites. Control: every fresh TT token history
is identical to stage6 TT; the same phrase occurs in matched HF outputs. All24
completions were read, with no sustained collapse or unintended language shift.
Separate readiness generation is128-token exact HF/TT parity and passes the
mechanical degeneracy gate. Resolution: controlled observational evidence; no
final-answer-success or statistical-equivalence claim, and no new failure
attribution requiring a10-seed blinded investigation.
