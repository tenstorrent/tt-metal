# Final28 bounded evidence execution

Prepared CPU-only. No numerical, performance or stage completion is claimed by
these scripts. The final source is selected by the parent; no script chooses a
policy or requests another optimization search.

From the repository root, once source/defaults/native binaries are frozen:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
python "$D/final28_evidence.py" prepare
```

`prepare` is CPU-only. It creates the immutable `final28_plan.json` and
`final28_native_build.json`, pinning every model tt/tests Python source,
run_job, HF config, six existing exact baseline reference files, immutable
stage4 control source, and native source/binary/build inventory. It refuses to
overwrite an existing plan. It does not claim the copied baseline reference is
HF: it is the accepted frozen single-chip TTNN reference. The independent HF
contract is exercised by the separate validation suite.

Only after the currently running `final_validation28` suite releases the mesh,
run these two queues serially with exclusive device ownership:

```bash
python "$D/final28_evidence.py" run-perf
python "$D/final28_evidence.py" run-profiles
```

The first queue contains28 independent processes: five alternating stage4/final
B1/S512 pairs for each of layers0 and3, plus fresh stage4/final B32/S33 and
B32/S513 rows for each kind. Pair0 supplies the corresponding B1 six-row
before/after entry without an extra run. Both arms use the same current harness,
reference, native build, seed, timing estimator (three warmed prefill calls),
and100 warmed trace repetitions; only output and source-override arguments
differ. The control executes the original source SHA
`082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`.
No policy files or JSON overrides are passed to either arm. Final paths are the
actual current defaults. Before B32 uses its original public `[B,1,1280]`
decode upload, final uses `[1,B,1280]`; uploads are outside forward timing, and
neither forward receives a shim or fallback.

The second queue contains six separate profiled current-default processes,
one per measured shape, using five trace repetitions and profile-drain.
`render_perf --advice` produces raw CSV, gzip copy, prefill/decode report tables
and CSV, metrics, and both accounting passes. It deliberately does **not** use
`--plain-artifacts`: that existing renderer mode writes a null raw-ops hash,
which cannot satisfy the final gate. Normal rendering retains the raw CSV and
its actual SHA without changing any harness source.

Each device job records command/log/result plus a signed `.final28.json`
receipt, verifies mapped native libraries and the frozen source/build before
and after execution, and aborts on failure. Perf disables profiler, watcher and
allocation tracking; profiling enables only profiler. The default per-job
wrapper timeout is1800s. `D/STOP` aborts before the next job. No reset/recovery
is automatic. `--resume` skips only an already successful contiguous prefix
from the same plan. Existing partial/failed artifacts are never overwritten;
parent must investigate and explicitly archive a failed suffix before retry.
For any failed pair, preserve both original arms and restart that pair and its
suffix in declared order; do not reuse a later arm to repair an earlier one.

Output naming:

- `final28_pair_l{0,3}_p{0..4}_{control,final}.{json,command.json,log}`
- `final28_perf_l{0,3}_b32_s{33,513}_{control,final}.{json,command.json,log}`
- `final28_profile_l{0,3}_b{1,32}_s{512,33,513}.{json,command.json,log}`,
  with only the six supported combinations generated; profile artifacts are in
  `perf/<tag>/` and Tracy source CSV in `tracy/<tag>/reports/`.

## Assembly and minimum remaining artifacts

`final28_seed_template.json` maps the actual `final_validation28_manifest.json`
job names to the required groups. It is explicitly incomplete and has no pass
claim. Copy it to an owner-reviewed seed, replace PENDING fields, verify the KV
contract-change flag against the final selected policy, include acknowledgment
reviews if any tracked jobs report acknowledged buffers, and fill final review
and local-commit evidence only after they exist. The current template cites the
latest native build27 receipt; the checker verifies its installed hashes.

Minimum remaining groups are:

1. The exact68-job seven-phase validation plan, all signed receipts and summary:
   correctness/unaligned batches and sequence lengths, independent HF contracts,
   stacked residual boundaries,64-owner resources at B1/B32,32K context,
   independent changed-input stress, watcher and trace allocation controls.
2. The34 performance/profile processes above, plus all six pairs of CSV/report
   tables, raw-ops hashes and actual matrix/traffic accounting.
3. Final context/capacity reconciliation, runtime scope review with no unresolved
   fallback, topology audit, final README/work log, rejected-option evidence and
   explicit untried handoff. There is no need to fabricate8h or70%: the exact user
   wrap-up instruction is separately recorded and does not waive other gates.
4. Exact profiler advice dispositions, actual independent stage review and a
   local stage-owned commit. Do not push. Manifest source inventory includes all
   tt/tests modules; every such source must exist at the recorded local commit.

The advice input is a JSON list of objects with exact CSV `text`, `decision`
(kept/rejected/not_applicable/not_tried_stop_rule), nonempty `reason`, and actual
`evidence` paths. Existing geometry-matched decisions may be reused. No advice
is dismissed automatically. With no `--advice`, the assembler emits the missing
exact strings and fails rather than inventing an explanation.

```bash
python "$D/final28_evidence.py" assemble \
  --seed "$D/final28_seed.json" --advice "$D/final28_advice.json" \
  --output "$D/final28_manifest.json"
python models/autoports/qwen_qwen3_6_27b/tests/check_optimized_multichip_evidence.py \
  --manifest "$D/final28_manifest.json" > "$D/final28_gate.json"
```

Assembly writes `final28_before_after.csv` and `final28_b1_statistics.json`,
checks identical effective final policies across measured shapes and exact
before/after fixture metadata, and fills only measured/source/native groups.
It never sets review/commit/quality passes. The gate independently verifies the
zero-margin95% B1 paired upper bound (five pairs minimum) and all unchanged
PCC>=0.995, state, trace, accounting and context requirements. Failed gates
remain work.

CPU proof is `final28_cpu.json`:34 unique jobs,10 independent matched pairs,
all six reference hashes, frozen control pin, exact arm-command equality after
removing source/output, alternating order, no policy overrides, and no TTNN or
Torch imports. The narrowly authorized harness adapter was separately checked
with96 admission cases; source and base pins, policy rejection and PCC gates
were unchanged. The adapter is frozen before the running validation suite.
