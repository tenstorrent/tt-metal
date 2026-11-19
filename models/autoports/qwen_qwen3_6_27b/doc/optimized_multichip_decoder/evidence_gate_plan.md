# Stage 5 final evidence gate

`tests/check_optimized_multichip_evidence.py` is a read-only, standard-library
artifact checker. It imports neither TTNN nor torch, launches no device jobs,
and does not modify prior-stage evidence or the context contract. It is ready
for a final manifest; **no stage-5 pass is claimed now**. Missing artifacts,
pending review, and pending commit each produce explicit failures.

Run from the repository root:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/check_optimized_multichip_evidence.py \
  --manifest models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/final_manifest.json \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/final_gate.json
```

Exit0 means every implemented artifact check passed; exit1 means pending,
missing, inconsistent, failed, or malformed evidence. Output is JSON with
`pass`, `checked`, and `errors`. It also accepts `--manifest -` for stdin and
`--repo-root /absolute/repo`. Artifact paths are **repo-relative**, regardless of
the manifest location; absolute paths work for artifacts, but committed source
and stage-owned path inventories must be repo-relative. Plain and gzip logs/CSV
are supported. The checker does not archive, hash-copy, or regenerate evidence.

## Required contract and existing runner reuse

The unchanged numerical bar is0.995 for prefill/decode, contributions, per-user
scores, state, replay comparisons, and stress. Finite output and exact
trace/eager state/output equality are mandatory. Historical stage-4 evidence
is accepted only as the declared **before** performance baseline, never as a
final-default gate. The accepted source pin currently used by the before rows
is `082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`.

| Evidence | Implemented check / existing artifact shape |
|---|---|
| Before and final performance | Six paired rows: layers0/3 each at B1/S512, B32/S33, B32/S513; standard `run_multichip_decoder.py` JSON, command sidecar and log |
| Default-path provenance | TP4, current source and command snapshots, empty `candidate_policy`, effective `selected_policy` identical to manifest; no `--policy-file` or nonempty `--policy-json` |
| Warm latency | Positive prefill and traced decode milliseconds, serial regime, positive timed replay count; final/before timings exclude profiler, watcher and allocation tracking |
| Baseline comparability | Matching shapes and exact reference metadata, declared accepted baseline source |
| Profile coverage | Same six final-default rows, separate prefill/decode CSV and text tables, metrics, raw ops and stage5 accounting; all four device IDs; source/result provenance and raw hash agree |
| Advice | Every nonempty CSV Advice string has a disposition, reason, existing evidence references; accepted decisions: kept, rejected, not_applicable, not_tried_stop_rule |
| Accounting | Physical matrix shape/dtype equals uploaded result inventory; GDN recurrent traffic is786432*B; host/roofline reconciliation matches the profiled run |
| Functional and stress | Both layer kinds have B1/128-input/at-least128-step repeated changed-input trajectories, plus both kinds at B32; no new fixed B32 step count imposed; each run must have at least2 identical trajectories |
| Functional performance floor | B1 final profile must remain above10% of declared decoder roofline; this is a decoder-stage sanity floor, not a claim about unimplemented full-model generation |
| Watcher | Both kinds, watcher10 with profiler disabled, nonempty watcher logs without error signatures |
| Trace allocation | Both kinds with a representative B32 case, tracking enabled, skip-program-cache0, rejected positive control; explicit review artifact if buffers were acknowledged |
| Context | Current optimized-multichip contract preserves262144 and TP4; both kinds run at least32K with traced continuation and current default policy; capacity artifact exists |
| Interlayer | Stacked GDN/full attention, exact trace, PCC>=0.995, width1280 boundary, current effective policies; a boundary conversion requires a measured exception artifact |
| Resources | Current default policies, TP4, passed repeated replay, no late program-cache growth, persistent preparation before capture, trace fits region, allocation tracker active with positive control |
| Runtime audit | Current hashes for optimized_decoder, multichip_decoder, flat_prefill, recurrent_variants; nonempty reachable-function inventory and zero unresolved violations; scope review artifact |
| Completion | Required reports, explicit user stop rule, accepted independent stage-review verdict and local commit containing the source/path inventory |

This reuses the substantive stage-4 checks in `collect_multichip_evidence.py`,
`run_multichip_decoder.py`, `run_multichip_context.py`,
`run_multichip_stack.py`, and `check_multichip_stack_resources.py`. It deliberately
does not clone stage4's entire fixed shape grid, repeated max-context suite, or
its hardcoded precision policy. Additional risk-specific checks can be listed
under `checks.correctness`. Existing B1/128 and B32 sustained-decode runners
are suitable; the independent reviewer judges whether step count, inputs,
boundary shapes and persistent-buffer coverage match the final implementation.

Do **not** run the old `audit_multichip_runtime.py` as stage5 acceptance: it
currently writes into the stage4 directory and only indexes two source modules.
Produce a stage5 audit that includes the newly reachable helper paths. Setup-only
torch and host string transformations that generate device kernel source are
not numerical runtime fallback merely because they are imports or Python calls.
`scope_review` must explain that distinction, CCL/native-op coverage, and any
limits of the AST reachability analysis. This gate checks the resulting audit
artifact; it does not infer those semantics from import names.

The new CPU-only implementation is available:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/audit_optimized_multichip_runtime.py
```

It writes `doc/optimized_multichip_decoder/runtime_audit.json`; `--output`
changes the destination and `--stdout` writes no artifact. Optional repeated
`--runtime-result RESULT.json` attaches existing final-default sentinel evidence
without executing it. The current runner's sentinel specifically patches
`ttnn.from_torch/to_torch` during decode trace capture; it does not independently
guard prefill or every host/readback API. The audit records that limitation.
Imported candidate helpers become reachable automatically when source-wired;
experimental instance-installed callbacks need explicit `--extra-root` and
binding review. All module hashes reported by the audit must appear in final
`sources` and current run command snapshots, including newly wired helpers such
as `prefill_rope_candidate.py`.

The source command snapshots already supported by the stage5 `run_job.sh`
must include `tt/{optimized_decoder,multichip_decoder,flat_prefill,
recurrent_variants}.py`. The checker accepts either these model-relative keys
or full repo-relative keys. Final source pins alone cannot establish the code
that ran; command-side snapshots are therefore checked for every current run.

## Manifest schema

Every run bundle has exactly these required references (extra metadata is fine):

```json
{
  "result": "models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/FINAL.json",
  "command": "models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/FINAL.command.json",
  "log": "models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/FINAL.log"
}
```

Use full measured effective policies from final result JSON, not just policy
overrides. The top-level object is:

```json
{
  "schema_version": 1,
  "stage": "optimized-multichip-decoder",
  "pcc_threshold": 0.995,
  "baseline_source_sha256": "082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44",
  "sources": {
    "models/autoports/qwen_qwen3_6_27b/tt/multichip_decoder.py": "FINAL_SHA256",
    "models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py": "FINAL_SHA256",
    "models/autoports/qwen_qwen3_6_27b/tt/flat_prefill.py": "FINAL_SHA256",
    "models/autoports/qwen_qwen3_6_27b/tt/recurrent_variants.py": "FINAL_SHA256"
  },
  "final_policy": {"0": {}, "3": {}},
  "measurements": [],
  "profiles": [],
  "checks": {
    "correctness": [],
    "stress": [],
    "watcher": [],
    "trace_alloc": [],
    "interlayer": [],
    "resources": [],
    "runtime_audit": {"result": "PENDING", "scope_review": "PENDING"}
  },
  "context": {
    "contract": "models/autoports/qwen_qwen3_6_27b/doc/context_contract.json",
    "capacity": "PENDING",
    "kv_dtype_or_layout_changed": false,
    "smokes": []
  },
  "documents": {
    "readme": "PENDING",
    "work_log": "PENDING",
    "topology_audit": "PENDING",
    "interlayer_contract": "PENDING",
    "candidate_decisions": "PENDING",
    "untried_handoff": "PENDING"
  },
  "stop_rule": {},
  "review": {
    "verdict": "pending",
    "rounds": 0,
    "reports": [],
    "unresolved_blocking": [],
    "carried": []
  },
  "local_commit": {"sha": "PENDING", "stage_owned_paths": []}
}
```

This is intentionally an **incomplete template**, not passable evidence.
Replace placeholders and fill arrays as follows:

- `measurements`: six objects `{"before": RUN_BUNDLE, "after": RUN_BUNDLE}`.
  Shape/layer fields are read from each result, not trusted from the manifest.
- `checks.correctness`, `stress`, `trace_alloc`: arrays of standard decoder
  bundles. Correctness extras are optional; the six measurement rows already
  enforce both kinds, largest batch, and nonaligned logical lengths33/513.
  The same tracked stress bundle can also appear in `trace_alloc`; no redundant
  device rerun is required. Acknowledged buffers add `acknowledgment_review`.
- `checks.watcher`: decoder bundles plus `watcher_log` path. A tracked watcher
  bundle can also satisfy `trace_alloc`. Watcher/profiler must remain separate.
- `checks.interlayer`: standard stack bundles. If `no_boundary_conversion` is
  false, add `measured_boundary_exception` with actual comparison evidence.
- `checks.resources`: standard resource bundles; add `acknowledgment_review`
  only if allocation acknowledgments are nonzero. Each should represent the
  final ownership/CCL lifetimes. The checker does not force rerunning irrelevant
  shape variants or assume this shared-weight test validates full-model memory.
- `context.smokes`: context-runner bundles for layers0 and3, normally32K–64K.
  Context JSON must have `metadata.layer`, `metadata.length`, `pass`,
  `full_decoder_all_tokens`, `traced_decode_bit_equal`, `pcc`,
  `contribution_pcc`, `source_sha256`, `selected_policy`, and positive
  `largest_sampled_allocation_bytes_per_device`.
  Contract JSON requires `optimized_multichip_decoder` with `target_mesh`,
  `supported_context`, `decoder_source_sha256`, `status="validated"`, and no
  capability reduction.
- If KV dtype/layout truly changed, add `context.max_context_validation` with
  both kinds' passing262144 bundles. Otherwise leave it absent: stage5 does
  not need another maximum-context run. The capacity artifact should record
  policy, physical pages, CCL signatures, reserve and observed peak explicitly.
- `documents` entries can point to the same real report where topics share a
  document. `untried_handoff` may be the work log, but must explicitly record
  untried items and why the authorized stop rule ended optimization.
- Add other stage-owned source files to `sources` as needed. Every entry must
  match both the current checkout and the accepted local commit.

Each `profiles` element is a run bundle augmented with these fields:

```json
{
  "result": "PENDING",
  "command": "PENDING",
  "log": "PENDING",
  "prefill_csv": "PENDING",
  "decode_csv": "PENDING",
  "prefill_table": "PENDING",
  "decode_table": "PENDING",
  "raw_ops": "PENDING",
  "metrics": "PENDING",
  "accounting": "PENDING",
  "advice": [
    {"text": "EXACT_NONEMPTY_CSV_ADVICE", "decision": "rejected",
     "reason": "Measured reason, including adaptations for material candidates",
     "evidence": ["BEFORE_OR_AFTER_ARTIFACT", "DECISION_REPORT"]}
  ]
}
```

The raw ops file may be `.csv` or `.csv.gz`. `metrics.json` should retain the
existing profiler schema, including `correctness_result`, provenance, commands,
raw hash and separate prefill/decode windows. `accounting.json` uses the current
stage5 `account_optimized_multichip_perf.py` extension. No profile may use an
earlier candidate's policy to stand in for the final default. An empty advice
list is valid only if the actual CSV contains no advice.

## User stop rule, review, and commit

The user explicitly requires **(>=70% roofline AND last two kept gains<3%) OR
8 hours of device time**. The installed skill's newer default wall cap,
checklist-complete exit, or plateau without70% do not replace that instruction.
Choose one schema:

```json
{
  "reason": "roofline_and_plateau",
  "scope": "Declared target workload and aggregation choice",
  "roofline_accounting": ["FINAL_PROFILE_ACCOUNTING_PATH"],
  "last_two_kept": [
    {"before_ms": 1.0, "after_ms": 0.98, "evidence": ["ACTUAL_COMPARISON"]},
    {"before_ms": 0.98, "after_ms": 0.97, "evidence": ["ACTUAL_COMPARISON"]}
  ],
  "evidence": ["WORK_LOG_WITH_ORDERED_KEPT_CANDIDATES"]
}
```

Those illustrative timings must be replaced with measured values. The selected
roofline files must be among the final six profile accounting files; every
declared target row must reach70%. The checker does not silently require all
six rows to reach70% when the owner's target workload is B32. The independent
reviewer checks that the declared scope is representative and that no redundant
movement, inactive resident copy, or allocator slack inflated the numerator.

Alternatively:

```json
{
  "reason": "device_time_cap",
  "device_hours": 8.0,
  "scope": "Stage5 device time; cite how the ledger totals were computed",
  "evidence": ["DEVICE_TIME_LEDGER", "WORK_LOG_WITH_UNTRIED_HANDOFF"]
}
```

This needs actual device-time evidence. Do not substitute calendar/wall elapsed
hours or multiply mesh elapsed time by chip count unless that is the owner's
explicit accounting definition. Stop rules bound optimization work; they never
waive correctness, trace, watcher, context, or review requirements.

`review.verdict` is `clean-pass`, or `pass-with-carried` only with at least3
rounds and an explicit nonempty carried list. `reports` references real
independent stage-review reports; `unresolved_blocking` must be empty. The
review's semantic conclusions are not recreated by this file-existence check.

`local_commit.sha` is a full40-character SHA of a local commit reachable from
HEAD. `stage_owned_paths` lists the stage-owned code/reports intended for that
commit; it must include `tt/multichip_decoder.py` using its full repo path.
The gate uses read-only `git show` and verifies committed content equals current
content. It neither commits nor pushes. The final manifest/gate output can be
written after that commit to avoid a self-referential commit-SHA cycle; list
the substantive committed report/source artifacts, not the post-commit gate
output itself. Pending review or commit fails clearly, even if all device rows
already pass, so an intermediate run cannot be mistaken for stage completion.

## CPU validation and limits

Verified PCC rejection for NaN/Inf/below-bar/boolean/empty values, empty and
malformed manifest failure without traceback, and rejection of a4-hour cap or
a weaker stop-rule label. Eight device hours passes only the stop-rule unit;
the remaining missing groups still fail. No device evidence was generated.

This is an artifact consistency gate, not an independent stage review. A
nonempty Markdown file is not proof that a material rejection was adequately
adapted, a kernel has correct synchronization, a device peak covers all
lifetimes, an advisory is inapplicable, or a resource test exercises enough
owners. Those questions remain for the authorized fresh stage reviewer.


## Superseding explicit wrap-up instruction (2026-11-13)

The user has now directed stop-search and final gate completion at the stage time/token cap. `user_wrapup28.json` preserves the exact instruction. The checker accepts `stop_rule.reason="explicit_user_wrapup"` with `instruction` pointing to that record and normal scope/evidence. This does not assert8h device time or70% roofline and does not waive any quality, provenance, context, runtime, independent-review or commit gate. The earlier stop-rule section remains the original contract history.
