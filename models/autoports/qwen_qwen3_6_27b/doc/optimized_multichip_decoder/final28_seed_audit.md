# Final28 seed template audit

CPU/source-only review of `final28_seed_template.json`, the exact `check_optimized_multichip_evidence.py` schema, and `final28_evidence.py assemble`. This is an assembly audit, not a stage pass or stage-review verdict. No device/source-test edits or hardware execution.

The seed's existing bundle mappings match the canonical68-job validation plan: correctness10, independent HF contract2, interlayer2, resource2, stress8, watcher12, trace-allocation2 (reuse of B32 correctness), and context2. A CPU call to `build_jobs` verified every result/command/log basename and every watcher `.watcher.log` path. Both independent stress process repetitions are present. The native build receipt points to the existing `fused_rms_minimal_build27.json`; the final assembler supplies `final28_native_build.json` and exact source/policy/perf/profile groups. The seed correctly leaves review and commit pending.

Required field completion, using repo-relative paths under `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/` unless stated otherwise:

| Field | Required value/action |
|---|---|
| `checks.runtime_audit.scope_review` | `final28_runtime_scope.md` (exists) |
| `context.capacity` | `final28_capacity.json`, only after gated updater succeeds |
| `documents.topology_audit` | `topology_audit.md` (exists; retain original operation-topology audit) |
| `stop_rule.evidence` | Keep `final28_handoff.md`; add existing `user_wrapup28_runner.txt` and preferably `user_wrapup28.json` for exact runner/user provenance |
| `review` | Actual independent report path(s), accepted verdict, actual round count, fixed findings and explicit carried findings; do not synthesize a pass |
| `local_commit` | Actual40-character local SHA plus stage-owned paths whose committed bytes equal checkout; never push |

The README, work log, `final28_interlayer_contract.md`, `final28_selection.md`, and `final28_handoff.md` names agree with intended final files. They still need final measured results/limitations, not assumed successful runs. All validation summary/receipts, native receipt file hashes, artifact freshness and watcher logs remain actual final-gate dependencies. The template itself does not claim those jobs completed.

One concrete schema mismatch was found and repaired in the documentation-only capacity updater: `Gate.context` requires the exact `optimized_multichip_decoder.status == "validated"`. `final28_capacity.py` now emits that value only after its existing artifact gates; the descriptive limitation moved to `validation_scope`. Historical maximum-context provenance and explicit `new_final_source_max_context_run_claimed=false` remain intact. The updater was not executed; the context contract was not changed by this audit.

At audit time, `final28_runtime_audit.json` has zero source-hash mismatches against its current runtime helpers,48 passing negative controls, and static pass, but `runtime_sentinel.status == "not_provided"`. Root must attach actual final-default runtime sentinel evidence for both kinds after jobs finish, as already planned. Its scope text must remain explicit about the six guarded APIs and executed eager/capture paths; static success alone is insufficient. The existing schema also checks each decoder bundle's runtime sentinel fields.

`user_wrapup28.json` contains the exact superseding instruction required by `Gate.stop_rule`, with search stopped and quality gates unchanged. The runner proof identifies the stage5 cap at14401 active seconds and wrap-up delivery. The seed uses `explicit_user_wrapup` rather than fabricating70% roofline or8h device-time acceptance.

Use `final28_evidence.py assemble --seed <completed-owner-seed> --advice <completed-advice-decisions> --output <final-manifest>` only after its final28 plan jobs and raw profile/table provenance pass. It intentionally overwrites the seed's source/native/effective-policy/performance groups from actual measured plan results. The subsequent evidence checker must still run after review/commit completion; assembly does not mark stage pass. Commit metadata must identify a real commit containing the listed source/artifact bytes, avoiding a self-referential claim that a manifest contains its own eventual commit SHA.
