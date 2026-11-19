# Exact format-only source closure proposal

Status: prepared CPU/source review only. No runtime source or existing verifier has been changed, and no numerical evidence is reclassified. Apply formatting only after the active PLE job closes.

`format19_transition_v1.json` binds the actual clang-format 19.1.4 ELF executable, root configuration, four exact before/after byte snapshots and diffs. The executable is invoked with `--style=file` against each original path; stdout is saved without writing runtime. Eight current verifier sources are preserved verbatim before any future integration.

## Reviewed source changes

- `gdn_compute.cpp`: whitespace and statement line splitting only; no lexical token changes.
- `ordered_projection_reader.cpp`: one declaration wraps after `=`; no lexical token changes.
- `gdn_compute_sfpu.cpp`: whitespace/comment spacing and statement line splitting, **plus braces around three existing single-statement `if (emit_state)` bodies**: `final.reserve_back(16)`, `pack_tile<true>(0, cb_state, ki * 4 + vi)`, and `final.push_back(16)`. Each existing call stays under the same condition and in the same order.
- `router_topk_algorithm.hpp`: ten added blocks around existing single statements (three early returns, one decrement, one heap-adjust call, two scan increments/decrements, a break, and the final if/else assignments). No declarations acquire a new scope, no label/goto is involved, and the final else remains attached to the same if.

Includes and preprocessor directives retain their exact order and spelling. No arithmetic, constants, type qualifiers, control predicates, FIFO/DST actions or call order changes. Formatting changes line numbers and may change diagnostic/debug metadata or source-keyed compilation caches; it is not proof of identical compiled binaries or measured timings on a recompiled binary.

## Proposed narrow verifier

Add `verify_format19_dependency.py` and CPU tests. Hard-code only the four authorized paths and exact before/after hashes, plus the manifest/config/executable identities. Check preserved snapshot bytes, reproduce exact canonical output with the pinned formatter, and separately check unchanged directives and expected single-statement brace additions. Reject any different byte, extra transition, different formatter/config or inventory mutation; do not generalize to arbitrary whitespace-normalized code or AST similarity.

Public API: `verify_format_transition(root)` returns exact current/previous identities and source-equivalence proof, requiring either the complete unchanged preformat set or complete canonical postformat set; `verify_source_identity(root, name, recorded_sha, format_binding=None)` permits a changed identity only for an explicitly bound authorized before-to-after pair. Exact equality still passes normally. Unknown binding fields/kinds/hashes and missing historical hashes fail. Preserve recorded hashes in every returned proof; never rewrite original records. The API proves source semantics only and returns stage acceptance not evaluated.

Negative tests: altered literal/operator, extra/missing brace, include/directive reorder, mismatched formatter/config, wrong preserved snapshot, unknown fifth path, changed shared Python file, missing recorded hash, mixed partial transition, stale binding, and altered historical/current inventory. Positive cases cover all four exact canonical transitions and records already made after formatting.

## Integration points (after parent approval)

1. `summarize_functional_evidence.evaluate_gate`: add a separate explicit `format_source_closure` binding, composable with the existing `source_closure=linear_qsa_constructor_v1`. Filter only proven exact format mismatches; all remaining mismatches must still satisfy the existing QSA branch proof or fail. Emit original/current hashes and both distinct proofs. Numeric checks and all-of semantics stay unchanged.
2. `verify_linear_qsa_dependency`: use the exact comparator for its baseline unchanged-runtime loop and `ADDED_RUNTIME` checks (the reader is one added file). Retain all 22 historical identities, exact 26-file current inventory and layer/config/constructor proof. The preformat QSA source/constructor proof itself is unchanged.
3. `verify_ple_capacity_dependency`: apply the same comparator to baseline shared/added files and required-runtime checks. Preserve the hardcoded layer-1 proof and deliberate failure semantics. No blanket exemption of expected failures.
4. `validate_trace_evidence.source_closure`: opt-in exact comparator while retaining full inventory and unchanged selected harness requirements. Performance validation must separately disclose source equivalence versus a newly compiled timing measurement; no new timing claim follows from formatting.
5. `validate_batch_capacity.validate_bound`: its independent initial `bound[source_sha256]` loop also needs the comparator. Refresh compound capacity/batch CPU reports after helper changes (new v4 records), retaining all old reports and helper snapshots.
6. `close_watcher_evidence`: pass the binding into normal source closure and the linear helper. Keep raw Watcher/log identities and all dump checks unchanged; regenerate its CPU closure with new helper hashes.
7. `summarize_synthetic_evidence`: its direct full-runtime hash loop needs the same explicit binding to regenerate a CPU summary from unchanged original six-test XML/provenance. Do not rerun or rewrite original pytest records.
8. QSA bridge preflight and version-1 prefix loader currently require literal runtime hash equality. The already closed bridge can be evaluated through the gate format proof; leave the loader's strict contract unchanged unless future device reuse is explicitly requested. Do not silently accept an old source-keyed export for a new invocation.

Nested validators pin their own source identities. Preserve current helper snapshots, then rerun CPU summary/closure producers with fresh provenance and bind those new outputs; do not whitelist arbitrary verifier-source changes in historical records. Gates requiring only device harness/runtime sources can retain original process evidence with the new explicit format proof. The active PLE job's record must be closed and bound before any runtime edit.

No device work or shared helper/gate edits are authorized by this proposal itself.
