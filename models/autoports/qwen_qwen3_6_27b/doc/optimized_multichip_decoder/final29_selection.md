# Final29 selection after required HF-gate repair

This is the selection/provenance map for the repaired candidate, not a final acceptance or performance report. The authoritative selected overlay is `final29_selected_overrides.json`, which CPU AST inspection confirms exactly matches installed `_STAGE5_DEFAULTS`. Main source SHA256 at preparation: `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b`.

Relative to `final28_selected_overrides.json`, the sole overlay change is GDN `attention_fidelity="HiFi4"`; the full-attention overlay is unchanged. This changes GDN attention-role arithmetic in prefill and decode through its setup-created compute configuration. It is not a prefill-only exception or runtime policy mutation. GDN column grain1, cumulative-exp recurrence, packed correction, local beta/decay, compact residual ownership and norm-off selection remain. Stored activation/weight/cache/state dtypes and shapes are unchanged.

The change repairs a required gate after the optimization-search cap:

- `final_validation28_contract_l0_tp4.json` fails the independent HF B1/S31 contribution gate at row12, PCC0.9860137708810336. Aggregate PCC alone did not expose the failure.
- `final28_contract_stage4_l0.json` reproduces the same failing metric with frozen stage4 TP4. This establishes an inherited discrepancy, not permission to waive the gate.
- `final28_contract_hifi4_l0.json` passes the complete unchanged HF contract with actual selected weights and a diagnostic HiFi4 override. Its old source hash and diagnostic flag remain visible; it cannot substitute for the final29 default-path contract.
- `AUTOFIX_contract29.md` and `debug_contract28/AUTODEBUG.md` record the verified diagnosis and final closure; earlier diagnosis text is historical. The parent selected the supported global attention-role fidelity repair; narrower per-phase/per-operation hypotheses remain unselected.

The historical optimization family comparisons and rejected candidates in `final28_selection.md` remain provenance for selecting the layout/ownership family. Its numerical timing table and final28 main-source pin are historical, not final29 performance claims. Final29 numbers must come from `final29_evidence.py` on installed defaults, with frozen stage4 control, no candidate/source override on the final arm, all six workloads, independent B1 paired nonregression and final profiles/advice.

Final29 requires fresh final-validation receipts, both independent HF contracts, source/runtime and mapped-native provenance, B1/B32 resource ownership, context smoke, nonaligned continuation, stress, watcher, exact traced replay, capacity reconciliation and independent stage review. All68 final-validation and34 timing/profile jobs now pass, with runtime and capacity reconciliation complete; exact results are in final_manifest.json. B1 statistical nonregression is inconclusive and retained as such. Independent review/commit closure remains pending, with round1 requesting no correctness remediation.

The actual user stop-search instruction remains `user_wrapup28.json`, corroborated by `user_wrapup28_runner.txt`. This repair and final gating do not restart optional tuning or claim70% roofline/8h-device-time stop-rule satisfaction.

Final closure: independent round3 `final29_review_round3.md` returned pass-with-carried. Local shared/model checkpoints and the unchanged false confidence result are recorded in `final29_acceptance.json`; postcommit source/review provenance passes in `final29_postcommit_gate.json`. No further tuning or device run occurred.
