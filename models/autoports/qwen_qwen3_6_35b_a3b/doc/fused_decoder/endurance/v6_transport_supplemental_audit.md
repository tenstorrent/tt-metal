# Transport candidate supplemental source audit

New host-only [auditor](../../../tests/audit_fused_transport_v6.py), SHA256 `159bf733d9dd6586eadb07010f4173522538b0c6a7eeb65aee3c1690c1ef5029`. It supports exactly ab28 and the three reviewed transport candidates, including formatting that preserves the existing explicitly bounded legacy-description normalization. New substantive source changes require review; this is not a general optimization exemption.

The existing `audit_fused_snapshot_v6.py` remains byte-identical, SHA256 `ab9bca957f8b10215aaa2b4681b6c10874e7941c1d3279e2cbb44e4cfdaf80e8`, archived before this work in [v6_transport_auditor_base.py.snapshot](v6_transport_auditor_base.py.snapshot). The existing source-format compatibility manifest remains SHA256 `fe5a3db52da5678a8d8041cd5078ff89f55838256bd9ec3c93fae220e87a085e`; no pair, inverse-format proof, existing checker, production file, or production entrypoint was changed. New file passed pinned Black23.10.1/py310, autoflake2.3.1, isort5.13.2.

Validation is cumulative:

1. Pin all four source references and the old auditor raw hash. Candidate execution AST must match one reviewed reference.
2. On a copy, restore only the exact linear `_forward` placement input statement and missing output statement, and/or change only the selector `indices` assignment's function from `ttnn.reshape` back to the original wrapper. Scope is explicitly outside the first fullkind guard.
3. Require **whole-module** restored AST equality to ab28. No method/class is exempted or replaced; no setup whitelist is added; assigned C++ strings and executable constants remain exact. The existing two legacy-description whitespace normalization rules are unchanged.
4. Run the original auditor on the **actual candidate**, retaining runtime/helper scans, setup hashes, originalV5fullkind guards, functional signatures/KV/batch behavior,262144context checks, aligned/nonaligned planner tests and mock dispatch tests. Actual candidate runtime hashes and source hash remain in the report; restored-source hashes never impersonate runtime evidence.
5. Add nine negative controls: runtime host readback/upload in changed methods, arithmetic change in norm, triangular helper readback, setup extra statement, public alignment restriction, kernel string change, fullkind guard change, and index dtype alteration. Every mutation must fail the exact restoration check, in addition to the original21negative checks.

Reports: [base](v6_transport_audit_base.json), [placement](v6_transport_audit_placement.json), [indices](v6_transport_audit_indices.json), [combined](v6_transport_audit_combined.json). Each report records variant, inverse statements, whole-module restored hash, original runtime audit, source hashes, planner/dispatch checks and negative controls. Their presence/status is the evidence of completed runs; this memo alone is not a pass record.

Example standalone invocation (choose a new output filename):

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/audit_fused_transport_v6.py \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_ab28_transport_combined.py.snapshot \
  --output /tmp/qwen_transport_audit.json
```

After selection, the production entrypoint can retain `UNCHANGED` and `sha` from the old auditor and import `audit` from this supplemental module. Its return preserves all existing evidence fields and adds `transport_restoration`, `transport_negative_controls`, `transport_reference_hashes` and the old auditor hash. This integration is deliberately not performed before selection. No historical manifest mutation is required.

The source check does not execute TTNN, native kernels, tensor arithmetic, allocator/capacity tests or traces. The separately recorded [component result](v6_transport_cleanup_device.json) proves exact adapter outputs and measured improvements, not whole-decoder qualification. Whole-model streams, watcher, full context, current-source profiles and matched complete-decoder timings remain unchanged gates.

Completed host results: all four references passed; each ran80,784valid planner cases,168mock dispatch cases, all21original negative controls and9new transport mutation controls. No Torch or TTNN module was imported. All39non-setup runtime/helper functions remain audited on each actual source.
