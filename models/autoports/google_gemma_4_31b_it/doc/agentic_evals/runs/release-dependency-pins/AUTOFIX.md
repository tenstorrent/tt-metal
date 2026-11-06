# Harbor source freeze repair

Diagnosis: AUTODEBUG.md in this directory. Earliest inconsistent decision is mutable Harbor ref selection inside setup_evals_agentic, before checkout/install.

Minimal repair: optional model metadata.harbor_git_ref validated as full40hex SHA and normalized lowercase. Omission retains stock branch selection. The unchanged checkout helper fetches that selected reference. Bounded git rev-parse HEAD verifies/logs the actual commit; malformed/mismatching output or command failure stops before installation. Default branch selection likewise logs a verified full Git HEAD, without changing which commit its fetch selected.

Changed source: workflows/workflow_venvs.py. Tests: tests/workflows/test_harbor_revision_pin.py. Local commit 7079fe20b4a7c0eb8dc441fd0e0bc455bafda97c on defect branch fix/model-scoped-harbor-revision. No model pin value, publication or active-checkout change.

Validation: 49 focused tests passed including default branch behavior, uppercase normalization, exact SHA forwarding, malformed/mutable ref rejection, checkout failure, HEAD mismatch and Git timeout/error. Ruff and diff whitespace checks passed. Full hooks status is recorded in RESULT.md and bringup/artifacts/stage12-harbor-pin-hooks.log when complete.

Remaining limits: actual CI Harbor SHA is not established by local source inspection; no actual CI checkout or task runtime was run by this effort. This fixes source identity selection, not transitive package locking. A Git verification failure now makes even the default route fail closed; that is intentional because an unverified checkout is not a reproducible source record.

Full hooks passed after replacing a suite-order-sensitive caplog assertion with a direct logger.info call assertion. The initial run printed the expected log but failed capture because earlier tests reconfigured propagation; no production change was required. Final log: bringup/artifacts/stage12-harbor-pin-hooks-final.log.
