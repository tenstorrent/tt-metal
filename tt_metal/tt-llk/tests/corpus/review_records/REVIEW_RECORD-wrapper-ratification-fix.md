# Review record — wrapper ratification-comment fix (external auditor lane)

Scope: repair of governance commit e1d668be1e's execution defects, found by the
2026-08-20 external verification (owner-side auditor, master-HANDOFF ledger 17):

1. Both sweep wrappers were broken: the 5-line RATIFIED comment block was inserted
   inside the multi-line `python3 sweep_2x2.py` continuation, truncating the command
   at `--phases` (dropping --skip-craq-gate/--allow-hardware/--baseline/drift args
   and "$@"), leaving an orphaned arg line -> `command not found`, RC=127.
   FIX: block relocated above the invocation (both wrappers). Verified: bash -n
   clean; stub arg-echo shows the full argument vector reaching sweep_2x2.py.
2. The pre-existing wrapper header still claimed the sweep "gates silicon on paired
   CRAQ" (lying-comment class): reworded to the ratified per-cell device-golden
   language in both wrappers.
3. sweep_2x2.py --skip-craq-gate argparse help still said "control experiments only",
   contradicting the ratified default: reworded.

Gates at this commit: python ast-parse OK, bash -n OK both wrappers,
selftest_enforcement_gates ALL GREEN, conf-lint GREEN on the real conf.
No conf/pin/baseline/measurement changes. Reviewer independence: this fix was
authored and verified by the auditing session that found the defect (not the
lane that introduced it).
