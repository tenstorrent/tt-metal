# Held current C8 larger-cohort candidates

Both source-only candidates are ready locally, each directly based on immutable80 commit `42dc40754e465cf8c9ec993027ff59d3c936832c`. No rung is admitted.

- GPQA160: `8b9623a7f1ccee85c04319e4c83e00f974322201`; tree `aa799190cef6bdd23862f9867afae579cd214c52`; checkout `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa160-current-c8`.
- GPQA198: `e888c28d08be5dc020a04a252994a9013c65e88d`; tree `6bd9438fb7dbbe9ffdc2dafd1ab3947cc79e6530`; checkout `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa198-current-c8`.

Registered cohort lists and nested prefixes are exact. Only GPQA list/count and two matching test expectations change; all serving settings, release limits, native logging and stock harness/workflows remain unchanged. Entire trees equal prior tested C8 references.

Each candidate passed37 focused tests, full hooks, and exact AST/tree/source guards. The copyright hook also added an unrelated missing final newline; that file was restored and all hooks passed again on the intended files (including the full pytest hook). Both worktrees are clean.

Device cost/prediction/box:0h; host box:0.4h; no speed credit. No push, dispatch, device action, network probe, execution runtime edit or central-ledger maintenance.

Candidates remain held pending valid preceding rungs and measured schedule admission. If160 is published first,198 must be rebased onto that current best and its tree/exactness reconfirmed before publication; no sibling force-push. Timing projection and scores are intentionally absent.
