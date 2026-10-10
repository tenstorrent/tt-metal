# t308: open the #194 draft PR

Done (2026-10-10 07:35 UTC):
- Draft PR https://github.com/tenstorrent/tt-metal/pull/60209, head ae3469d0a26 (ttp/ltx23-main-pr, fast-forward from df9e5ecaac6), base main. Draft, no reviewers. GitHub says CONFLICTING (fused RMSNorm op + rotary_embedding_llama factory vs main #59195/#58676); the body says a rebase and a rerun of the A/B come before review.
- Unblocked checks: harness commit a794f64 adds checks/ltx-ref-tests.sh (runs the ltx-rt-only CPU ref tests only where present); project.json push_checks points at it (uncommitted in harness, next to the coordinator's own uncommitted home_timezone line). On ltx-rt trees it still runs all 80 tests.
- `ttp checks` passed on ae3469d0a26 (3 checks).
- PR body: tt-project/t308/PR_BODY.md; summary corrected to the tables (S2 -0.14..-0.15 s, S1 -0.03..-0.06 s, Total -0.17 vs job 210 / +0.05 vs job 214 due to Encoder noise).

Next (not this task): rebase ttp/ltx23-main-pr on current main, resolve the 13 hunks, rebuild, rerun the 8+3 A/B, update the PR. Never gh pr ready without the user's yes.
