# Stage 11 independent review — initial

Date: 2026-07-17 UTC

Verdict: **MORE-WORK-NEEDED**

The independent `$stage-review` verified the exact autoport path, external no-Docker server mode, 113280-token context contract, smoke gates, final6 workflow completion, all 17 benchmark points, 21/21 API parameter-conformance tests, dynamic irregular-batch coverage, device recovery, and cleanup.

Required work found:

1. The `meta_ifeval` and `meta_gpqa_cot` rows were waived only because no published/GPU reference existed. That disclosure is not a valid row-specific waiver. The rows need exact canonical-control evidence or must remain release-readiness blockers.
2. The report and runtime spec named base revision `2be0f245e20`, while final6 exercised the dynamic-grid repair later committed as `97a16e1c982a27fbc2f4e27b65dbd6b077f9e34f`.
3. The successful final6 run log was not yet tracked.
4. An empty top-level `.env` remained in the TTI checkout.

Remediation status:

- Provenance was corrected to `97a16e1c982a27fbc2f4e27b65dbd6b077f9e34f`; the earlier base is retained separately in `RUN_NOTES.md`.
- The successful final6 log was copied and sanitized.
- The empty top-level TTI `.env` was removed.
- `$autofix` was invoked for the mandatory Meta rows. Its outcome and the rereview verdict are recorded separately.

The reviewer also noticed an unrelated pre-existing process exposing a credential through its argument vector. The credential is not reproduced here; rotation and safer secret passing are recommended outside Stage 11.
