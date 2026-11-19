# Stage 8 owner decision — 2026-11-16

> Owner decision: accept the resumed-prefill PCC miss (start 68 minimum PCC 0.968 vs 0.98; the matched HF BF16 control also fails minimum PCC on the same case, and top-1, final argmax and negative controls pass) as a known limitation, the same as the stage 6 decision. Keep the selected precision configuration, record this in limitations, finish the stage review and complete stage 8. No further resumed-prefill investigation.

Applied to closing stage 8 with the existing selected configuration and recorded resumed-prefill PCC failures. The owner specifically cites start 68; acceptance does not assert that HF reproduces every TT-only failure at starts 68, 2528 or 2559. Raw thresholds, failed results and HF comparisons remain unchanged. No further resumed-prefill investigation is authorized in this stage.

The matched HF control fails case-level PCC at 68 and 2528, but passes 2559; position-level failures only partly overlap. Top-1, final argmax and applicable negative controls pass on the final TT run. This is an owner acceptance, not a numerical repair or raw gate pass. See [owner_acceptance.json](owner_acceptance.json) for evidence.
