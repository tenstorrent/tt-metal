# Datatype sweep stage review

Date: 2026-08-13

Verdict: `clean-pass`

An independent fresh-context review found no remaining required work after remediation. It inspected the goal and skill contracts; precision-policy loader and runtime plumbing; raw candidate, non-aligned, token-out, and qualitative logs; selected and candidate policies; JSON/CSV schemas and provenance; Pareto plots; capacity arithmetic and context contract; README and work log; and metric/reference reconciliation.

Controlled anomalies:

- The first BF16-CCL readiness run had high setup/readback overhead outside the captured interval. Its internal traced samples were stable at 54.155957 and 54.268538 t/s/u, and only that trace interval was used for ranking.
- Nanobind leak diagnostics appeared during Python teardown after passing results and completed device closure; post-stage device discovery remained healthy.

Residual risks accepted by the review:

- Most rejected candidates have one timing sample; the baseline and selected policy have two, and the winner is corroborated by a matched warmed token-out benchmark.
- Accuracy is the required fixed 100-token AIME24 chat-template reference, not a broad evaluation suite.
- Selected BFP8 KV capacity retains 32,768 tokens from inherited physical-gate evidence, unchanged cache layout, and recomputed allocation accounting; BF16 KV was rejected rather than reducing capability.

No vLLM work was reviewed or started, as required by this stage.
