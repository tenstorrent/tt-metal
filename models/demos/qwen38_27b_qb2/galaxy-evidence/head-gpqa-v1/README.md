# Completed BFP8/HiFi2-head GPQA control

Full GPQA-D completed at **08:53:32 UTC, Oct 9**, scoring **166/198 (83.84%)**
in **51m43s** of measured evaluation time. The unchanged gate is 177/198; this
policy is not qualified. All 198 questions remain in the denominator.

The saved-response audit checked all original private responses against their
scored receipts. There were 197 natural stops (166 correct, 31 incorrect) and
one incorrect output-budget cutoff at 65,536 tokens. Both the original harness
score and the stricter naturally-completed-response score are 166/198. The
cutoff contributes no credit, and increasing its budget alone cannot close
the eleven-answer gap.

The input hash and full GPQA protocol match the previous native-head control:
temperature 1.0, top-p 0.95, top-k 20, seed 42, thinking enabled, 65,536 output
tokens, concurrency 128 across eight TP4 replicas. The model policy changes
only the head to BFP8/HiFi2; decoder projections remain BFP4/LoFi, with native
recurrence and accurate-full-tile decode attention. See the earlier
[G0 and serving receipts](../head-g0-v1/README.md) for the exact source bundle.

| Matched question outcome | Count |
|---|---:|
| Both runs correct | 157 |
| Native head only correct | 13 |
| BFP8/HiFi2 head only correct | 9 |
| Neither correct | 19 |

The native-head run scored 170/198. This control did not qualify an improvement;
one sampled run per policy does not prove that the head change causes lower
expected accuracy or localize the numerical difference to a kernel bug.

Mean client decode rate was 14.90 tok/s/user, aggregate output rate 447.60
tok/s, and mean TTFT 12.35 s for this reasoning workload. These are evaluation
metrics, not a fixed-shape decode roofline or a clean head-cost microbenchmark.

Artifacts include the original summary, protocol, answer receipts (hashes and
metrics without private answer text), compressed complete audit, audit-service
completion record and matched-question comparison. Raw responses remain on
the allocated host. The head service completed and stopped owned workers; the
persistent CPU follow-up advanced automatically at 08:54:05 UTC to image import
and runtime checks, followed by the HF reference. Those later checks have not
passed yet. The prepared decoder precision controls remain unqueued.
