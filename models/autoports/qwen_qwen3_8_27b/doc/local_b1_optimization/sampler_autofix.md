# Sampler candidate preparation

Status: pad128 workaround verified on device; this sampler form is rejected as slower.

`AUTODEBUG.md` ties the captured four-device hang to a producer/consumer mismatch:
the admitted width32 sampling input publishes one tile, while the pinned kernel
waits for and sorts pairs. Increasing capacity or reducing just the wait count
would not supply the missing operand. The failed pad32 probe is preserved in raw
evidence. Recovery completed before the lease transferred to the memory agent.

The smallest model-side workaround changes only the experimental
`tests/probe_greedy_topk1.py`: append negative-infinity values and zero token IDs to
width128 before final sampling. This matches the existing sampler's final width
and leaves the pinned backend and selected model policy unchanged. Local top-k
remains1 for this candidate; stochastic equivalence is not claimed.

The probe synchronizes and prints completion separately for each warmup, reads
all four replicas, checks real logits against CPU argmax, and requires exact
baseline/candidate token equality for unique and tied maxima. Controls cover all
vocabulary-shard boundaries, the highest token, within-shard ties and cross-shard
ties. For larger batch diagnostics rows differ. A tie mismatch rejects this
candidate; maximizing-set membership alone does not pass it. Preparation is
inside the timed trace.

Validation: `sampler-b1-topk1-pad128.json` completes with tracker1/program-cache
skip0. All four replicas match real-logit CPU argmax and baseline tokens; all13
tested unique/boundary/within-shard/cross-shard tie controls match exactly. Three
alternating200-replay samples give baseline450.628us vs candidate492.389us.
This form is slower, so it is not integrated or promoted to full64. No hardware
ran during the peer's lease. The probe proves the workaround for these tested
cases, not universal tie identity or a full64 TSU/TTFT gain.

Prioritized alternatives: (1) this baseline-width caller workaround; (2) reject
width32 in backend validation as a separate admission fix; (3) implement complete
single-tile reader/compute/intermediate handling. Alternatives2/3 would change the
pinned backend and require a separately built, identified runtime; neither is
part of this experiment.
