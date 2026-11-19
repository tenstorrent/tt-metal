# Matched controls, 2026-11-13 17:45 UTC

The initial late-linear failure is an overstrict per-row assertion in the shared
stress harness, not a failed whole-output gate or a demonstrated optimized
regression. No HF-known-limitation claim is made without a matched HF control.

`diagnostic34_fused.json`: unchanged fused decoder same layer34/seed154,
128prefix +128decode +33continuation, tracking/program-cache enabled,
three repeats. Row276 PCC0.990553211/contribution0.988274663. Aggregate decode
0.999995226, continuation0.999855066. Prior final fused stress used8192 decode
steps, so it never tested this33-row continuation boundary at256.

`stress34_controls.json`: optimized row276 .990702112 (slightly closer), whole
decode .999913701, continuation .999794035. Expert HiFi2, BF16 experts+HiFi4,
BF16 shared+HiFi4 and automatic prefill matmul do not remove the row outlier.

`natural_stress_34.json`: actual HF-propagated teacher-forced natural inputs,
identical128/128/33 partition and three repeated traces: decode whole
.999965367, continuation whole .999963736, minimum row .999873394, all strict
per-row/contribution checks pass. Input provenance natural290.metadata.json.

Proposed correction: retain unchanged fused strict harness defaults; optimized
wrapper explicitly uses the owner's whole-output .995 acceptance, records
whole-output decode/continuation and contribution values and all per-row
outliers diagnostically. No precision or runtime changes. Independent review
must check this against the original user goal ('whole-output; per-row PCC is
diagnostic'), and must not mistake the historical diagnostic failure for a
current missing aggregate gate. Determinism/cache/tracker checks stay mandatory.
