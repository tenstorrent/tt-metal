# North-Mini advisor contribution at decode batch 1

## Headline

Full-model decoder estimate: **39,940.817 us before and 39,940.817 us after,
with a +/-534.941 us uncertainty band**. The measured `$shard-advise`
contribution is therefore **zero**. The incumbent remains shipped unchanged.

This is a contribution measurement, not a claim that every surfaced cost is
zero. Every layer kind received a batch-1 frozen control and pinned-advisor
capture. In all three reconciliations, the entire advisor-attributable ceiling
was below that kind's own repeat spread, so the method prohibits screening.

| Layer kind | Layers | Estimate before/after | Band | Advisor ceiling/layer | Floor/layer | Verdict |
|---|---:|---:|---:|---:|---:|---|
| dense full attention | 1 | 274.481 us | 1.841 us | 1.709 us | 1.841 us | not measurable (0.93x) |
| sliding attention + sparse MoE | 36 | 29,795.040 us | 522.936 us | 1.706 us | 14.526 us | not measurable (0.12x) |
| full attention + sparse MoE | 12 | 9,871.296 us | 10.164 us | 0.562 us | 0.847 us | not measurable (0.66x) |

The counts come from the model config at revision
`d11e61a842617a22dc328552fa5bb86231ee4f37`: 49 hidden layers split as
1 dense/full-attention, 36 sliding-attention/sparse-MoE, and 12
full-attention/sparse-MoE. They match the decoder's `layer_types` plus
`mlp_layer_types` dispatch.

## Control and capture provenance

All timing used the fixed `advisor-challenger/scripts/harness_template.py`
protocol through `harness.py`: batch/requested batch 1, 10 untimed warmups,
five timed block means, 50 traced replays per block, and the median as the
control. Each kind ran in a fresh process. The captures occurred only after
the corresponding control and used `capture_template.py` through `capture.py`.

The advisor binary exists at `$TTMLIR_ADVISOR_HOME/build/bin/ttnn-advise` and
was run at pin `618cd4e75d`. Captures used the executed batch-1
`candidate=default` policy: BFP8 attention/cache, BFP8 dense MLP, BFP4 sparse
experts, and FP32 sparse router. These values are recorded in both the frozen
policy inputs and capture provenance; they are not constructor-default
inferences.

The sparse-MoE capture hit the documented terminal at `ttnn.sparse_matmul`.
The advisor reports therefore cover the complete attention prefix and declare
the sparse tail uncapturable. Reconciliation quantifies the resulting
untraced shares as 76.625% for sliding sparse-MoE and 77.154% for full sparse-MoE;
the stage does not imply advisor coverage of those tails.

## Reconciliation and decisions

All three `reconcile.py --incumbent ...` outputs close at 100% and are not
degraded. They rank the candidate boundaries by advisor-attributable removed
time. Because `feasibility.verdict` is `not_measurable` in every kind, none was
screened, combined, or shipped. This also means no advised-core-count sweep was
performed; doing one would violate the feasibility gate. Every losing knob is
named in `final.json` and remains default-off.

The unresolved sparse-tail rows were checked against source and the captured
IR. The 31.584/31.464 us `ReshapeView` rows consume L1-interleaved tensors and
correspond to shape-only sparse-output/routing reshapes; source places no
`to_layout` or `to_memory_config` on those edges, so they are compute/data-view
cost for `$optimize`, not advisor placement boundaries. The 4.732/4.791 us
`FillPad` rows consume DRAM-interleaved router metadata inside the uncapturable
sparse tail; neither source nor the reachable advisor IR changes placement on
that edge. They likewise are not bookable advisor contribution. Retilize rows
on the terminal tail remain undetermined by the advisor and are reported, not
screened.

`advised_boundaries.us_advisor_agrees` is 0.000 us in every kind. The profile
does show one-core `NLPCreateQKVHeadsDecodeDeviceOperation` costs of
25.452/26.016/25.950 us that the advisor independently agrees with; those are
accounted as `agrees_with_shipped`, not boundary-removal contribution. The
one-core RMSNorm costs (26.015/26.029/26.062 us) are surfaced for `$optimize`;
the stage does not screen them because all cells are `not_measurable` and this
experiment may not turn into a direct grid sweep.

`model_estimate.layer_handoff` reports no layer-boundary DRAM round trip in all
three kinds, so the handoff contribution is 0 us and was not screened.

## Correctness and shipped result

No source topology or precision changed. The unchanged incumbent retains the
completed optimized stage's real-checkpoint layer-1 oracle at PCC 0.995917.
No fresh-process confirmation is applicable because there is no winner to
confirm. `tt/optimized_decoder.py` is unchanged.

The first two dense profiler attempts overflowed device buffers and were
discarded because their signposted windows had no device rows. The retained
profiles use the same model hooks in profiler-only mode with ten warmups and
one bounded replay; all latency decisions still come exclusively from the
fixed timing template runs.
