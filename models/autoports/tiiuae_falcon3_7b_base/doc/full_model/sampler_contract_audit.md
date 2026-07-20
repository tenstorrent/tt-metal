# Sampler contract audit

## Common path comparison

| Common path | Fit | Decision |
| --- | --- | --- |
| `models.common.modules.sampling.Sampling1D` | Accepts TP local-vocabulary shards, device sampling parameters, device seeds, and caller-owned token output; supports split trace capture | selected |
| `SamplingGenerator` / `TTSampling` | Its exact-greedy mode first all-gathers the complete vocabulary and then reduces it; it is semantically correct but slower, while its broader request-seed state disables the abstraction's internal trace ownership for mutable request parameters | rejected |

No model-local custom sampler was written. The selected common `Sampling1D`
module was extended with a reusable exact-greedy method: local BF16 max/index,
four-candidate Linear gather through the common TP4 fallback, and on-device
winner selection. The decoder's persistent two-link Ring collectives are
unchanged; this tiny sampler packet follows `Sampling1D`'s canonical 1D
collective selection.

## Semantic and performance control

The exact split result was compared on hardware with both common controls from
the same TT logits: `Sampling1D` full-vocabulary all-gather plus force-argmax,
and an actual `TTSampling` exact-greedy call. All three returned token `2107`.
Over 20 warmed calls, exact split greedy measured 0.806366 ms/call, full gather
plus force-argmax measured 1.008621 ms/call, and `TTSampling` measured 1.008905
ms/call.

The selected sampler trace measured 0.793900 ms/token against 12.416047 ms for
the model trace: 6.009% of the pair. It therefore does not dominate token-out
decode, and no LM-head/sampler contract rewrite is required. Reduced real-
weight Tracy evidence is under `profile/reduced`; its local max and argmax are
large only in the deliberately reduced one-layer graph, not in the 28-layer
token-out path used for the measured result.

Watcher independently covers both the selected two-tile FP32 candidate gather
and the rejected full-vocabulary BFP8 gather. The latter exposed generic
one-tile-scatter and invalid-Linear-endpoint guards in the async all-gather
writer. The complete trace then exposed a missing per-invocation
`PacketHeaderPool::reset()` plus a Watcher false positive for the valid
`RUN_MSG_REPLAY_TRACE` state. After all four source fixes, a 600-replay trace
control, the exact reduced regression, and the complete 28-layer Watcher run
pass.
