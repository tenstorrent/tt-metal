# Sampler contract audit

## Decision

Selected: canonical `Sampling1D` exact split greedy. Each TP rank computes its local BF16 maximum and exact global index, a compact FP32 candidate packet is gathered across ranks, and a device op chooses the global winner. The winner is written to `tt_out_tok`, which is copied device-to-device into the next traced decode input.

Rejected common alternative: `TTSampling` exact greedy. It gathered the full BFP8 vocabulary before argmax, measured slower, carried broader mutable request state, and its explicit request-seed behavior disables the SamplingGenerator internal trace. It remains a comparison oracle, not the optimized path. Custom sampler code was unnecessary.

## Semantic comparison

The comparison was genuinely greedy: temperature-independent argmax with no top-k/top-p truncation and no stochastic draw. On real all-40-layer Falcon logits, all three paths selected token 2107:

| Path | Token | Mean latency |
|---|---:|---:|
| Sampling1D split greedy | 2107 | 0.8035 ms |
| Sampling1D force-argmax/full gather | 2107 | 1.0054 ms |
| TTSampling exact greedy | 2107 | 1.0091 ms |

The batch-32 gate independently compared split greedy with host argmax for every row across forward, repeat, slot-reversed, physical-page-remapped, and representative batch-1 cases. Every comparison matched.

## Shape, layout, and tracing

- Local candidate values stay BF16 until packed with exact global indices in a small FP32 rank packet.
- Synthetic Watcher coverage exercised all 32 rows and vocabulary-partition winners spanning token IDs 37 through 127106.
- The packet gather produced shape `[1, 4, 64, 1]` exactly; the predecessor hidden gather was also exact.
- Sampling is a separate trace from the model. Its output is the model trace's next token input; there is no Python token-feedback loop.
- In 128 paired full-model replays, sampling took 0.7939 ms/token, 4.14% of combined model and sampler time. It does not dominate token-out decode, so no LM-head/sampling redesign is required.
- Caller-visible generation reads one final sampled token per step only to return output. It never reads full logits and never uses that host value as feedback.

## Compatibility mode

`Generator.generate(..., sampling_mode="host")` is an explicit test-only compatibility mode. It gathers logits and permits a host callback/argmax where a readiness test requires it. The default `sampling_mode="device"` path is the traced split-sampling path measured above.
