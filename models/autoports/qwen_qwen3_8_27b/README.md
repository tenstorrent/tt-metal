# Qwen/Qwen3.8-27B TTNN autoport

Optimized full-model performance on four Blackhole p300c devices, warmed B1:

| Measurement | Before stage7 | Final default |
| --- | ---: | ---: |
| TTFT, S128/G128 | 86.312 ms | **59.295 ms** |
| Traced token-out with immediate delivery | 39.433 t/s/user | 40.328 t/s/user |
| Traced token-out with deferred complete delivery | Not implemented | **40.385 t/s/user** |
| Traced teacher forcing with token delivery, S203/G100 | 39.300 t/s/user | **40.238 t/s/user** |

TTFT improves31.30%. Deferred decode includes one final history transfer and
output construction; its steady loop keeps token feedback, sampling and
position/RoPE advance on device. Teacher forcing explicitly uploads reference
tokens and is measured separately. See [stage7 evidence](doc/optimized_full_model/README.md)
for exact commands, delivery boundaries, profiles, validation and review status.

Stage7 has an independent [clean-pass review](doc/optimized_full_model/stage_review.md).
Fresh AIME24 prefill and decode both achieve **100% top-5 and top-100**.
The public context remains **262,144 tokens**, including valid non-aligned
prompts, explicit cache/page/position state, fixed slots and inactive rows.
The selected [Stage5 decoder policy](doc/optimized_multichip_decoder/README.md)
and residual layout are preserved. [Stage6](doc/full_model/README.md) records
the completed starting model. No vLLM integration is included.
