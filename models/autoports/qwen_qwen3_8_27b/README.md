# Qwen/Qwen3.8-27B TTNN autoport

Full-model batch-1 performance on four Blackhole p300c devices: **97.24 ms TTFT,
39.06 decode tokens/s/user**, warmed S128/G128, canonical split model/sampling
traces with on-device token feedback. AIME teacher forcing is **38.76 tokens/s/user**
at S203/G100 and 99.56 ms TTFT; it explicitly uploads reference feedback tokens.
These measurements include request reset and first-token sampling in TTFT.

Stage6 is validated with independent **clean-pass**; see [full-model evidence](doc/full_model/README.md).
Prefill and decode both achieve **100% top-5 and top-100** on the fresh AIME24 reference.
The public context remains **262,144 tokens**, with internal logical-length handling
and page32 KV caches. The decoder policy and TP4 strategy remain inherited from
[Stage5](doc/optimized_multichip_decoder/README.md). No vLLM integration is included.
