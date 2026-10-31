# AutoFix: BF16 KV attention warmup

Failing contract: changing KV dtype must preserve all valid prompt lengths and262144 context. Reduced layers0/5 BF16 cache warmup failed at736 tokens with static SDPA CB end1408256 overlapping live L1 frontier1406208.

Independent diagnosis: AUTODEBUG.md reproduces the exact2048-byte overlap from the current SDPA factory. BF16 K/V doubles tile storage relative to the baseline BFP8 payload. Full-attention D512 prefillQ128/K128 exceeds available L1. No evidence of device hang or numerical failure.

Minimal repair: tt/multichip_decoder.py selects prefillQ64/K128 only for BF16 full-attention layers. The existing32/32 offset fallback and1024-token outer chunks remain. Static calculation predicts299008bytes/core saved. AlternativeQ128/K64 remains a tuning comparison if this policy becomes competitive.

Validation completed: smoke_kv_bf16_repaired.log exits0 with TT_METAL_TRACE_ALLOC_TRACKING=1 and TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0. All32 physical warmup buckets through1024 succeeded, followed by traced generation at logical129/1023/1025 (smoke_kv_bf16_repaired/smoke.json). Full60-layer kv_bf16/readiness_accuracy.json subsequently passed the 90%/98% accuracy gates (minimum top1 91%, top5/top100 100%); performance_b1.json and long_prefix.json record completed autoregressive and retention checks. No context reduction. BF16 KV is rejected because its target32x32768 cache alone exceeds usable DRAM (kv_capacity_estimate.json), not because of the repaired admission failure.
