# Gemma4 26B A4B IT — full model

Full-model TTFT **2115.07 ms** and trace-verified token-out decode
**49.22 tokens/s/user** (4096 input, 128 generated, batch 1,
one concurrent request; four Blackhole ASICs, TP4; warmed request trace).
Teacher-forcing decode: **49.84 tokens/s/user** on the separate
161-input/100-position AIME accuracy workload; it includes host token injection
and is not the autoregressive performance result.

The full model retains the optimized multichip decoder and 262144-token context.
Prefill and decode top-5/top-100 are both 100% over 100 AIME token positions.
See [full-model evidence](doc/full_model/README.md) and [work log](doc/full_model/work_log.md).
Independent [stage review](doc/full_model/stage_review_final.md): **clean-pass**.
Stage06 is complete; local checkpoint provenance is recorded in the work log.
