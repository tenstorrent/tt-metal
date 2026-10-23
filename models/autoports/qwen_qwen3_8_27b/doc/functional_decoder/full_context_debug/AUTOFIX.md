# Full-attention long-context AutoFix

Verified repair: explicit SDPA decode q_chunk_size=32, k_chunk_size=256, max_cores_per_head_batch=16, exp_approx_mode=False. BF16 tensors, HiFi4 and FP32 destination accumulation remain.

Baseline full_262142_decode_max.json failed decode PCC .69018358/.60927504; prefill passed .996382296. Source-only AutoDebug report in this directory audits the long serial BF16 online softmax state, indices, page table, GQA, and trace lifetime. k512/core16 candidate failed the exact L1 capacity check (full_262142_decode_config_fixed.log). k256/core16 fits and passes with the same seeds/real weights: full_262142_decode_k256.json, prefill .996382296, replay .999959588, changed input/position .999991536, bitwise repeatability. All262144 context positions remain supported.

The combined configuration is experimentally verified. Individual effects of chunk size, parallel reduction and exponent flag are not isolated; no claim of a proved single-knob root cause or upstream kernel repair. No invalid L1 configuration is retained.
