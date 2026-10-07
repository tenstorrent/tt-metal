# Bring-up LOG (append-only; see models/common/bringup_kit/README.md)

## E261007-0043-ff3
- date: 2026-10-07T00:43:19
- commit: 43e71cc3770
- branch: ssinghal/mimo
- host: DC16-2-BG-0204-u20-44
- track: decision
- status: inferred
- claim: Coordinator start: read onboarding/skills/guide/study. Spot-checked against modeling_mimo_v2.py: router (sigmoid, bias for selection only, weights from unbiased score, norm_topk_prob, scale 1.0) matches study; SWA sinks, rope_dim=int(192*0.334)=64, v_scale present. NOT verified: qkv row layout, SWA mask off-by-one, sink convention on device, op limits. Hosts OK at 2026-10-07: 30 33 34 40 41 44; busy with DSV4: 32 43 45 47. Scorecard adds 64k gates; gsm8k threshold n=64/min 55 is a template default, provisional.
- evidence: -
- paths: bringup/mimo_v26/scorecard.json

## E261007-0110-968
- date: 2026-10-07T01:10:27
- commit: f010907ebbb
- branch: ssinghal/mimo-attn
- host: 30
- track: attn-probe
- status: verified
- claim: Phase A decode+prefill correctness, one BH chip (per-chip TP8 shape: 8 Q heads, 1 KV head, QK 192, V 128, bf16 KV, cache 2048, B=1 and 32 users with positions 0..2047): paged_flash_multi_latent_attention_decode with an explicit independent V tensor (head_dim_v=128) and non-MLA paged_scaled_dot_product_attention_decode with V zero-padded to 192 are both correct vs torch fp32 with sink and sliding window 128: PCC 0.99972 (SWA, sink/scale), 0.99996 (GA, B=32), min per-user 0.9995; identical numbers for both ops. SINK CONVENTION: device sink input must be sink/scale (kernel multiplies by scale); raw sink gives PCC 0.82 at sink x5 (decode B=32) and 0.963 (prefill), at real magnitude (x1) 0.998 so PCC alone does not discriminate. Prefill scaled_dot_product_attention(V pad 192, window 128, sink/scale) PCC 0.99973 (S=2048); chunked_scaled_dot_product_attention (paged cache 8192, chunk 2048 at 4096, window+sink) PCC 0.99972 (SWA) / 0.99966 (GA); flash_mla_prefill with explicit V (128) S=2048 PCC 0.99977 (= pad192 path 0.99977), out [1,8,2048,128]. sink x1 evidence: attn_probeA3_sink1.log.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeA3_sink5.log
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py
- data: {"pcc_dec_swa_sink_scale":0.999723,"pcc_dec_swa_sink_raw_x5":0.8196,"pcc_pre_swa_sink_raw_x5":0.9631,"pcc_pre_chunked_swa":0.99972,"pcc_pre_chunked_ga":0.999662,"pcc_flash_mla_prefill_explicitV":0.999769}

## E261007-0110-64d
- date: 2026-10-07T01:10:27
- commit: 0d3b876e60f
- branch: ssinghal/mimo-attn
- host: 30
- track: attn-probe
- status: verified
- claim: Phase A needle tests (one attractive key j0 with V=50, queries/users at j0+d): sliding window 128 includes exactly keys p-127..p in paged decode (pad192 and MLA128), prefill self and chunked-with-cache SDPA (d=127 included, d=128 excluded; abs max out 50 vs 0.09). Determinism: 5 repeated decode calls (pad192 and mla128, B=32) and 5 repeated chunked prefill calls (SWA and GA) bit-identical in-process. Ring: non-MLA paged decode with cache_position_modulo=256, window 128, sink, positions 255/256/5000 (B=1,4, V pad 192): PCC 0.99968..0.99975 (probe at 545d48c85a3, attn_probeA1_corr.log). paged_flash_multi_latent_attention_decode has NO cache_position_modulo argument. GQA acceptance at un-sharded heads: nh64/nkv8 (SWA) and nh64/nkv4 (GA) work for decode pad192 + mla128 (PCC 0.99995) and prefill pad192 (0.99975/0.99979), flash_mla_prefill explicit V nkv4 0.99979.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeA5_edge.jsonl
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py
- data: {"window_keys":128,"determinism_5x":true,"ring_pcc_min":0.999678}

## E261007-0110-595
- date: 2026-10-07T01:10:27
- commit: f010907ebbb
- branch: ssinghal/mimo-attn
- host: 30
- track: attn-probe
- status: verified
- claim: Phase A decode device-call time (wall time of 20 back-to-back calls after 3 warmups, one process, paired pad192 vs mla128 in the same process, host .30, per-chip shape 8 Q heads/1 KV head, bf16 KV, paged block 64, all users at the same position, DRAM interleaved): GA 64k cache pos 65535: B=32 pad192 3.973 ms vs MLA V128 3.313 ms (k_chunk 512; k128 3.978/3.308, k256 3.984/3.314) = -16.6%; B=8 1.029 vs 0.866 (k128) = -15.8%; B=1 0.171 vs 0.151 (k256) = -11.5%; pos 32767 B=32 2.010 vs 1.673 (-16.8%). K-only lower bound (V read skipped, NOT MiMo semantics) B=32 2.02 ms. Matches the byte ratio 320/384 = -16.7%: GA decode is DRAM bound (B=32, 64k: 1.34 GB read in 3.31 ms = 405 GB/s). k_chunk 128/256/512 equal at B>=8. SWA window 128: 0.025-0.09 ms per call for both ops at all B (below dispatch/timing floor, cannot resolve pad vs native; ring variant 0.059-0.079 ms).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeA4_dec_perf.jsonl
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py
- data: {"ga_b32_64k_pad192_ms":3.973,"ga_b32_64k_mla128_ms":3.313,"ga_b8_64k_pad192_ms":1.029,"ga_b8_64k_mla128_ms":0.866,"ga_b1_64k_pad192_ms":0.1707,"ga_b1_64k_mla128_ms":0.151}

## E261007-0110-d3a
- date: 2026-10-07T01:10:28
- commit: f010907ebbb
- branch: ssinghal/mimo-attn
- host: 30
- track: attn-probe
- status: verified
- claim: Phase A prefill device-call time (10 calls after 3 warmups, one process, paired, host .30, 8 Q heads/1 KV head, QK 192, causal, wall time): GA self-attention S=8192: pad192 SDPA 2.931 ms vs flash_mla_prefill explicit V128 2.464 ms (q256/k256) = -15.9%; q128/k128 4.539 vs 3.808 (-16.1%); S=2048 0.294 vs 0.256 (q256k256). q256/k256 is faster than q128/k128 (S=8192 -35%); q512/k512 does not fit L1 (CB 2.4 MB > 1.57 MB, attn_probeA5_edge.log). Chunked prefill reading a 64k paged cache (chunk 2048, only padded-V op exists): GA start 61440 13.17 ms (q256k256) / 15.87 (q128k128); start 30720 6.73 / 8.06; start 2048 0.73 / 0.85 per chunk per layer per chip; SWA window 128 chunk is flat in cache length: 0.19 ms (q256k256, 0.125 for q128k128) at start 61440/30720/2048. chunked_flash_mla_prefill takes V as the K prefix (MLA), cannot express independent V, so GA chunked prefill must read a V192 (padded) cache.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeA4_pre_perf.jsonl
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py
- data: {"ga_self_s8192_pad192_ms":2.931,"ga_self_s8192_mla128_ms":2.464,"ga_chunked_start61440_ms":13.17,"swa_chunked_ms":0.187}
