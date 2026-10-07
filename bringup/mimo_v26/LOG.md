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

## E261007-0053-27e
- date: 2026-10-07T00:53:45
- commit: 070deda149e
- branch: ssinghal/mimo-moe
- host: 33
- track: moe
- status: verified
- claim: MoE ops present in this build (src and _ttnn.so strings): experimental.ccl.moe_compute, deepseek_prefill dispatch/combine/unified_routed_expert_ffn/hybrid_routed_expert_ffn/routed_expert_ffn/moe_grouped_topk/masked_bincount/offset_cumsum/post_combine_reduce/moe_fused_swiglu/moe_hash_gate, deepseek.moe generalized_moe_gate. TTMoEGate python class absent (DSV4 python only). No rebuild needed on op-existence grounds (not yet run on device). Host 33 BUSY: foreign DSV4 job pid 1320945 (timeout 3000s), Phase A not started
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_opcheck.log
- paths: ttnn/cpp/ttnn/operations/experimental

## E261007-0102-0e6
- date: 2026-10-07T01:02:27
- commit: 3fa410abae9
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: moe_compute single card (1x1 BH, ring_n=8, width_shard=4) K=4096 N=2048 8 experts/chip top8 compute_only, UPSTREAM bfp4 weights: runs, matmul PCC 0.9902 (experts 6,7; upstream bfp4 gate 0.984). Control: upstream DeepSeek dims K=7168 bfp4 PCC 0.9885 on same host. So op limits/ring tile counts FIT MiMo dims at bfp4
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_A_control_mimo_bfp4.log
- paths: models/demos/blackhole/mimo_v26, ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute

## E261007-0102-5e2
- date: 2026-10-07T01:02:28
- commit: 3fa410abae9
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: moe_compute with bfp8_b weights (my uploader, same dims) FAILS: matmul PCC -0.001/0.005, inf values. Cause: weight CB cb_r2c_w0 hardwired tt::DataFormat::Bfp4_b in moe_compute_program_factory.cpp:772 of this build; DSV4 repo has patch MOE_COMPUTE_BFP8_WEIGHTS env flag (C++). bfp8 routed experts via moe_compute needs C++ patch+rebuild (needs user approval, private build dir); bfp4 is a precision change needing approval. TTMoEGate class exists in models/common/modules/moe (sigmoid+bias, 256 experts k8 supported); only DSV41Gate is DSV4-only. Ring weight packers: ttnn.experimental.moe_compute_utils.prepare_w0_w1_tensor_for_moe_compute / prepare_w2_tensor_for_moe_compute / get_weight_mem_configs; moe_hash_gate present in .so
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_A_moe_compute_bfp8.log
- paths: models/demos/blackhole/mimo_v26, ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute

## E261007-0102-74b
- date: 2026-10-07T01:02:37
- commit: 44dc4fb5b8b
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: handoff (partial, STOPPED): Phase A moe_compute done at bfp4 only; bfp8 blocked on C++ patch (CB hardwired Bfp4_b). Router MiMoGate + tests written but not run (no real-token dump / checkpoint path available). Awaiting coordinator decision: (a) approve private-build C++ patch porting MOE_COMPUTE_BFP8_WEIGHTS, or (b) approve bfp4
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_A_moe_compute_bfp8.log
- paths: models/demos/blackhole/mimo_v26
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

## E261007-0114-093
- date: 2026-10-07T01:14:22
- commit: 7f276ceac79
- branch: ssinghal/mimo-attn
- host: 30
- track: attn-probe
- status: verified
- claim: Phase A correctness re-run at 7f276ceac79 (supersedes first pass, same results; sink x1 run in attn_probeB_corr_sink1.log), one BH chip, per-chip TP8 shape (8 Q heads, 1 KV head, QK 192, V 128, bf16 KV, cache 2048, B=1/32, positions 0..2047 incl 127/128/129): paged_flash_multi_latent_attention_decode with explicit independent V (head_dim_v=128) and paged_scaled_dot_product_attention_decode with V zero-padded to 192 give identical PCC vs torch fp32 (sink + window 128): SWA 0.99968..0.99975 (min user 0.99939), GA 0.99965 (B=1) / 0.99996 (B=32). SINK: device input must be sink/scale; raw sink gives 0.82 (decode B=32, sink x5) / 0.963 (prefill self) / 0.978 (chunked); at real magnitude (x1) raw still shows 0.998 (not discriminating). Prefill pad192 scaled_dot_product_attention window+sink S=2048 0.99971; chunked_scaled_dot_product_attention (paged cache 8192, chunk 2048 @4096) SWA 0.99971, GA 0.99966; at a 64k cache chunk @61440: GA 0.99949, SWA 0.99972; flash_mla_prefill explicit V128 S=2048 0.99977 = pad192 path; GQA un-sharded (nh64/nkv8 SWA, nh64/nkv4 GA) accepted by decode pad192+mla128 (0.99990/0.99996) and prefill pad192 (0.99972/0.99979); ring (cache_position_modulo=256, pos 255/256/5000, B=1/4, window+sink, pad192) 0.99964..0.99975; native chunked V128 (K192/V128 paged) is rejected: TT_FATAL K and V hidden dim must match.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeB_corr_sink5.log
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py
- supersedes: E261007-0110-968
- data: {"pcc_dec_swa_scale_b32":0.999682,"pcc_dec_ga_b32":0.999961,"pcc_dec_swa_raw_x5_b32":0.8196,"pcc_pre_swa_self":0.99971,"pcc_chunked_swa_8k":0.999712,"pcc_chunked_ga_8k":0.999662,"pcc_chunked_ga_64k":0.999493,"pcc_chunked_swa_64k":0.999716,"pcc_ring_min":0.999637}

## E261007-0114-1e1
- date: 2026-10-07T01:14:22
- commit: 7f276ceac79
- branch: ssinghal/mimo-attn
- host: 30
- track: attn-probe
- status: verified
- claim: Phase A decode time per call at 7f276ceac79 (supersedes first pass; reproduced within 0.3%): 20 back-to-back calls after 3 warmups, one process, paired pad192 vs MLA V128, host .30, 8 Q heads/1 KV head, bf16 KV, paged block 64, all users at one position, DRAM interleaved. GA 64k cache pos 65535: B=32 pad192 3.973 ms vs MLA128 3.313 (k_chunk 512; k128 3.973/3.309; k256 3.984/3.319) = -16.6%; B=8 1.028 vs 0.864 (k128) = -16.0%; B=1 0.174 vs 0.1505 (k256) = -13.5%; pos 32767 B=32 2.009 vs 1.675 (-16.6%). K-only lower bound (not MiMo semantics) B=32 2.02. Byte ratio K192+V128 vs K192+V192 = 320/384 = -16.7%: decode attention is DRAM-bound (B=32/64k reads 1.34 GB in 3.31 ms = 405 GB/s), 9 GA layers x 3.31 ms = 29.8 ms per step at B=32 users per chip at 64k (before any bfp8 KV). SWA window 128: 0.037-0.074 ms per call for both ops, pad vs mla not resolvable (timing floor), ring (modulo 256, pos 65535) 0.057-0.074 ms.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeB_dec_perf.jsonl
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py
- supersedes: E261007-0110-595
- data: {"ga_b32_64k_pad192_ms":3.973,"ga_b32_64k_mla128_ms":3.313,"ga_b8_64k_pad192_ms":1.028,"ga_b8_64k_mla128_ms":0.864,"ga_b1_64k_pad192_ms":0.174,"ga_b1_64k_mla128_ms":0.1505}

## E261007-0114-cd6
- date: 2026-10-07T01:14:22
- commit: 7f276ceac79
- branch: ssinghal/mimo-attn
- host: 30
- track: attn-probe
- status: verified
- claim: Phase A prefill time per call at 7f276ceac79 (supersedes first pass; reproduced within 1%): 10 calls after 3 warmups, one process, paired, host .30, 8 Q heads/1 KV head, QK 192, causal. GA self S=8192: pad192 SDPA 2.944 ms vs flash_mla_prefill explicit V128 2.478 (q256/k256) = -15.8%; q128/k128 4.535 vs 3.816 (-15.9%); S=2048 q256k256 0.297 vs 0.255 (-14%). q256/k256 beats q128/k128 by 35% at S=8192; q512/k512 does not fit L1 (CB 2.44 MB > 1.57 MB; attn_probeB_long.log). Chunked prefill reading a 64k paged cache (chunk 2048, padded-V op only): GA chunk @61440 13.20 ms (q256k256) / 15.89 (q128k128), @30720 6.76 / 8.06, @2048 0.74 / 0.86 per chunk per layer per chip (13.19 ms in attn_probeB_long.log); SWA window-128 chunk is flat in cache length: 0.19-0.20 ms (q256k256) / 0.124-0.134 (q128k128). chunked_flash_mla_prefill takes V as the K prefix and cannot express MiMo's independent V; GA chunked prefill reading a cache therefore needs V stored at width 192. SWA self S=8192 0.45-0.53 ms.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeB_pre_perf.jsonl
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py
- supersedes: E261007-0110-d3a
- data: {"ga_self_s8192_pad192_ms":2.944,"ga_self_s8192_mla128_ms":2.478,"ga_chunked_start61440_ms":13.197,"swa_chunked_ms":0.198}

## E261007-0114-2df
- date: 2026-10-07T01:14:39
- commit: 7f276ceac79
- branch: ssinghal/mimo-attn
- host: 30
- track: decision
- status: verified
- claim: Phase A decision (proposal for coordinator, derived from E261007-0114-093/1e1/cd6 and E261007-0110-64d): native V=128 is available and correct in DECODE (paged_flash_multi_latent_attention_decode, explicit V, sink, window; -16.6% time and -16.7% KV bytes on global layers at 64k, B=32) and in non-chunked PREFILL (flash_mla_prefill explicit V, -15.8%, but no window/sink/paged args so global layers only). It is NOT available for chunked prefill reading a cache (chunked_scaled_dot_product_attention needs K,V equal width; chunked_flash_mla_prefill implies V=K prefix) nor with a ring cache (MLA decode has no cache_position_modulo). Proposal: SWA layers: padded V192 everywhere (non-MLA paged decode with cache_position_modulo ring + chunked SDPA with window/sink; SWA cost is tiny, 0.2 ms/chunk, <0.08 ms/decode call). Global layers: for bring-up use padded V192 for prefill and decode (one cache layout, correct, +20% KV bytes, decode attention 3.97 vs 3.31 ms/layer at B=32/64k); optimisation afterwards: prefill into a one-user V192 scratch and copy V[..., :128] into a V128 decode pool at user completion, or ask for a C++ change (chunked MLA prefill with explicit V; needs a private build dir and coordinator approval). q_chunk/k_chunk 256/256 for prefill (q512 does not fit L1), k_chunk 128..512 equal for decode at B>=8.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeB_dec_perf.jsonl
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py
- data: {"swa_v":"pad192","ga_v_bringup":"pad192","ga_v_target":"native128","ga_decode_gain_pct":16.6,"ga_prefill_self_gain_pct":15.8}

## E261007-0114-de6
- date: 2026-10-07T01:14:40
- commit: 7f276ceac79
- branch: ssinghal/mimo-attn
- host: 30
- track: skill-defect
- status: verified
- claim: Defects found during the attention Phase A: (1) kit: models/common/bringup_kit/__pycache__/*.pyc were tracked, so running any kit tool dirtied the tree and launch.py/log_tool.py then refused (fixed: untracked + bringup_kit/.gitignore, commit d0f087b6b61). (2) tt-moe-correctness had no attention-op section; added section J (sink/scale convention, window=128 keys incl self, needle tests, binding quirks), validated at 0d3b876e60f. (3) ttnn.transformer.chunked_scaled_dot_product_attention rejects scale=<python float> (noconvert binding; omit scale, default is Q_dim**-0.5, checked by PCC 0.9997 vs 192**-0.5 reference). (4) black/isort pre-commit hook rewrites new python files so the first git commit fails and must be repeated after re-add. (5) launch.py defaults TT_METAL_CACHE to .../tt-metal-cache/kit, tt-lab-ops says h<host digits> (not changed). (6) ONBOARDING mentions mimo_sdpa_probe.md which does not exist: this probe replaces it (bringup/mimo_v26/ attn LOG entries + probe_sdpa_ops.py).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeA2_dbg.jsonl
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py

## E261007-0114-9dc
- date: 2026-10-07T01:14:58
- commit: 7510d7e4a07
- branch: ssinghal/mimo-attn
- host: 30
- track: handoff
- status: verified
- claim: Attention track handoff after Phase A (module NOT designed or written, waiting for coordinator ack on the pad/native decision E261007-0114-2df). Worktree /mnt/tt-data/ssinghal/mimo/wt/attn, branch ssinghal/mimo-attn. Probe: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py; repro: launch.py --worktree $PWD --commit <tip> --host 30 --env PROBE=<group[,..]> --env PROBE_OUT=<jsonl> [--env PROBE_SINK_MULT=5] -- python models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py; groups dec_corr dec_gqa dec_ring dec_perf pre_corr pre_gqa pre_perf edge pre_long. Logs /mnt/tt-data/ssinghal/mimo-logs/attn_probe*. Not done/unverified: real MiMo weights and activations (all data is random with real shapes), bfp8 weights, RoPE/V*0.707 path, TP8 mesh run (single chip only), 32-chip behaviour, determinism across hosts, bfp8 KV (not asked), 1 KV head per chip assumes KV replication for GA, trace-compatibility of chunk_start_idx_tensor variant (chunk_start_idx int variant only tested), decode with ragged positions at B>32 per chip, timing is host-wall with 20 calls (no device profiler).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_probeB_pre_perf.jsonl
- paths: models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py

## E261007-0059-cb0
- date: 2026-10-07T00:59:11
- commit: 7e0f49121b3
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: decision
- status: verified
- claim: qkv_proj layout (weight-level evidence, not yet a forward on real tokens): checkpoint rows are per-TP4-rank [q_r(16 heads x192)|k_r|v_r] x4, fp8 128x128 scale blocks run over each rank's rows contiguously (GA 3392 rows=27 blocks, SWA 3712=29; GA block 25 holds 64 k rows + 64 v rows). Evidence on qkv_L0.pt (GA): rank-blocked dequant gives the same q/k/v rms signature in all 4 ranks (k 0.07-0.10, q 0.018-0.037, v 0.0065-0.010), contiguous reading mixes sections (q .030 k .018 v .028); adjacent-window log-norm roughness 0.110 vs 0.164 contiguous; anomalous raw-norm rows 3264-3327 (and every 3392 rows), none at contiguous V start 13056. Consistent with GQA: rank r has q heads 16r..16r+15 and kv head r (GA) or 2r,2r+1 (SWA, n_rep 8). loader.deinterleave_qkv gives HF contiguous Q|K|V. Remaining proof (needs checkpoint): HF layer forward on real tokens.
- evidence: models/demos/blackhole/mimo_v26/tests/test_loader_qkv.py
- paths: models/demos/blackhole/mimo_v26/reference/loader.py
- data: {"layout":"per_tp4_rank_qkv","ga_rows_per_rank":3392,"swa_rows_per_rank":3712,"scale_blocks_per_rank_ga":27,"scale_blocks_per_rank_swa":29}

## E261007-0059-fcc
- date: 2026-10-07T00:59:20
- commit: e67e7aecf55
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: decision
- status: verified
- claim: BLOCKED: no local MiMo-V2.6-Flash checkpoint on /mnt/tt-data (searched ssinghal/, hf-home, hf-cache, hf-home-ds, weights; only config/index/qkv_L0/L1 samples in bringup_study/mimo; sgl.py there is a 404 stub). HuggingFace reachable from this host (HTTP 200). Need coordinator approval to download text-only shards (~170 GB, 65 shards, skip visual/audio) into /mnt/tt-data/ssinghal/mimo-ckpt. Until then: no real-token oracle, no fp4 nibble/e8m0 check, no dumps.
- evidence: bringup/mimo_v26/SKILL_DEFECTS.md
- paths: bringup/mimo_v26/SKILL_DEFECTS.md

## E261007-0059-2c0
- date: 2026-10-07T00:59:20
- commit: e67e7aecf55
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: skill-defect
- status: verified
- claim: tt-model-reference lacks a 'check for local weights / ask before download' step; study sgl.py is a 404 stub; ONBOARDING implies a checkpoint exists. Recorded in SKILL_DEFECTS.md items 4-5.
- evidence: bringup/mimo_v26/SKILL_DEFECTS.md
- paths: bringup/mimo_v26/SKILL_DEFECTS.md, .claude/skills/tt-model-reference
## E261007-0135-20d
- date: 2026-10-07T01:35:54
- commit: 9df3eedb884
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: DECISION (coordinator, from attn probe E261007-0114-2df, not yet merged here): bring-up uses V padded to 192 for SWA and global attention (non-MLA decode with ring cache, chunked SDPA with window+sink; sink input = sink/scale). Native V=128 decode is 13-17% faster on global 64k decode (paired, 1 chip, wall clock) and is a later perf item; the C++ route (chunked MLA prefill with explicit V) needs user approval for a private build.
- evidence: -
- paths: bringup/mimo_v26

## E261007-0135-56e
- date: 2026-10-07T01:35:54
- commit: 9df3eedb884
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: DECISION (coordinator): checkpoint downloaded from HF XiaomiMiMo/MiMo-V2.6-Flash-RL into /mnt/tt-data/MiMo-V2.6-Flash-RL (user approved download and location 2026-10-07; whole repo, ~173 GB, log /mnt/tt-data/ssinghal/mimo-logs/download_ckpt.log). MIMO_CKPT=/mnt/tt-data/MiMo-V2.6-Flash-RL. Hosts approved by user for use (health-check each first): 30 35 40 41 43 44 46 47 48.
- evidence: -
- paths: bringup/mimo_v26

## E261007-0157-4f4
- date: 2026-10-07T01:57:27
- commit: 1227f2c3d8a
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: Decode-like shape through the same ops/weights: N=64 tokens per row (min allowed, %64) unified op PCC 0.99925; host-timed eager step 1.8 ms (hint only, not a perf claim); N=2048 3.4 ms hint. So one per-expert bfp8 copy can serve prefill AND decode via dispatch/unified|fused|hybrid/combine (hybrid_routed_expert_moe = same layout, untested). moe_compute ring layout cannot be the shared copy: bfp4-only here, and RING_WEIGHTS mode of the unified op exists only in the DSV4 repo C++. DRAM at bfp8 per layer per chip: analytic 3*4096*2048*1088/1024 B*8 = 214 MB (+ND-shard N padding), x47 layers ~10.1 GB (analytic; ttnn memory view API returned 0, not measured)
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_unified_n64_1227f2c3d8a.log
- paths: models/demos/blackhole/mimo_v26, ttnn/cpp/ttnn/operations/experimental/deepseek_prefill

## E261007-0157-eef
- date: 2026-10-07T01:57:27
- commit: a061d266de2
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: Router modes on SYNTHETIC data only (random x, bias~10.4 std .03; NOT a routing claim): all three MiMoGate modes execute on device (1x1). set agreement plain 0.031, hilo shift0 0.082, hilo with calibrated shift 0.879, exact (fp32 matmul+sigmoid+ttnn.topk) 0.980. Shows hi/lo without the bias shift is useless at bias 10 (op sums bf16 score+b_hi); shift is mandatory. Real-token numbers pending checkpoint/track-0 dump
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_router_synth_smoke_1227f2c3d8a.log
- paths: models/demos/blackhole/mimo_v26, ttnn/cpp/ttnn/operations/experimental/deepseek_prefill

## E261007-0157-049
- date: 2026-10-07T01:57:36
- commit: 5671f1e9e9f
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: Prefill-style MoE on 4x8 (host 40), bfp8 weights (DRAM ND-sharded), MiMo dims 256 experts top-8 8/chip K=4096 N=2048, SYNTHETIC random hidden/gate/weights (not real tokens): masked_bincount+offset_cumsum+dispatch+unified_routed_expert_moe(Silu, no clamp)+combine+post_combine_reduce, column partials summed on host, vs dense torch bf16-weight fp32-math MoE: PCC 0.99925 per mesh row (per-token min 0.99914) N=512/row; same at N=64 and 2048 (commit 1227f2c3d8a); hifi2 and lofi identical (config seems ignored); skew case (1/3 tokens to experts 0-7, commit 9dffe3fdbb0) PCC 0.99924, worst chip load 5952 of 8416 rows (cf=4); 2 runs bit-identical; no capacity overflow. moe_fused_swiglu (bf16 out, packer_l1_acc must be false) PCC 0.99948. bfp8 prefill path works WITHOUT C++. Logs: moe_B_prefill_unified_hifi2_n512.log, moe_B_unified_lofi_n512.log, moe_B_fused_hifi2_n512.log, moe_B_unified_skew_n512.log
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_prefill_unified_hifi2_n512.log
- paths: models/demos/blackhole/mimo_v26, ttnn/cpp/ttnn/operations/experimental/deepseek_prefill

## E261007-0203-fac
- date: 2026-10-07T02:03:17
- commit: 479ee2847b3
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: bfp8 vs real MiMo fp4 experts (layers 1,20,46 x experts 0,137,255 x gate/up/down, device layout [K,N]; DSV4 nibble order low-first, e8m0=2^(s-127), sanity: 15/16 codes used, absmax 0.06-1.0, scales e8m0 118-125): bf16 holds the dequantised values exactly; bfp8 (host pack and device typecast identical) reproduces 24/27 tensors bit-exactly; 3 tensors differ in 1-27 of 8.4M values (L1 e0 gate: 27, max abs 1.95e-3; L46 down_proj: 6-17 values, max abs 7.8e-3, absmax 1.0); PCC >= 0.99999992. So bfp8 is essentially, not strictly, lossless (rare small values in a 16-block beside large ones)
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_bfp8_exact_479ee2847b3.log
- paths: models/demos/blackhole/mimo_v26/tests/test_bfp8_checkpoint_exact.py

## E261007-0210-e8c
- date: 2026-10-07T02:10:10
- commit: 42c08b7f5a0
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: Prefill path (dispatch/unified expert bfp8/combine/post_combine_reduce) on 4x8, synthetic data, N=256,512,1024,2048,4096 tokens per mesh row in ONE process: 5 runs bit-identical to run 0 at every N, per mesh row all True; PCC 0.99925 every row; 32-row corruption check (per-token PCC of all tokens, grouped in 32-row blocks, per mesh row): min per-token PCC 0.99909, zero blocks below 0.99; capacity ok. No corruption seen on this build/host 40. Also: masked_bincount requires tokens%64==0 (so decode at 32 users/row must be padded to 64 per row)
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_det_corrupt_42c08b7f5a0.log
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests/test_moe_prefill_4x8.py

## E261007-0215-4d9
- date: 2026-10-07T02:15:49
- commit: b435b8394f9
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: Decode-style 4x8 (ragged live users/row 32,20,7,1; 32 users padded to 64 tokens/row because masked_bincount needs tokens%64==0), REAL gate weights (bf16 in ckpt) and e_score_correction_bias (fp32) of layers 1/20/46 (bias mean 2.05/10.46/11.98, std .041/.025/.049), SYNTHETIC unit-rms hidden and synthetic bfp8 experts. Mode exact (fp32 matmul+sigmoid+ttnn.topk, weights from fp32 score): MoE PCC vs dense torch with the device's own routing 0.99922-0.99928 on live tokens, 3 runs bit-identical, set agreement vs fp32 CPU routing 1.00/0.983/0.967 (synthetic x: not a routing claim). Mode hilo (kernel op with calibrated shift): agreement 0.82/0.73/0.78 on the same synthetic input AND NOT deterministic across 3 runs (live tokens differ) in my wrapper. Proposal: use exact router; hilo kernel path rejected until a cause is found
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_decode_b435b8394f9.log
- paths: models/demos/blackhole/mimo_v26

## E261007-0230-284
- date: 2026-10-07T02:30:54
- commit: ab01b0c449b
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: Router on REAL ffn_in (track-0 dump s128b16, 2048 tokens, real gate+bias), 1x1 device, vs fp32 CPU routing of the same bf16 input; set match L1..L5: exact 0.9937/0.9883/0.9912/0.9946/0.9888 (flips 11-24 of 2048, ALL with fp32 gap<0.01, max gap 3e-4); plain TTMoEGate bf16 path 0.289/0.297/0.637/0.597/0.578; hilo kernel+shift 0.822/0.840/0.806/0.793/0.732. Gap<0.01 share of tokens 0.88/0.96/0.86/0.80/0.85 (routing is knife-edge; weights ~0.125 each). Floor bf16-chain dump vs fp32-chain dump: 0.970/0.881/0.873/0.894/0.856 (flips 99-100% gap<0.01). exact vs fp32-chain oracle dump: 0.969/0.879/0.873/0.895/0.855 = at the floor. Decision: exact router; the 99% set-match vs fp32 oracle gate is not achievable from bf16 upstream noise (floor 86-97%) but exact vs same-input fp32 is 98.8-99.5% with flips only at near-ties. hilo stage diagnostic (1x1, T=64, 5 repeats): logits, score, folded, kernel idx all bit-identical; the 4x8 non-determinism seen before is not reproduced here; hilo dropped (worse agreement, cause open)
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_router_real_ab01b0c449b.log
- paths: models/demos/blackhole/mimo_v26

## E261007-0251-c0a
- date: 2026-10-07T02:51:00
- commit: d89ecbbb0cb
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: Real-weight MoE layer on 4x8 (all 256 real experts fp4->bf16->bfp8, unified op, hifi2), real ffn_in of track-0 dump (2048 tok), vs bf16-chain dump ffn_out, PCC per mesh row. L1: dump routing 0.99982-0.99983; device exact router (set match 0.9937 vs dump) 0.99967-0.99979 (vs fp32-chain dump 0.99897-0.99967; floor bf16 vs fp32 dump 0.99926-0.99982); deterministic. L5: dump routing 0.9915-0.9949 (FAILS 0.999); exact router (match 0.9888) 0.9915-0.9949; floor bf16 vs fp32 dump 0.9985-0.9991. Own CPU recompute with the same reader (fp32 math, real experts, dump routing) matches dump ffn_out at PCC 0.99999 (L1 0.999997), so reader/fp4 decode/semantics confirmed vs track 0
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_real_layer5_d89ecbbb0cb.log
- paths: models/demos/blackhole/mimo_v26

## E261007-0251-bf9
- date: 2026-10-07T02:51:00
- commit: d89ecbbb0cb
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: Real skew from routing_idx (2048 tokens, 4 mesh rows x 512): L1 per-expert count max 1151 (expert 250 gets 56% of tokens) mean 64 (17 experts zero), max/mean 18.0; L5 max 356 (expert 27) max/mean 5.6, 2 zero. Worst chip load (rows, 32-padded) L1 1856 [per mesh row 1568/1088/736/1856], L5 896; max_buf at N=512/row: cf=1 2272, cf=2 4320, cf=3 6368, cf=4 8416 -> no overflow even at cf=1 for this small chunk (long chunks and later layers not yet measured)
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_real_layer1_d89ecbbb0cb.log
- paths: models/demos/blackhole/mimo_v26

## E261007-0251-559
- date: 2026-10-07T02:51:01
- commit: 4d8cda87b37
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: L5 MoE accuracy limiter is ACTIVATION precision: ffn_in has outliers (absmax 134 at L5 vs 13 at L1). The prefill ops hard-wire bfp8 activations (unified: dispatched x tilized to Bfp8_b in cb_in0_x, intermediates Bfp8_b; fused: bf16 output PCC 0.9955-0.9973; config fp32acc/fidelity ignored by unified). CPU emulation (bfp8 round trip of x only) L1 0.99997, L5 0.99839; h only L5 0.99987. Device probe 1x1 plain ttnn per-expert FFN bfp8 weights HiFi2 fp32acc, first 128 tokens L5: bf16 activations PCC 0.99989 vs bfp8 activations 0.99838. So bfp8 weights are fine; bfp8 activations in unified/fused cost L5 PCC (0.993-0.997 vs gate 0.999). Options need coordinator: C++ patch for bf16 x/intermediates in unified/fused (private build), or a bf16-activation expert path (routed_expert_ffn / per-expert matmul) at some perf cost, or accept
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_B_probe_L5_bf16_4d8cda87b37.log
- paths: models/demos/blackhole/mimo_v26
## E261007-0201-218
- date: 2026-10-07T02:01:39
- commit: 2d555515344
- branch: ssinghal/mimo
- host: coordinator
- track: infra
- status: verified
- claim: Checkpoint XiaomiMiMo/MiMo-V2.6-Flash-RL downloaded to /mnt/tt-data/MiMo-V2.6-Flash-RL: 65 safetensors listed in model.safetensors.index.json all present, summed file bytes 172,932,505,264 vs index total_size 172,923,364,096 (+9.1 MB = headers). Not hashed against HF; sizes only.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/download_ckpt.log
- paths: bringup/mimo_v26/scorecard.json
- data: {"ckpt_files":65,"bytes":172932505264}

## E261007-0228-fbc
- date: 2026-10-07T02:28:57
- commit: 15702a7f60a
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: DECISIONS (coordinator, 2026-10-07): (1) no L6-47 CPU chain; later layers verified teacher-forced on device-captured hidden states (policy: no full-model CPU run). (2) Router gate interpretation per tt-moe-correctness A.3: expert-SET match with mismatches only at near-ties; track-0 floor says bf16-CPU vs fp32 matches the same set for only 90.6-98.6% of tokens (L1-L5, E261007-0228-84b), so a raw 99% vs bf16 is unreachable by noise; report flip rate per layer and share of flips with gap<0.01 against that floor. This is a clarification not a relaxation; user to be told in the status report. (3) exact router mode proposed by moe agent (E261007-0215-4d9, synthetic data) pending real-token check; hilo non-deterministic, rejected until isolated.
- evidence: -
- paths: bringup/mimo_v26/scorecard.json
## E261007-0159-101
- date: 2026-10-07T01:59:21
- commit: 97024a064ab
- branch: ssinghal/mimo-attn
- host: 30
- track: attn
- status: verified
- claim: Attention module v1 (models/demos/blackhole/mimo_v26/tt/attention.py) on the real 4x8 mesh (host .30), random weights of real shapes (bfp8 qkv/o, bf16 KV, V padded to 192 with V*0.707 folded into Wv, TP8 over columns, GA 4 KV heads on 8 columns, partial RoPE 64/192 split-half via x*cos+(x@R)*sin, 4 mesh rows with different data) vs torch fp32 HF-semantics reference: chunked prefill (chunk 1024 x3, ragged lengths user0 3 chunks / user1 2 chunks, 2 users x 4 rows) PCC min 0.99956 (SWA, ring 24 pages, sink x1) and 0.99961 (GA), per chunk >= 0.99954; decode (32 slots/row, 3 steps, positions 3072.. past the SWA ring of 1536 slots, ragged 2048/3072) PCC min 0.99939 (SWA) / 0.99937 (GA); column outputs identical after all_reduce (spread 0.0), 3 repeated decode steps bit-identical. Same at 5 chunks x3 users ragged (C5): prefill 0.99955/0.9996, decode 0.99945/0.99953; bf16 weights give 0.9996 so bfp8 weights cost ~0.0001. Real activations/weights not yet used.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_C4_ragged.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/attention.py
- data: {"swa_prefill_pcc":0.99956,"ga_prefill_pcc":0.99961,"swa_decode_pcc":0.99939,"ga_decode_pcc":0.99937,"mesh":"4x8","ring_slots":1536,"chunk":1024}

## E261007-0159-07c
- date: 2026-10-07T01:59:21
- commit: 87e19f95595
- branch: ssinghal/mimo-attn
- host: 30
- track: attn
- status: verified
- claim: Attention module v1 UNOPTIMISED warm wall time per layer call on the 4x8 mesh (20 decode calls / 5 prefill chunks after warmup, no trace, host-dispatch inclusive, ctx 2048-4096, host .30, one process): decode 32 users/row 3.46 ms (GA) / 3.46 ms (SWA); prefill chunk 2048 tokens at start 2048: 3.06 ms (GA) / 2.70 ms (SWA). Decode is far above the DRAM roofline (~0.05 ms for 15 MB weights/chip) so it is dispatch/op-overhead bound (~40 small ops incl. embedding, repeat, to_memory_config, all_reduce): needs trace and op fusion in the next step. Not a throughput claim.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_C6_perf.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/attention.py
- data: {"decode_ms":3.46,"prefill_chunk2048_ms_ga":3.06,"prefill_chunk2048_ms_swa":2.70}

## E261007-0159-bd0
- date: 2026-10-07T01:59:21
- commit: 87e19f95595
- branch: ssinghal/mimo-attn
- host: 30
- track: skill-defect
- status: verified
- claim: Fixed per coordinator: tt-lab-ops sec.0 now says TT_METAL_CACHE defaults to /mnt/tt-data/ssinghal/tt-metal-cache/kit (shared, --cache-dir overrides) to match launch.py (text changed, tool unchanged). Merge of ssinghal/mimo into the attn branch had LOG.md and bringup_kit/.gitignore conflicts (both sides kept). Also learned: the black pre-commit hook reformats files so scripted multi-line text replacements on a file committed once fail silently (my ragged-test edit only half applied and produced a wrong PCC for ragged users; caught because the per-chunk and per-user numbers disagreed): re-read a file after the first commit before patching it.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_B2_ga.log
- paths: .claude/skills/tt-lab-ops/SKILL.md

## E261007-0247-7eb
- date: 2026-10-07T02:47:47
- commit: ce276b7d346
- branch: ssinghal/mimo-attn
- host: 30
- track: attn
- status: verified
- claim: REAL-WEIGHT REAL-ACTIVATION GATE vs track-0 bf16 dump (mimo-prefill-s128b16, B=16 S=128, 4 users/row on the 4x8 mesh, real L1 SWA + L5 GA weights via rank-blocked loader, bfp8 weights, teacher-forced dump attn_in as input): prefill attn_out PCC min 0.99973 (L1, per-row min 0.99973..0.99975) and 0.99997 (L5) vs bf16 dump, 0.99971 / 0.99997 vs the fp32 dump (bf16-dump-vs-fp32-dump floor 0.99996 / 0.99999); KV cache contents vs dump after device prefill: K PCC min 0.99999 (both), V (first 128 dims, 0.707 scaled) 0.99988 (L1) / 0.99996 (L5), V pad dims exactly 0, all 8 columns and 4 rows; 3 decode steps (dump dec_in -> host RMSNorm -> device decode, reference = track-0 attention() with dump KV as past, because the dump holds only the whole-layer dec_out) PCC min 0.99998 (L1) / 0.99996 (L5), columns identical. Gate >= 0.999 PASS for L1 and L5.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_I1_dump.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/attention.py
- data: {"L1_prefill_pcc":0.99973,"L5_prefill_pcc":0.99997,"L1_k_pcc":0.99999,"L5_k_pcc":0.99999,"L1_v_pcc":0.99988,"L5_v_pcc":0.99996,"L1_decode_pcc":0.99998,"L5_decode_pcc":0.99996}

## E261007-0247-1dc
- date: 2026-10-07T02:47:47
- commit: ce276b7d346
- branch: ssinghal/mimo-attn
- host: 30
- track: attn
- status: verified
- claim: Trace variant (chunk_start_idx_tensor, paged_fill_cache with device page-table slice, embedding-based RoPE, trace-captured chunk and decode, per-replay inputs via copy_host_to_device_tensor), random weights real shapes, 4x8 mesh: ragged prefill (chunk 1024, 2 users/row) PCC min 0.99955 (SWA) / 0.99961 (GA), decode 0.99945 / 0.99946; replays bit-identical for prefill chunk and decode, both layer types. Same at 8 users/row ragged (4096..1024 tokens, J3): prefill 0.99955/0.9996, decode 0.99942/0.99939; and 32 users/row decode (J2, chunk 256, lengths 256..1024): prefill 0.99955/0.99962, decode 0.9994/0.99937; all replays bit-identical, columns identical. Evidence also attn_J2_many32.jsonl, attn_J3_u8.jsonl.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_J1_trace_ragged.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/attention.py
- data: {"trace_prefill_swa":0.99955,"trace_prefill_ga":0.99961,"trace_decode_swa":0.99945,"trace_decode_ga":0.99946,"u8_decode_min":0.99939,"u32_decode_min":0.99937}

## E261007-0247-b41
- date: 2026-10-07T02:47:47
- commit: ce276b7d346
- branch: ssinghal/mimo-attn
- host: 30
- track: attn
- status: verified
- claim: Real layer-1 (SWA) / layer-5 (GA) weights (rank-blocked qkv dequant) with real text tokens (repo docs, 12.3k tokens) through the real embedding and input_layernorm, 4x8 mesh, traced, 3x1024 prefill + 2 decode steps vs torch fp32 HF-semantics reference: prefill PCC min 0.99998 (SWA) / 0.99988 (GA), decode 0.99996 / 0.99987, replays bit-identical. NOT a layout proof (device and torch use the same deinterleaved weights): see the separate layout-statistics entry.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_J4_real.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/attention.py
- data: {"swa_prefill":0.99998,"ga_prefill":0.99988,"swa_decode":0.99996,"ga_decode":0.99987}

## E261007-0247-b0d
- date: 2026-10-07T02:47:47
- commit: ce276b7d346
- branch: ssinghal/mimo-attn
- host: 30
- track: attn
- status: verified
- claim: 64k context on the 4x8 mesh, chunks of 2048 (traced), three users with TWO context lengths in one process (64k, 32k, 64k), sampled-row torch reference: GA prefill PCC per row 0.9999 (min chunk 0.99956) for both 64k users and 0.99989..0.9999 for 32k; chunk time (wall, blocking, replay only) GA first chunk 2.24 ms, median 9.2 ms, last (start 63488) 15.7 ms, 0.29 s per 64k per layer (x9 global layers = 2.6 s per user per row, attention only); SWA flat 2.15 ms/chunk, 0.07 s per 64k. DRAM allocated per bank after build 200.6 MB (GA, 32 slots x 65792 positions K+V bf16) / 19.5 MB (SWA ring), 204.7 MB after the first prefill (trace + io) and UNCHANGED after the 32k and second 64k users (no leak across context lengths); free ~3806 MB/bank. Decode at positions 65536 / 32768 / 65536 (slots 0..2, other slots at 0): PCC min 0.99933 (GA) / 0.9995 (SWA), replay bit-identical; traced decode 32 slots/row 1.93 ms (GA, only 3 slots long) / 0.72 ms (SWA) per layer. bf16 RoPE tables are sufficient at 64k (no PCC loss).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_J5_long64k.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/attention.py
- data: {"ga_prefill64k_pcc":0.9999,"ga_chunk_ms_last":15.7,"ga_prefill64k_s_per_layer":0.29,"ga_dram_alloc_mb_per_bank":204.7,"ga_decode64k_pcc":0.99933,"swa_decode64k_pcc":0.9995}

## E261007-0247-bfa
- date: 2026-10-07T02:47:48
- commit: ce276b7d346
- branch: ssinghal/mimo-attn
- host: 30
- track: attn
- status: inferred
- claim: qkv row-layout statistics on real tokens (CPU, 512 tokens, embedding+input_layernorm of layers 1 and 5, attention-structure measure): rank-blocked deinterleaved layout gives structured attention (mean max-prob 0.015/0.014, entropy 0.04/0.11 nats below uniform, max-head mass on offsets 0-3 0.043/0.031) while reading the rows as contiguous Q|K|V gives exactly uniform attention (max-prob 0.009/0.005, entropy = uniform). Consistent with the rank-blocked layout but weak evidence on its own (embedding inputs, not layer inputs); track 0 proved it against the HF forward (coordinator message), which supersedes this.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/attn_H2_layout.log
- paths: models/demos/blackhole/mimo_v26/tt/attention.py
## E261007-0222-e5a
- date: 2026-10-07T02:22:11
- commit: a367b32445b
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: decision
- status: verified
- claim: qkv layout PROVEN with the real HF MiMoV2DecoderLayer (fp32, real weights, 2x128 real GSM8K chat-template tokens): rank-blocked de-interleave gives sane layers (L0 out rms 0.0275 vs in 0.0107; L1 out 0.0284 vs in 0.0275, attention entropy/log n 0.79 = peaked, logit rms 27.9) while the naive contiguous Q|K|V reading blows the stream up (L0 out rms 0.284, L1 2.55 = 90x input) and gives near-uniform attention (L0 entropy 0.99999, logit rms 0.006). DECISION: loader.load_layer(qkv_layout='rank') is the default; every consumer must use per-TP4-rank [q_r|k_r|v_r] with scale blocks over each rank's rows.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_qkv_layout_proof.log
- paths: models/demos/blackhole/mimo_v26/reference/loader.py, models/demos/blackhole/mimo_v26/reference/hf_layer.py
- supersedes: E261007-0059-cb0
- data: {"L0_out_rms_rank":0.0275,"L0_out_rms_naive":0.2836,"L1_out_rms_rank":0.0284,"L1_out_rms_naive":2.554,"in_rms_L0":0.0107}

## E261007-0222-f3c
- date: 2026-10-07T02:22:11
- commit: a367b32445b
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: decision
- status: verified
- claim: fp4 experts (real L1 experts 17/100/200, 2048 real tokens): nibble order = LOW nibble first (DSV4 convention): act-norm ratio routed/unrouted tokens 1.56/1.52/1.74 with low-first vs 1.09/0.97/0.80 with high-first (an expert must respond more to tokens the router sends it). e8m0 scale byte b -> 2**(b-127) (torch float8_e8m0fnu view equals exactly; bytes 118..122 in expert 3 gate). Bias 127 gives expert gate rms 0.0215 (126: 0.043, 128: 0.0108; dense L0 gate rms 0.037) - bias is the standard e8m0 bias, magnitude check only plausibility, no independent bf16 ground truth exists. Per-block max |code| is 6 for 53% of blocks, >=4 for 100% (amax-based quantisation signature). No HF dequant path exists for fp4 (HF code sees dequantised weights), so the check is functional.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_fp4_order_check.log
- paths: models/demos/blackhole/mimo_v26/reference/loader.py, models/demos/blackhole/mimo_v26/reference/probes/fp4_order_check.py

## E261007-0222-c56
- date: 2026-10-07T02:22:11
- commit: a367b32445b
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: decision
- status: verified
- claim: fp32 independent oracle (ref_layer.py, written from the spec) vs HF MiMoV2DecoderLayer on real tokens (embeddings of 2x128 GSM8K prompts): L0 PCC 1.000000 maxabs 0; L1 (SWA+MoE) PCC 1.000000 rel 1.1e-8; L5 (global+MoE) PCC 1.000000 rel 5e-8; router expert SETS identical on all 256 tokens for L1 and L5. Mask: transformers (5.12.1) create_sliding_window_causal_mask == oracle mask (j<=i, j>i-128: window 128 INCLUDING self; row 200 sees 73..200 = 128 keys); global mask causal. Covered by this equality (same semantics as modeling_mimo_v2.py eager path): sink = raw per-head logit appended after the 192**-0.5 scaling, softmax over [scores, sink], sink column dropped; V*0.707 before cache/attention; RoPE on first 64 of 192 dims, split-half rotate_half, theta 1e7 global / 1e4 SWA; router fp32 sigmoid, selection on score+bias, weights from unbiased score renormalised, scale 1.0, no clamp. NOTE: the checkpoint's modeling file targets transformers 5.3 (calls create_causal_mask(input_embeds=, cache_position=)); this env has 5.12.1 (inputs_embeds, no cache_position) so the harness builds masks itself with the new API.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_oracle_vs_hf.log
- paths: models/demos/blackhole/mimo_v26/reference/ref_layer.py, models/demos/blackhole/mimo_v26/reference/hf_layer.py, models/demos/blackhole/mimo_v26/reference/probes/oracle_vs_hf.py

## E261007-0224-c48
- date: 2026-10-07T02:24:20
- commit: 0a673da0295
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: dump
- status: verified
- claim: FIRST DUMP (MoE agent can use it): /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16/ layers 0-5 (dense L0, SWA-MoE L1-4, global-MoE L5), bf16 chain, S=128 B=16 real GSM8K chat-template tokens (first 128 tokens of chat prompt + gold solution), prompt sha256 d2b8fd683e1e4e09c2f437e1e71b55a011f493d3bbfe57ba01ed40b2049e8f00 (tokens.pt also holds 3 teacher-forced decode tokens/user and texts). Files layer_{L}.pt = {prefill:{h_in,h_out,attn_in,attn_out,ffn_in,ffn_out [16,128,4096] bf16, routing_idx [2048,8] int64, routing_w [2048,8] fp32 (normalised unbiased sigmoid), route_gap [2048] fp32 = (score+bias) 8th-9th margin}, state:{global k [16,4,128,192] post-rope, v [16,4,128,128] already x0.707 | SWA k_ring/v_ring [16,8,128,*] slot p%128}, decode:[3 steps {dec_in,dec_out [16,1,4096], routing_idx, routing_w}], meta}. Layer L+1 h_in = layer L h_out (chain). CPU ~1-2 min/layer incl. weight load. Router near-ties are common: route_gap<0.01 for 85% of L5 tokens, so device-vs-oracle expert-set mismatch must be judged against the bf16-vs-fp32 floor (fp32 dump -f32 dir coming). Repro: nice python -m models.demos.blackhole.mimo_v26.reference.ref_prefill_dump --layers 0-5 --S 128 --B 16 --dtype bf16 --out <dir> (env MIMO_CKPT).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_dump_s128b16_bf16_L0-5.log
- paths: models/demos/blackhole/mimo_v26/reference
- data: {"S":128,"B":16,"layers":"0-5","dtype":"bf16","sha256":"d2b8fd683e1e4e09c2f437e1e71b55a011f493d3bbfe57ba01ed40b2049e8f00"}

## E261007-0228-172
- date: 2026-10-07T02:28:02
- commit: c488d1c5736
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: correctness
- status: verified
- claim: GATE (a) PASS for track 0 on real tokens (16x128 GSM8K chat prompts, sha d2b8fd68..): single-layer teacher-forced on the real bf16-chain h_in, fp32 oracle vs HF fp32: L0 (dense) PCC 0.99999994, L1 (SWA-MoE) 0.99999994, L5 (global-MoE) 0.99999994 (L2-4 1.0); router expert sets identical on 100% of the 2048 tokens for L1..L5. Chain L0..L5: HF chained vs fp32 oracle chain per-layer h_out PCC >= 0.9999999 each layer; logits after L0..L5 (norm+lm_head, prefix not language) PCC 1.000000 and greedy tokens equal on 100% of 2048 positions. bf16 oracle dump vs HF fp32: h_out PCC 0.99996-1.0 per layer, greedy-equal 98.3% (bf16 noise). Note PCC saturates because of residual outliers; use rel err (noise.pt) too.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_chain_vs_hf_L0-5.log
- paths: models/demos/blackhole/mimo_v26/reference
- data: {"L0_pcc":0.99999994,"L1_pcc":0.99999994,"L5_pcc":0.99999994,"router_sets_equal":1.0,"chain_logits_pcc":1.0,"chain_greedy_equal":1.0}

## E261007-0228-84b
- date: 2026-10-07T02:28:03
- commit: c488d1c5736
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: noise-floor
- status: verified
- claim: Reference noise floor, S=128 B=16 real tokens, L0-5, teacher-forced (fp32 layer on the bf16 dump's h_in vs the bf16 oracle layer): h_out relative L2 error 0.26% (L0), 0.31/0.63/0.43/0.36/0.27% (L1-L5); PCC 0.99998-1.0. Router: identical 8-expert set for 98.6/90.6/93.6/94.5/93.7% of tokens (L1-L5), per-slot overlap 98.8-99.8%. So ~6-9% of tokens flip >=1 expert between bf16 and fp32 purely from rounding; a device router mismatch below that is noise, not a bug. Dumps: /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16 (bf16 chain) and ...-f32 (fp32 own chain, noise.pt holds the numbers per layer).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_dump_s128b16_fp32_L0-5.log
- paths: models/demos/blackhole/mimo_v26/reference
- data: {"rel_h_out":[0.00259,0.00309,0.00630,0.00435,0.00356,0.00272],"router_set_equal":[null,0.986,0.906,0.936,0.945,0.937]}

## E261007-0242-981
- date: 2026-10-07T02:42:36
- commit: 0f46e26c5b9
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: dump
- status: verified
- claim: DUMP S=512 B=4 L0-5 (L5 = global path): bf16 chain /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s512b4, fp32 chain /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s512b4-f32 (noise.pt). Prompts: 4 disjoint 512-token windows of Frankenstein (Project Gutenberg, public domain, file sha256 7810cd48...fead3, read in place from the DSV4 tt_transformers context_cache) wrapped in the MiMo chat template with the DSV4 'quote and summarise' question; prompt sha256 c139b1779c3324c425106e6f5efa6580d97a0b5459a426016e75aadae5ecd3d5. Same file layout as the S=128 dump. Noise floor (teacher-forced fp32 vs bf16): h_out rel L2 0.26% L0, 0.41/0.58/0.42/0.38/0.40% L1-L5; identical expert sets 98.5/92.9/92.2/93.8/94.1% (slot overlap 99.0-99.8%). L5 route_gap<0.01 for 82% of tokens. Repro: ref_prefill_dump --layers 0-5 --prompts book --S 512 --B 4 --dtype bf16|fp32 [--tf_dir <bf16 dir>].
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_dump_s512b4_bf16_L0-5.log
- paths: models/demos/blackhole/mimo_v26/reference
- data: {"S":512,"B":4,"sha256":"c139b1779c3324c425106e6f5efa6580d97a0b5459a426016e75aadae5ecd3d5","rel_h_out":[0.00258,0.00413,0.00579,0.00422,0.00376,0.00401],"router_set_equal":[null,0.985,0.929,0.922,0.9375,0.941]}

## E261007-0242-e11
- date: 2026-10-07T02:42:36
- commit: 0f46e26c5b9
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: dump
- status: verified
- claim: DUMP S=2048 B=1 L0-5 bf16 chain: /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s2048b1 (no fp32 variant by instruction). Prompt: Frankenstein tokens 0.. in the chat template + quote/summarise question (same builder), sha256 98a359e229fb69d926a223193555d65cb82071f4591b0a1b4c164654df2d930c. CPU ~16 s/layer fwd + ~12 s weight load. Global L5: k [1,4,2048,192], v [1,4,2048,128]; SWA ring [1,8,128,*]. L5 route_gap<0.01 for 81%.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_dump_s2048b1_bf16_L0-5.log
- paths: models/demos/blackhole/mimo_v26/reference
- data: {"S":2048,"B":1,"sha256":"98a359e229fb69d926a223193555d65cb82071f4591b0a1b4c164654df2d930c"}

## E261007-0242-191
- date: 2026-10-07T02:42:36
- commit: 0f46e26c5b9
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: dump
- status: verified
- claim: DUMP S=8192 B=1 L0-5 bf16 chain: /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s8192b1 (2.3 GB), global path included (L5 k [1,4,8192,192]). Prompt: Frankenstein start in the chat template (exactly 8192 tokens), sha256 e38e6e111a3cb4d6a17a0c775c0d5995b31c84ba67788fb369faa4b01ec1313c. CPU ~15-20 s/layer. No fp32 variant. Caveat: SWA windows only see 128 keys, so this dump mainly exercises the global L0/L5 attention at long S and RoPE positions to 8191.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_dump_s8192b1_bf16_L0-5.log
- paths: models/demos/blackhole/mimo_v26/reference
- data: {"S":8192,"B":1,"sha256":"e38e6e111a3cb4d6a17a0c775c0d5995b31c84ba67788fb369faa4b01ec1313c"}

## E261007-0242-1c5
- date: 2026-10-07T02:42:36
- commit: 31782cc4ca0
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: tool
- status: verified
- claim: Teacher-forced single-layer oracle CLI: python -m models.demos.blackhole.mimo_v26.reference.single_layer --layer N --h-in <tensor|layer_*.pt|prefill dict> --out o.pt [--dtype bf16|fp32] [--dec-in dec.pt]; any layer 0-47, writes prefill{h_in..h_out, routing_idx/w/route_gap}, state (global k/v or SWA ring), decode steps. pytest tests/test_single_layer.py: reproduces the chain dump for L3 (h_out maxabs<1e-2), and runs finite with right shapes on a random real-scale input (per-channel mean/std of the real L5 output) for SWA L20, SWA L46, global L47 (4 passed, 51 s).
- evidence: models/demos/blackhole/mimo_v26/tests/test_single_layer.py
- paths: models/demos/blackhole/mimo_v26/reference/single_layer.py, models/demos/blackhole/mimo_v26/tests/test_single_layer.py

## E261007-0242-7cc
- date: 2026-10-07T02:42:37
- commit: 31782cc4ca0
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: dataset
- status: verified
- claim: GSM8K accuracy sets (no model run): /mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_128.json (sha256 of templated prompts 7bff43ed33869d3355d9a6a4bcc0387c80e5e2ef48d272fe6f91a9fc92394271, max 135 prompt tokens) and gsm8k_512.json (sha256 f5be16c2d438187236f472cdc36958cb8d2bbea75b97ff85e8e833c092b53edc, max 148 tokens): first 128/512 rows of /mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl (same rows/order as DSV4 gsm8k_o0..o448; the DSV4 128-set questions == rows 0-127, checked), MiMo chat template applied (add_generation_prompt, no system prompt), gold = normalised number after ####, prompt_ids stored. Scorer reference/score_gsm8k.py: extraction rule copied from DSV4 reference/e2e_analyze.py extract/norm (as used by tools/score_gsm.py and wt/h43c/score_gsm_big.py): last '#### x', else last \\boxed{x}, else last number; self-test: gold solutions score 512/512. Usage: score_gsm8k.py <set.json> <outputs.json> [--min-correct K]. Note the template ends the prompt at '<|im_start|>assistant\\n' (check whether thinking mode needs a prefix before the device runs).
- evidence: /mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_512.json
- paths: models/demos/blackhole/mimo_v26/reference/gsm8k_set.py, models/demos/blackhole/mimo_v26/reference/score_gsm8k.py
- data: {"sha256_128":"7bff43ed33869d3355d9a6a4bcc0387c80e5e2ef48d272fe6f91a9fc92394271","sha256_512":"f5be16c2d438187236f472cdc36958cb8d2bbea75b97ff85e8e833c092b53edc"}

## E261007-0242-e3b
- date: 2026-10-07T02:42:37
- commit: 31782cc4ca0
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: decision
- status: inferred
- claim: NO published GSM8K figure for MiMo-V2.6-Flash exists in the local sources: the technical report (44 pp.) benchmark tables are agentic (SWE/code/visual/cyber) and README has none, so the GSM8K gate threshold cannot be derived from the paper (coordinator to decide). MTP findings (reference/MTP_NOTES.md, page refs): report sec 2.4 (p.5-6, Table 1) 'MTP module' = the 5-layer DFlash drafter (block 8 = 1+7 tokens, window 1024), NOT the 3 in-checkpoint layers; p.32-33: RL rollouts use block-6 DFlash replacing 'MTP-3 inherited from SFT', accepted length +31.3% vs MTP-3; p.4: MTP is SWA + dense-FFN based. The report says nothing on eh_proj concat order, chaining, window or shared embed/head of MTP-3; no sglang source on disk (sgl.py is a 404 stub). Checkpoint facts measured: 3 layers, eh_proj [4096,8192], enorm/hnorm/input/pre_mlp/final norms, SWA-dim qkv fp8 + sink, dense 16384 FFN, no own embed/head. Everything structural is INFERRED (DeepSeek-V3-style).
- evidence: models/demos/blackhole/mimo_v26/reference/MTP_NOTES.md
- paths: models/demos/blackhole/mimo_v26/reference/MTP_NOTES.md

## E261007-0242-ebe
- date: 2026-10-07T02:42:50
- commit: 3b845b0f8c5
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: handoff
- status: verified
- claim: HANDOFF track 0 (ref). Worktree /mnt/tt-data/ssinghal/mimo/wt/ref, branch ssinghal/mimo-ref. Package models/demos/blackhole/mimo_v26/reference/: loader.py (load_layer, rank-blocked qkv, fp8/fp4 dequant), hf_layer.py (HF layer with real weights), ref_layer.py (independent oracle), ref_prefill_dump.py (chain dumps), single_layer.py (teacher-forced any-layer CLI), prompts.py, gsm8k_set.py + score_gsm8k.py, MTP_NOTES.md, probes/. DONE+logged: qkv per-TP4-rank layout proof, fp4 low-nibble-first + e8m0 bias 127, conventions == HF, gate (a) PCC 0.99999994 L0/L1/L5 + chain L0-5 + identical router sets, dumps S=128 b16 (bf16+fp32), S=512 b4 (bf16+fp32), S=2048 b1, S=8192 b1 (L0-5, dir /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s*), noise floor, GSM8K 128/512 sets + scorer, MTP notes. NOT done / unverified: decode-step dumps never compared to a device; no L6-47 dumps (policy: use single_layer on device-captured h); GSM8K threshold has no published number; MTP-3 structure inferred only; fp4 e8m0 bias magnitude only plausibility-checked; layers beyond L5 only exercised on random input (L20, L46, L47) not real hidden states; transformers 5.12 vs checkpoint's 5.3 mask API difference handled in harness; mask/HF equality checked only in the SDPA-free eager path. Composition: other tracks read dumps via torch.load(layer_L.pt)['prefill'|'state'|'decode']; ring slot p%128 for SWA; V in cache already x0.707; router weights are normalised unbiased sigmoid. Skill defects: see SKILL_DEFECTS.md items 4-7 (kit pycache tracked, host list stale, sgl.py 404 + no checkpoint, skill lacked local-weights step -> fixed in tt-model-reference).
- evidence: bringup/mimo_v26/SKILL_DEFECTS.md
- paths: models/demos/blackhole/mimo_v26, bringup/mimo_v26/SKILL_DEFECTS.md

## E261007-0253-fcc
- date: 2026-10-07T02:53:39
- commit: db7dbcd991c
- branch: ssinghal/mimo
- host: 30
- track: attn
- status: verified
- claim: MERGED-COMMIT re-check of attention gate (merge protocol step 5): test_attention_dump.py ATTN_LAYERS=1,5 on db7dbcd991c, host .30, 4x8 mesh, bfp8 weights, teacher forced vs track-0 dump mimo-prefill-s128b16 (B=16, S=128, 4 users/row). Prefill attn_out PCC min L1 (SWA) 0.99973, L5 (GA) 0.99997 (>=0.999 gate); K cache 0.99999/0.99999, V cache 0.99988/0.99996; 3 decode steps min 0.99998/0.99996; V-pad dims exactly 0; column spread 0. Same values as branch run.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/integ_attn_dump_db7dbcd991c.log
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests
- data: {"layer_type":"swa_moe+global_moe attention","pcc":0.99973,"pcc_swa":0.99973,"pcc_ga":0.99997,"kv_pcc_min":0.99988,"decode_pcc_min":0.99996}

## E261007-0318-bfb
- date: 2026-10-07T03:18:29
- commit: 76fba8807cb
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: USER DECISION 2026-10-07: keep expert activations bf16 (layer-5 MoE PCC 0.9915-0.9949 with the unified op's internal bfp8 activations, E261007-0251-c0a / -559). Weights stay bfp8. Next: bf16-activation expert path with existing ops, no C++; any C++ patch still needs separate approval.
- evidence: -
- paths: bringup/mimo_v26/scorecard.json

## E261007-0326-160
- date: 2026-10-07T03:26:38
- commit: 4e7bca60e3d
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: USER DECISION 2026-10-07 (/goal): agents may build C++ autonomously; use any free host on the cluster, skip hosts that are busy or bad; aim is end-to-end prefill + decode + spec decode on MiMo-V2.6-Flash. Coordinator reading: C++ builds go in PRIVATE build dirs per worktree (never the shared build_Release, never the DeepSeek repo); still NOT authorised: bfp4/fp8-KV, pushing, deleting data, touching or killing foreign jobs, gate weakening.
- evidence: -
- paths: bringup/mimo_v26/scorecard.json

## E261007-0346-6df
- date: 2026-10-07T03:46:23
- commit: a150b001c4b
- branch: ssinghal/mimo-asm
- host: 35
- track: asm
- status: verified
- claim: First on-device decoder layers L0+L1 (dense SwiGLU + SWA/MoE) on 4x8 vs track-0 dump mimo-prefill-s128b16 (B=16,S=128, 4 users/row), real bfp8 weights, expert op mm (bf16 activations, fp32-acc matmuls), exact router. TEACHER FORCED prefill (min over 16 users) L0: attn_out 0.99997, ffn_out 0.99995, h_out 0.99998 vs bf16 dump. L1: attn_out 0.99968, ffn_out 0.99512 (bf16/fp32 dump floor 0.99739; on tokens whose routed expert SET matches the dump 0.99996), h_out 0.99993 (floor 0.99995); router set match 0.95996 (82 flips/2048 tokens, all 82 with gap<0.01 near-ties; track-0 floor 90.6-98.6%). Teacher-forced decode (3 steps, dump dec_in/dec_out, 32 slots/row, MoE decode padded to 64 rows) min PCC L0 0.99997, L1 0.99981. Free-running chain with device embedding (embedding PCC 1.0): h_out vs bf16 dump L0 0.99998, L1 0.99989 (vs fp32 0.9999, floor 0.99995). Weight convert+cache first build: L0 5.5 s, L1 93.6 s (27 tensors); 2-layer test 374 s total.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_chain2_a150b001c4b.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/layer.py, models/demos/blackhole/mimo_v26/tt/model.py, models/demos/blackhole/mimo_v26/tt/weights.py, models/demos/blackhole/mimo_v26/tests/test_chain_dump.py
- data: {"l0_h_out":0.99998,"l1_ffn_out_min":0.99512,"l1_set_match":0.95996,"l1_h_out_tf":0.99993,"chain_l1_vs_bf16":0.99989}
## E261007-0346-4c1
- date: 2026-10-07T03:46:11
- commit: a9f8e9910cd
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: bf16-activation expert path 'mm' (NO C++): dispatch(bf16 RM) -> tilize bf16 -> per local expert extract -> ttnn.matmul gate/up (bfp8 weights DRAM-interleaved, HiFi2 fp32 acc, bf16 out) -> silu*up -> down matmul (M chunked 256) -> insert -> combine -> post_combine_reduce; exact router. Real weights, real dump tokens, 4x8, PCC vs bf16 dump ffn_out per mesh row: L5 dump routing 0.99986-0.99991, exact router (set match 0.9888) 0.99978-0.99990 (unified op: 0.9915-0.9949); L1 dump routing 0.99997, exact router 0.99982-0.99994; 5 runs bit-identical; no dropped tokens (worst chip load L1 1856, L5 896 of max_buf 8416 at cf4; cf=1 also fits at N=512/row). N=2048/row (real tokens repeated 4x, L5): same PCC, worst load 3264 rows vs max_buf 8416 (cf1) / 32992 (cf4: 258 MB bf16 dispatch buffer/chip), max_per_expert 8192. Gate >=0.999 met for L1 and L5. routed_expert_ffn (no C++) cannot do fp32 acc (fixed blocking fails: out_subblock_w 6 > 4); with bf16 DEST it gives 0.9986-0.9991 (L5, below gate)
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_C_L5_mm_unified_a9f8e9910cd.log
- paths: models/demos/blackhole/mimo_v26

## E261007-0346-36b
- date: 2026-10-07T03:46:11
- commit: a9f8e9910cd
- branch: ssinghal/mimo-moe
- host: 40
- track: moe
- status: verified
- claim: mm path timing hint (paired alternating one process, host-timed, N=512/row incl dispatch+combine): mm 30 ms vs unified 2.3-2.8 ms per layer (13x slower: extract/FFN run on max_per_expert=2048 padded rows for each of 8 experts; real counts are ~64 avg). N=64/row decode-sized (32 slots padded to 64, ragged live 32/20/7/1): mm 7.8 ms vs unified 2.1 ms, mm PCC 0.99989-0.99998 vs unified 0.9917-0.9997 at L5. Decode-sized finding: dead pad tokens routed to a fixed expert set overflowed max_buf (worst chip load 1280 of 1248 at N=64 cf=4) -> pad tokens must be spread/masked (padding_config) or cf raised. Weights: one DRAM-interleaved bfp8 per-expert copy (analytic 214 MB/chip/layer, ~10 GB at 47 layers) works for mm, loop, unified and fused ops, so prefill and decode via these ops can share it; moe_compute ring layout still needs the C++ patch
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_C_L5_decode_a9f8e9910cd.log
- paths: models/demos/blackhole/mimo_v26

## E261007-0401-3fc
- date: 2026-10-07T04:01:16
- commit: 28b183e4d8b
- branch: ssinghal/mimo-asm
- host: 35
- track: asm
- status: verified
- claim: L0..L5 chain on 4x8 vs dump mimo-prefill-s128b16 (B=16,S=128), real bfp8 weights, mm experts (bf16 act), exact router, device embedding (PCC 1.0). FREE-RUNNING h_out PCC (min over 16 users) vs bf16 dump / fp32 dump / bf16-vs-fp32 floor: L0 0.99998/0.99998/1.0; L1 0.99989/0.9999/0.99995; L2 0.99986/0.99987/0.99994; L3 0.99986/0.99987/0.99994; L4 0.99989/0.99989/0.99995; L5 0.99977/0.99978/0.99994 (L5 vs fp32 mean 0.9998). TEACHER-FORCED per layer h_out min PCC vs bf16 dump: 0.99998,0.99993,0.99992,0.99995,0.99995,0.9999 (floor 1.0..0.99994); attn_out min 0.99997,0.99968,0.9991,0.99956,0.99967,0.99997; ffn_out min 0.99995,0.99512,0.99771,0.9978,0.99112,0.92992 (floor 1.0,0.99739,0.99826,0.99627,0.96141,0.99497): L5 ffn_out min user 0.92992 is BELOW floor, and router expert-SET match vs dump routing (device router on device ffn_in) 0.960,0.723,0.763,0.782,0.823 for L1..L5 (flips 82,568,485,447,363 of 2048, ~all with gap<0.01), ffn_out PCC on set-matched tokens 0.99996,0.99974,0.99984,0.99929,0.99943. Decode teacher-forced 3 steps min PCC L0 0.99997, L1 0.99981, L2 0.99893, L3 0.99992, L4 0.99996, L5 0.9874 (step 2). Cache hit path: 123 tensors loaded in 144 s for 6 layers (cache built by 3 hosts in parallel, ~80-110 s per MoE layer). Note the set-match is lower than the track-0 floor (90.6-98.6%) because the device router here sees device-computed ffn_in; a follow-up run feeds the dump ffn_in to separate router from upstream noise.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_chain6_28b183e4d8b.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/layer.py, models/demos/blackhole/mimo_v26/tt/model.py, models/demos/blackhole/mimo_v26/tt/weights.py
- data: {"free_chain_l5_vs_bf16":0.99977,"tf_l5_ffn_out_min":0.92992,"tf_l5_dec_min":0.9874}

## E261007-0408-758
- date: 2026-10-07T04:08:31
- commit: fd73ec06bc7
- branch: ssinghal/mimo-asm
- host: 35
- track: asm
- status: verified
- claim: FFN-only teacher forcing L0..L5 (dump ffn_in -> device exact router + mm bf16-activation experts + column all-reduce), min PCC over 16 users of ffn_out: vs bf16 dump L0 0.99996, L1 0.99936, L2 0.99984, L3 0.99922, L4 0.99849, L5 0.99965; vs fp32 dump = bf16/fp32 floor in every layer (L1 0.99739/0.99739, L2 0.9982/0.99826, L3 0.99625/0.99627, L4 0.96018/0.96141, L5 0.99478/0.99497). Device router on the dump ffn_in: expert-SET match 0.99365,0.98828,0.99121,0.99463,0.98877 for L1..L5 (13,24,18,11,23 flips of 2048, ALL gap<0.01 near-ties). So the low set-match in the chain run (0.72-0.82, E261007-0401-3fc) comes from upstream device attention noise in ffn_in, not from the router/FFN. Expert op mm, cf 4, 4 users x128 tokens per mesh row.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_chain6b_1522188db07.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/layer.py
- data: {"l5_ffn_vs_bf16":0.99965,"l4_ffn_vs_bf16":0.99849,"setmatch_l1_l5":[0.99365,0.98828,0.99121,0.99463,0.98877]}

## E261007-0414-cd3
- date: 2026-10-07T04:14:54
- commit: 81eee22f9d0
- branch: ssinghal/mimo-asm
- host: 35
- track: asm
- status: verified
- claim: 4-layer smoke (L0..L3 + final norm + lm_head bf16 vocab-sharded) on 4x8, one process, mm experts: (1) dump tokens (16 users x128): device last-token logits vs CPU norm+lm_head of dump L3 h_out: PCC min 0.9998 (mean 0.99992), top-1 agreement 16/16. (2) real GSM8K prompts (chat template, 4 slots x 4 rows with equal lengths per slot, right-padded with pad id): first token + 1 decode step at position len run for TWO ISLs in the same process (S=128 lens 102/101/100/95, then S=256 lens 94/92/90/89), logits finite; 4-layer model emits the same junk token for every prompt (141806, then 143877), as expected for a truncated model (no accuracy claim). Decode path includes MoE decode padded to 64 rows with a per-layer capacity assertion (passed).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_smoke4_81eee22f9d0.jsonl
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests/test_smoke_model.py
- data: {"logits_pcc_min":0.9998,"top1_agree":1.0}
## E261007-0354-e2f
- date: 2026-10-07T03:54:31
- commit: 4e4e06559c5
- branch: ssinghal/mimo
- host: 35
- track: moe
- status: verified
- claim: MERGED-COMMIT re-check of MoE gate: test_moe_real_layer.py layer 5, MIMO_UM_OPS=mm,unified, on 4e4e06559c5, host .35, 4x8, 256 real experts bfp8, N=512 tok/row, vs track-0 dump mimo-prefill-s128b16. op=mm (bf16 activations, HiFi2 fp32acc): PCC vs bf16 dump ffn_out per row with dump routing 0.99986-0.99991, with exact router (set match 0.9888) 0.99978-0.9999, 5 runs bit-identical, no dropped tokens (worst chip 896 of 8416 rows, cf=4). op=unified: 0.99151-0.99491 (FAILS 0.999). bf16-vs-fp32 floor 0.9985-0.99911. Time hint (host-timed, paired, not a perf claim): mm 30.7 ms vs unified 2.3 ms per layer. The pytest exit code is 1 only because the test asserts unified too (test design defect, unified is expected to fail at L5).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/integ_moe_L5_35_4e4e06559c5.log
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests
- data: {"layer":5,"pcc_mm_min":0.99978,"pcc_unified_min":0.99151,"bit_identical_runs":5,"dropped_tokens":0}
## E261007-0430-3d4
- date: 2026-10-07T04:30:34
- commit: 6be783ac580
- branch: ssinghal/mimo-moe
- host: 35
- track: moe
- status: verified
- claim: A-perf C++ patch (commit 1260bf4b1f6, private relink build via bringup/mimo_v26/tools/priv_relink.py: recompiles only the op unity TU with the shared build's flags, relinks a private _ttnncpp.so, 20 s; shared build untouched; kernels JIT from worktree): env switches in unified_routed_expert_moe MOE_UNI_BF16_X (cb_in0_x Float16_b), MOE_UNI_BF16_INTERMED (gate/up/activated/down_full Float16_b), MOE_UNI_BF16_OUT (expert output tensor bf16), MOE_UNI_FIDELITY (default LoFi). Default (no env) bit-unchanged: L5 unified 0.99491-0.99312-0.99153-0.99324 as before. L5 real weights/tokens 4x8 PCC vs bf16 dump, dump routing: default 0.9915-0.9949; X only 0.9943-0.9967 (LoFi) / 0.9939-0.9965 (HiFi2, fidelity irrelevant); all three bf16 (LoFi) 0.99946-0.99959; all bf16 + HiFi2 0.99953-0.99970. With exact router 0.9994-0.9996. L1 0.99994 (exact router 0.99979-0.99991). 5 runs bit-identical, no dropped tokens. Gate >=0.999 met at unified-op speed: paired one process N=512/row host-timed unified(all-bf16) 2.3 ms (L5) / 2.8 ms (L1) vs mm 30 ms; default-bfp8 unified earlier 2.3/2.8 ms (separate process, hint) -> no measurable slowdown. N=2048/row L5: same PCC, worst load 3264 rows, max_buf 8416 (cf1) .. 32992 (cf4, 258 MB bf16 dispatch buffer per chip). Artifact: /mnt/tt-data/ssinghal/mimo/artifacts/moe_A_1260bf4b1f6/lib/_ttnncpp.so (run with LD_LIBRARY_PATH=<dir>/lib)
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_D_L5_bf16uni_h35_6be783ac580.log
- paths: models/demos/blackhole/mimo_v26, ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn

## E261007-0430-08b
- date: 2026-10-07T04:30:35
- commit: 6be783ac580
- branch: ssinghal/mimo-moe
- host: 35
- track: moe
- status: verified
- claim: Decode-sized (32 slots/row padded to 64 tokens, ragged live 32/20/7/1, L5 real weights, all-bf16 unified): PCC vs bf16 dump live tokens 0.99949-0.99995 (mm 0.99989-0.99998), 5 runs identical. Dead pad slots now routed round-robin over all experts (spread_pad_routing in tt/moe_prefill.py): worst chip load 256 rows vs max_buf 1248 (cf4) / 480 (cf1); the test asserts on capacity overflow. Timing hint N=64/row host-timed paired: unified 1.9 ms, mm 5.0 ms per layer
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_D_L5_decode_bf16uni_h35_6be783ac580.log
- paths: models/demos/blackhole/mimo_v26, ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn

## E261007-0438-cd8
- date: 2026-10-07T04:38:55
- commit: 23b56c47ac7
- branch: ssinghal/mimo
- host: 30
- track: moe
- status: verified
- claim: MERGED-COMMIT re-check, unified_routed_expert_moe with bf16 x/intermediates/output (MOE_UNI_BF16_X/INTERMED/OUT=1) via private relink lib artifacts/integ_23b56c47ac7/lib (LD_LIBRARY_PATH), host .30, 4x8, 256 real experts bfp8, N=512 tok/row, layer 5 vs track-0 dump: PCC vs bf16 dump ffn_out dump-routing 0.99946-0.99959 per row, exact router (set match 0.9888) 0.9994-0.99959, 5 runs bit-identical, no dropped tokens (worst chip 896/8416 rows). bf16-vs-fp32 floor 0.9985-0.99911. Gate >=0.999 PASS.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/integ_moebf16uni_L5_h30_23b56c47ac7.log
- paths: models/demos/blackhole/mimo_v26/tt, ttnn/cpp/ttnn/operations/experimental/deepseek_prefill
- data: {"layer":5,"pcc":0.9994,"bit_identical_runs":5}

## E261007-0438-ff0
- date: 2026-10-07T04:38:56
- commit: 23b56c47ac7
- branch: ssinghal/mimo
- host: 41
- track: moe
- status: verified
- claim: MERGED-COMMIT re-check, same config as layer 5 entry, host .41, layer 1: PCC vs bf16 dump ffn_out dump-routing 0.99994 all rows, exact router (set match 0.9937) 0.99979-0.99991, 5 runs bit-identical, no dropped tokens (worst chip 1856/8416 rows). Gate >=0.999 PASS.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/integ_moebf16uni_L1_h41_23b56c47ac7.log
- paths: models/demos/blackhole/mimo_v26/tt, ttnn/cpp/ttnn/operations/experimental/deepseek_prefill
- data: {"layer":1,"pcc":0.99979,"bit_identical_runs":5}

## E261007-0456-8fe
- date: 2026-10-07T04:56:09
- commit: 43f8c5f81d5
- branch: ssinghal/mimo-asm
- host: 35
- track: asm
- status: verified
- claim: L0..L5 chain re-run with the FAST bf16-activation unified expert op (expert_op=unified, MOE_UNI_BF16_X/INTERMED/OUT=1, private relink lib integ_23b56c47ac7), same test as E261007-0401-3fc: free-running h_out PCC min vs bf16 dump L0..L5 = 0.99998,0.99989,0.99984,0.99984,0.99986,0.99973 (vs fp32 same to 5 digits; floor 1.0,0.99995,0.99994,0.99994,0.99995,0.99994). FFN-only teacher forcing (dump ffn_in) ffn_out vs bf16 dump L1..L5 0.99932,0.9998,0.99917,0.99849,0.99926 (L5 0.99926 vs 0.99965 with mm); vs fp32 = floor-level in all layers (L5 0.99461/0.99497). Teacher-forced decode min over 3 steps L0..L5: 0.99997,0.99981,0.99893,0.99992,0.99996,0.98723 (L5 step 2; same low value as mm 0.9874, 4 live tokens per row, near-tie routing). Router set match unchanged (0.96,0.72,0.76,0.78,0.82 in-chain; 0.994,0.988,0.991,0.995,0.989 on dump ffn_in). Whole 6-layer test (weights from cache 2.8 s, TF prefill+ffn-only+decode+free chain) 41 s. L1-exhaustion fix included: per-gate L1 buffers freed in exact mode.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_chain6u_43f8c5f81d5.jsonl
- paths: models/demos/blackhole/mimo_v26/tt/layer.py, models/demos/blackhole/mimo_v26/tt/model.py
- supersedes: E261007-0401-3fc
- data: {"free_chain_l5_vs_bf16":0.99973,"ffn_only_l5":0.99926,"dec_l5_min":0.98723}

## E261007-0458-0a3
- date: 2026-10-07T04:58:52
- commit: 43f8c5f81d5
- branch: ssinghal/mimo-asm
- host: 48
- track: asm
- status: verified
- claim: FULL 48-LAYER MODEL first token + 1 decode step on real GSM8K prompts (chat template), 4x8 mesh, unified bf16-activation experts, bfp8 weights (bf16 embedding/lm_head), exact router, ONE process, TWO ISLs (S=128 lens 102/101/100/95 then S=256 lens 94/92/90/89; 16 users each; prompts right-padded with pad id): all 32 first tokens are '<think>' (id 151667) and the decode-step tokens are coherent reasoning openers ('Let', 'First', 'She', 'Total', 'Steve', 'Monday', ...), all logits finite. Weights: 1131 cached tensors loaded in 69.7 s on a warm cache (938 s cold NFS); test 187 s after load. mm expert op on the same 48-layer build also loaded (L1 of the first prefill hit an L1 clash from per-gate L1 buffers, fixed in 43f8c5f81d5). No accuracy claim (no GSM8K scoring yet; no CPU full-model reference by policy).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_smoke48u_43f8c5f81d5.jsonl
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests/test_smoke_model.py
- data: {"layers":48,"first_token":"<think>","isl":[128,256]}

## E261007-0459-e2c
- date: 2026-10-07T04:59:12
- commit: 43f8c5f81d5
- branch: ssinghal/mimo-asm
- host: 35
- track: skill-defect
- status: verified
- claim: Defects found in assembly: (1) repo .gitignore has 'build_*': a new tt/build_wcache.py was silently untracked (renamed cache_weights.py); kits should warn on gitignored names. (2) host_health.py exits 0 for BUSY and launch.py then queues on the host flock (my chain waited behind the moe job): launch recipe must gate on the verdict. (3) TTMoEGate allocates 4 L1-sharded kernel buffers per instance; 47 layers exhaust L1 and the first rms_norm fails ('static dataflow buffers clash with L1 buffers') at 48 layers though 6 is fine: freed in exact mode in layer.py. (4) One 48-layer launch on host .46 sat 18 min at ~2500% CPU after layer 0 with no progress (killed, own pid); same run on .48 progressed at 25 s/layer: single occurrence, no bad-host claim. (5) launch.py unset-prefix does not unset MOE_UNI_*: pass them explicitly. (6) tests need mesh_device fixtures with FABRIC_1D_RING + l1_small_size 16384.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_smoke48b_4bb27d3fd91.log
- paths: models/common/bringup_kit/launch.py, models/demos/blackhole/mimo_v26/tt/layer.py

## E261007-0459-4db
- date: 2026-10-07T04:59:12
- commit: 43f8c5f81d5
- branch: ssinghal/mimo-asm
- host: 35
- track: handoff
- status: verified
- claim: HANDOFF track 3 (asm), worktree /mnt/tt-data/ssinghal/mimo/wt/asm branch ssinghal/mimo-asm. Code: tt/layer.py (MiMoDecoderLayer prefill/decode, dense L0 + MoE, capacity assertion), tt/model.py (embedding, 48 layers, final norm, lm_head vocab-sharded bf16, prefill_first_token, decode_step), tt/weights.py (flatbuffer cache of sharded bfp8 tensors under /mnt/tt-data/ssinghal/mimo-wcache/v1, ~6.5 GB/MoE layer, all L0-47 built), tt/cache_weights.py (parallel cache builder), tests/test_chain_dump.py, tests/test_smoke_model.py. Evidence: chain L0..L5 free-running h_out PCC >=0.99973 vs bf16 dump (E261007-0456-8fe), FFN-only teacher forcing at floor, 48-layer first token '<think>' on GSM8K prompts for two ISLs (see 48-layer smoke entry). Repro: launch.py --worktree WT --commit C --host H --env MIMO_CKPT=/mnt/tt-data/MiMo-V2.6-Flash-RL --env MOE_UNI_BF16_X=1 --env MOE_UNI_BF16_INTERMED=1 --env MOE_UNI_BF16_OUT=1 --env LD_LIBRARY_PATH=/mnt/tt-data/ssinghal/mimo/artifacts/integ_23b56c47ac7/lib --env MIMO_EXPERT_OP=unified --env SMOKE_LAYERS=48 -- python_env/bin/python -m pytest -x -s -q models/demos/blackhole/mimo_v26/tests/test_smoke_model.py (chain: test_chain_dump.py, CHAIN_LAYERS=6 CHAIN_DECODE=1). NOT done/unverified: no GSM8K accuracy, no traced/chunked/paged prefill or decode loop, no timing (host-timed only), decode dead slots not spread-routed (capacity assertion passes at 4 live/row; spread_pad_routing not wired), decode teacher-forced L5 step 2 min PCC 0.987 (near-tie routing, 4 live tokens), layers 6-47 verified only by finite/plausible tokens not by PCC, per-slot decode positions are shared by all mesh rows (attention decode_set takes one vector), max_pos 512, bf16 lm_head/embedding replicated, MTP not built. Merge notes: ssinghal/mimo merged twice into this branch (LOG.md conflicts resolved by keeping both).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_smoke48u_43f8c5f81d5.jsonl
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests

## E261007-0509-4c0
- date: 2026-10-07T05:09:36
- commit: d699f503706
- branch: ssinghal/mimo-asm
- host: 48
- track: asm
- status: verified
- claim: DEVICE per-layer dump EXISTS: /mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16/ (layer_0..47.pt + logits.pt, 4.85 GB). Free-running 48-layer device prefill (unified bf16-activation experts, bfp8 weights), same 16 GSM8K sequences/tokens as mimo-prefill-s128b16 (B=16, S=128, user b = mesh row b//4, slot b%4). layer_L.pt = {'prefill': {h_in, attn_in, attn_out, ffn_in, ffn_out, h_out [16,128,4096] bf16 (h_in = DEVICE h_out of layer L-1, i.e. chained), routing_idx [2048,8] int64 and routing_w [2048,8] float32 for L>=1 (device router, token order b*128+s), no route_gap/state/decode}, 'meta'}; logits.pt = {logits [16,152576] float32 of position 127, norm_in [16,4096] bf16 (last-token hidden before final norm), prefill_tokens}. Sanity: all fields finite at L47; h_out PCC vs CPU bf16 dump L0..L5 0.99998,0.99994,0.99989,0.99987,0.99988,0.99976 (whole-tensor). Last-token argmax decodes to plausible continuations of the dump texts ('The', ',', ' day', ' cheaper', ...). Capture run takes ~4 min incl. weight load. Repro: CAP_OUT=<dir> pytest tests/test_capture_device.py with the unified env (see handoff).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_capture48_d699f503706.log
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests/test_capture_device.py
## E261007-0502-8d6
- date: 2026-10-07T05:02:47
- commit: 4169b64e156
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: DECISIONS (coordinator, 2026-10-07, at 4169b64e156): (1) GSM8K gate protocol uses chat template with enable_thinking=false ('<think></think>' after the assistant tag, chat_template.jinja line 93); thinking-mode accuracy not measured. (2) File ownership for the next stage: tt/attention.py decode + tt/decode.py + tt/generator.py = attn agent; tt/prefill.py = asm agent; tt/moe*.py router*.py = moe agent; tt/kv_pool.py shared, layout agreed via LOG/notes; tt/layer.py tt/model.py changes via coordinator. (3) Per-layer gate for layers 6..47: reference agent runs the CPU oracle teacher-forced on DEVICE-captured per-layer inputs (mimo-device-s128b16), not a full CPU model run. (4) Spec decode drafter choice (MTP-3 vs DFlash) pending user decision; MTP oracle hypotheses built on device-captured hidden states first.
- evidence: -
- paths: bringup/mimo_v26/scorecard.json

## E261007-0514-f37
- date: 2026-10-07T05:14:25
- commit: 0972f38a750
- branch: ssinghal/mimo-asm
- host: DC16-2-BG-0204-u20-44
- track: asm
- status: inferred
- claim: KV POOL LAYOUT PROPOSAL for the decode agent (not yet exercised on device): see bringup/mimo_v26/notes/kv_pool.md and tt/kv_pool.py (KVLayout). Per-layer per-device [nblocks,1,64,192] bf16 K and V (V padded), slots 0..U-1 own blocks, slots U..31 alias a scratch region; GA table [32, max_pos/64] contiguous per slot; SWA ring R = CH/64+8 blocks, logical block i -> ring block i%R, decode modulo R*64; one table per layer KIND shared by all layers; per-row lengths. 64k DRAM per chip for all caches: U=4 2.06 GiB, U=8 4.03 GiB (arithmetic, not measured).
- evidence: -
- paths: bringup/mimo_v26/notes/kv_pool.md, models/demos/blackhole/mimo_v26/tt/kv_pool.py

## E261007-0524-d40
- date: 2026-10-07T05:24:24
- commit: afd3e45361a
- branch: ssinghal/mimo-asm
- host: 35,48,43
- track: asm
- status: verified
- claim: FIRST TRACED CHUNKED PREFILL (tt/prefill.py ChunkedPrefill: whole-model chunk forward captured in ONE trace, row-sharded device inputs rope_idx/fill_pt/sdpa_pt/chunk_start/tokens, paged KV pool of tt/kv_pool.py with scratch slot for idle rows), L0..L5, unified bf16 experts, vs track-0 CPU dumps; replay bit-identical to the eager-warmup+trace run in all 3 runs. (a) s2048b1 CH=2048 (row 0 only, other rows idle->scratch): hidden after L5 at all 2048 positions PCC 0.99972, first-token logits PCC 0.99945 top1 1/1; KV vs dump K min 0.99974 (GA L5)/0.99993+ others, V[:128] min 0.9993 (GA L5), V pad dims exactly 0; hand-off (per layer decode of 3 dump steps reading the pool) min PCC 0.99986 L0-L5; chunk step 55.7 ms (6 layers, 2048 tok/row, replay incl. host input copy). (b) s512b4 CH=256 (2 chunks/user, 1 user per row, 4 rows): hidden L5 0.99931, logits PCC 0.99817 top1 3/4 (near-tie), KV K min 0.99963 V min 0.99915, hand-off decode min 0.99959 L2 / 0.99991 L5, step 12.6-12.9 ms. (c) s128b16 CH=128 (4 chunks, 4 users/row): hidden L5 0.99974, logits 0.99977 top1 16/16, KV K min 0.99953 V 0.9991, hand-off decode min 0.99893 (L2 s0), L5 step2 0.98713 (same near-tie value as chain test), step 11.0 ms. Ring for SWA = CH/64+8 blocks, decode modulo R*64 works with chunked-prefill-written rings. Pad tokens of ragged last chunks repeat the prompt cyclically. NOT yet: ragged multi-user rows (>1 user per row per chunk queue exercised only in (c)), 8192/64k, 48 layers, DRAM gate, two ISLs in one process.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_pf_s2048b1_ch2048_afd3e45361a.jsonl
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests/test_prefill_handoff.py
- data: {"s2048_hidden_l5":0.99972,"step_ms_6L_2048":55.7,"handoff_min":0.99959}

## E261007-0533-cdf
- date: 2026-10-07T05:33:20
- commit: a43e4229c9f
- branch: ssinghal/mimo-asm
- host: 35,43,47
- track: asm
- status: verified
- claim: Traced chunked prefill L0..L5 (unified bf16 experts) vs dumps, more cases at a43e4229c9f: (1) RAGGED MULTI-USER ROWS: s128b16 truncated to lengths 128/100/64/33 per slot (4 users per mesh row, 4 steps CH=128, pad tokens cyclic): hidden after L5 at all real positions PCC min 0.9996 (per user 0.9996-0.99977), logits PCC 0.9997 top1 15/16, KV GA K min 0.99959 V 0.99908 (SWA KV compared only where the dump ring covers the truncated positions), replay bit-identical; decode hand-off skipped (ragged lengths need per-row decode positions). (2) SEVERAL CHUNKS PER USER: s2048b1 with CH=512 (4 chunks): hidden L5 0.99972, logits 0.99945, KV K min 0.99974 V 0.9993, hand-off decode min 0.99986, step 20.0-20.9 ms (512 tok/row); (3) s8192b1 CH=2048 (4 chunks, GA reads the growing paged cache): hidden L5 0.99977, logits 0.99962 top1 1/1, KV K 0.99977 V 0.9993, hand-off decode min 0.99983, step 55.9-58.2 ms per 2048-token chunk (6 layers, row 0 busy, others on scratch). All replays bit-identical.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_pf_s8192_ch2048_a43e4229c9f.jsonl
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests/test_prefill_handoff.py
- data: {"ragged_hidden_min":0.9996,"s8192_hidden":0.99977}

## E261007-0533-fb3
- date: 2026-10-07T05:33:53
- commit: a43e4229c9f
- branch: ssinghal/mimo-asm
- host: 48
- track: asm
- status: verified
- claim: 4-LAYER (L0-L3, 1 GA + 3 SWA) chunked traced prefill at scale, real GSM8K text concatenated to the ISL, 4 users (1 per mesh row), CH=2048, unified bf16 experts, TWO ISLs in ONE process 64k -> 8k -> 64k: step (2048 tokens per row) 23.5 ms first / 30.0 median / 36.7 last at 64k (GA chunk cost grows with cache length), 24.1 ms median at 8k; replay-only 0.956 s per 64k user per row (4 users in parallel) = 274k tok/s aggregate for 4 layers (hint; scales ~1/layers), first case wall 25 s includes eager warm-up + trace capture, repeat 64k wall 1.84 s. DRAM per bank (ttnn memory view, 8 banks/chip): before build free 3882.4 MB, after build 3592.1 MB free (290.3 allocated: weights + KV pool max_pos 67584, 1 slot/row), after 64k 3589.8, after 8k 3589.8, after second 64k 3589.8: FREE DRAM IDENTICAL after each ISL (no leak across ISLs), +2.3 MB/bank only for the trace/io tensors. All logits finite. Built with PS_LAYERS=4 so first tokens are junk (no accuracy claim).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_ps4_64k_a43e4229c9f.jsonl
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests/test_prefill_scale.py
- data: {"step_ms_64k_median_4L":30.0,"free_mb_per_bank_after":3589.8}

## E261007-0557-332
- date: 2026-10-07T05:57:13
- commit: 990094ad6bc
- branch: ssinghal/mimo-asm
- host: 35,43,47,48
- track: asm
- status: verified
- claim: FULL 48-LAYER chunked traced prefill, memory_fit cells, real GSM8K text concatenated to the ISL, CH=2048 per mesh row, unified bf16 experts, bfp8 weights, KV pool sized to the cell (KVLayout users=B/4, max_pos=ISL+8 rounded to 2048), no OOM in any cell, all logits finite. FREE DRAM per bank after prefill (ttnn memory view, 8 banks/chip, so x8 = per chip; measured, not arithmetic): (B=4,ISL 4096) 2321.0 MB/bank = 18.1 GiB/chip free (after build 2324.7, weights+pool 1557.8 MB/bank allocated); (16,4096) 2278.3 MB/bank (17.8 GiB/chip); (4,65536) 2000.2 MB/bank (15.6 GiB/chip, after build 2003.9); (16,65536) 1805.7 MB/bank (14.1 GiB/chip, after build 1809.4). Chip DRAM 3882.4 MB/bank before build. Trace+io after first prefill costs only 3.7 MB/bank. TWO/THREE ISLs in one process (4 users, 64k -> 8k -> 64k): free DRAM identical (2000.2 MB/bank) after each case. Replay step (2048 tokens per row, 4 rows in parallel = 8192 tokens): 370-376 ms at 4k and 8k (median 375/391), 382 ms first chunk -> 445 median -> 505 ms last chunk at 64k (global-attention cost grows with cache length). Prefill tokens/s (replay only, real tokens over summed replay time): 21.9k at 4k (B=4 and B=16), 18.4k at 64k (B=4 and B=16); 64k per user: 14.2 s (B=4, 32 steps, 4 users in parallel), 56.9 s for 16 users (128 steps). Wall includes eager warm-up + trace capture (20-33 s first case). Weights load from the warm/cold NFS cache: 816-1260 s when 4 loaders ran at once (152 s alone). First tokens at 64k of the concatenated GSM8K text are not meaningful (no accuracy claim).
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_ps48_16_65536_990094ad6bc.jsonl
- paths: models/demos/blackhole/mimo_v26/tt, models/demos/blackhole/mimo_v26/tests/test_prefill_scale.py
- data: {"free_mb_bank":{"4_4096":2321.0,"16_4096":2278.3,"4_65536":2000.2,"16_65536":1805.7},"tok_s_4k":21900,"tok_s_64k":18400}
## E261007-0529-93f
- date: 2026-10-07T05:29:09
- commit: 56601b284bb
- branch: ssinghal/mimo-moe
- host: 35
- track: moe
- status: verified
- claim: Decode-sized MoE in a TRACE on 4x8 (MiMoMoEDecode: exact router + dead-slot spread via device mask + dispatch + unified bf16-activation op (patched .so, MOE_UNI_BF16_*) + combine + post_combine_reduce + all-reduce over columns), REAL layers 1-5 (real gate/bias/256 experts bfp8/dump hidden states), 32 slots/row padded to 64 tokens. Trace output bit-identical to eager (5 layers). PCC vs dump ffn_out (live tokens) 0.9993-0.99997 except L1 rows 0.991-0.998 at few users (routing flips). Traced time (host-timed, 30 warm replays of ONE trace holding 5 layers): 1.52 ms/layer at users/row 1, 8, 16 and 32 alike (7.60-7.63 ms per 5 layers) => 47 layers = ~72 ms/token for MoE alone (hint). Per-layer stage deltas (cumulative traces, users 32): pad+router+spread 0.288, bincount/offset_cumsum 0.091, dispatch 0.028, experts (unified) 0.775, combine 0.098, post_combine_reduce 0.192, all_reduce+slice 0.053 = 1.525 ms. Exact router alone in trace 0.241 ms/layer (x47 = 11.3 ms)
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_E_decode_trace_stages_15f4a1426e7.log
- paths: models/demos/blackhole/mimo_v26/tt/moe_decode.py, models/demos/blackhole/mimo_v26/tt/moe_prefill.py, models/demos/blackhole/mimo_v26/tt/router.py

## E261007-0529-48e
- date: 2026-10-07T05:29:09
- commit: 56601b284bb
- branch: ssinghal/mimo-moe
- host: 35
- track: moe
- status: verified
- claim: B decision data: single-card moe_compute (bfp4, compute_only, K=4096 N=2048, 8 experts, top-8) traced: 0.396 / 0.393 / 0.392 ms per call at T=32 / 16 / 8 tokens (independent of T => weight-read bound: 107 MB bfp4 / 0.393 ms = 273 GB/s). The unified op's expert stage on the real 4x8 trace: 0.775 ms for 214 MB bfp8 = 276 GB/s, i.e. the same effective bandwidth. A bfp8 moe_compute would read 2x bytes => ~0.79 ms, no gain over the unified op; only bfp4 would halve it (0.39 ms, NOT authorised). The expert stage of MoE at bfp8 is therefore ~0.775 ms x 47 = 36 ms/token at this bandwidth, above the 26 ms whole-model roofline (which assumed 450 GB/s). Recommendation: do NOT port the bfp8 moe_compute patch (no speed gain); remaining non-expert MoE overhead is 0.75 ms/layer (router 0.29, post_combine_reduce 0.19, combine 0.10, bincount 0.09) = 35 ms/token, the part worth optimising
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_E_mc_timing3_h47_56601b284bb.log
- paths: models/demos/blackhole/mimo_v26/tt/moe_decode.py, models/demos/blackhole/mimo_v26/tt/moe_prefill.py, models/demos/blackhole/mimo_v26/tt/router.py
## E261007-0516-fa4
- date: 2026-10-07T05:16:12
- commit: cce07a4da2b
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: dataset
- status: verified
- claim: GSM8K gate sets with enable_thinking=False (template appends '<think></think>' after '<|im_start|>assistant\\n'): /mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_128_nothink.json sha256 858aa02ca82df59cbfb881943e6039b272fdb38539b4a404c58e14caff82190b (max 137 prompt tokens), /mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_512_nothink.json sha256 4095204002434c3e96bf6e260384395260cfe43b4c9064f7b1cf001079086c10 (max 150). Verified: every prompt equals the old thinking-default prompt + '<think></think>' (same questions/order); scorer on gold solutions = 512/512. Old sets kept unchanged (gsm8k_128.json, gsm8k_512.json). Regenerate: gsm8k_set --n N --no-thinking --out ...
- evidence: /mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_512_nothink.json
- paths: models/demos/blackhole/mimo_v26/reference/gsm8k_set.py

## E261007-0540-c3b
- date: 2026-10-07T05:40:37
- commit: 4718774ea4d
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: correctness
- status: verified
- claim: PER-LAYER GATE L6-47 (device captures mimo-device-s128b16, teacher-forced: CPU oracle fp32 and bf16 run on each layer's DEVICE h_in; 16x128 real GSM8K tokens; table in /mnt/tt-data/ssinghal/mimo-logs/ref_device_vs_oracle_L6-47.jsonl, summary via probes/device_vs_oracle.py). Device vs fp32 oracle, rel L2 / floor (oracle bf16 vs fp32, same input): attn_out median 0.029/0.010 (SWA, ratio 2.9x), 0.030/0.0077 (global, 3.9x); ffn_out median 0.044/0.023 (1.9x SWA), 0.047/0.020 (2.4x global); h_out median 0.0023/0.0013 (SWA), 0.0018/0.0012 (global). Min PCC device-vs-fp32: SWA attn 0.99943 (L18), ffn 0.99763 (L6), h_out 0.99957 (L8); global attn 0.99948 (L17), ffn 0.99884 (L23), h_out 0.99964 (L47). Floor min PCC 0.9989 (ffn). FLAGGED (device error >=5x its floor): L6 SWA attn_out 5.6x; L16 SWA ffn_out 10.2x (rel 0.0119 vs 0.0012) and h_out 5.5x; L41 global ffn_out 5.8x and h_out 5.6x; next worst L17 global attn 4.4x, L47 global attn 3.7x/ffn 3.9x. No layer below PCC 0.9976 on any quantity; no systematic SWA/global gap beyond global attn ~1.3x worse ratio. ROUTER: device vs fp32 oracle (on oracle ffn_in) identical 8-expert sets median 94.8% SWA / 94.6% global (min 77.2% L7, 78.3% L8, 85.7% L6) vs floor median 97.8%/98.2%; but the fp32 router run on the DEVICE ffn_in reproduces the device sets for >=98.7% of tokens in every layer (min L7 98.7%, rest >=99.0%) => the device router op itself is right, flips come from ffn_in differences (earlier layers L6-8 worst). Near-tie share of flips (fp32 route_gap<0.01): 96.6-100% in every layer, i.e. all flips are near-ties.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/ref_device_vs_oracle_L6-47.jsonl
- paths: models/demos/blackhole/mimo_v26/reference/probes/device_vs_oracle.py, models/demos/blackhole/mimo_v26/reference/ref_layer.py
- data: {"swa": {"n": 35, "attn_out": {"min_pcc_dev_vs_fp32": 0.9994280338287354, "min_pcc_floor": 0.9998753666877747, "median_rel_dev": 0.02922009862959385, "median_rel_floor": 0.009965920820832253, "median_ratio": 2.9031930380016235, "argmin_pcc_layer": 18}, "ffn_out": {"min_pcc_dev_vs_fp32": 0.9976307153701782, "min_pcc_floor": 0.9988591074943542, "median_rel_dev": 0.04369692504405975, "median_rel_floor": 0.02291366085410118, "median_ratio": 1.8858154472183226, "argmin_pcc_layer": 6}, "h_out": {"min_pcc_dev_vs_fp32": 0.9995651841163635, "min_pcc_floor": 0.9998465776443481, "median_rel_dev": 0.0023199869319796562, "median_rel_floor": 0.0013270527124404907, "median_ratio": 1.7344244550224144, "argmin_pcc_layer": 8}, "router": {"median_set_equal_dev": 0.94775390625, "median_set_equal_floor": 0.97802734375, "min_set_equal_dev": 0.77197265625, "min_set_equal_dev_vs_router_on_dev_ffn_in": 0.98681640625, "min_near_tie_share_of_flips_tau0.01": 0.9655172228813171}}, "global": {"n": 7, "attn_out": {"min_pcc_dev_vs_fp32": 0.9994773864746094, "min_pcc_floor": 0.9999620318412781, "median_rel_dev": 0.029929591342806816, "median_rel_floor": 0.0076560028828680515, "median_ratio": 3.909297292687898, "argmin_pcc_layer": 17}, "ffn_out": {"min_pcc_dev_vs_fp32": 0.9988360404968262, "min_pcc_floor": 0.9995326399803162, "median_rel_dev": 0.04708743095397949, "median_rel_floor": 0.019828787073493004, "median_ratio": 2.43977421819306, "argmin_pcc_layer": 23}, "h_out": {"min_pcc_dev_vs_fp32": 0.9996418356895447, "min_pcc_floor": 0.9999626278877258, "median_rel_dev": 0.001813865383155644, "median_rel_floor": 0.0012090299278497696, "median_ratio": 2.269878716682899, "argmin_pcc_layer": 47}, "router": {"median_set_equal_dev": 0.94580078125, "median_set_equal_floor": 0.982421875, "min_set_equal_dev": 0.9111328125, "min_set_equal_dev_vs_router_on_dev_ffn_in": 0.99609375, "min_near_tie_share_of_flips_tau0.01": 0.9819819927215576}}, "flagged_ratio_ge_5": [[6, "swa", "attn_out", 5.6], [16, "swa", "ffn_out", 10.2], [16, "swa", "h_out", 5.5], [41, "global", "ffn_out", 5.8], [41, "global", "h_out", 5.6]]}

## E261007-0540-a8e
- date: 2026-10-07T05:40:37
- commit: 4718774ea4d
- branch: ssinghal/mimo-ref
- host: DC16-2-BG-0204-u20-44
- track: tool
- status: inferred
- claim: MTP-3 CPU oracle reference/mtp_ref.py (hypotheses in MTP_NOTES.md; no full model): load_mtp(k), mtp_layer (eh_proj on concat of enorm(embed(tok t+1+k)) / hnorm(h_prev), input_layernorm, SWA attention window 128 with sinks, pre_mlp_layernorm, dense FFN, final_layernorm) and evaluate(h_last pre-model.norm [B,S,4096], tokens [B,S+n+1], n_layers) = top-1 accuracy of predicting token t+2+k for 16 variants (concat order eh|he x h_src pre|post x head_norm final|final+norm x chain pre|post) with the shared lm_head. Only crash-tested on a stand-in hidden state (tests/test_mtp_ref.py passes, 76 s); NO accuracy measured: needs device-captured L47 h_out (pre model.norm) with the token stream. Status inferred: structure is a hypothesis.
- evidence: models/demos/blackhole/mimo_v26/tests/test_mtp_ref.py
- paths: models/demos/blackhole/mimo_v26/reference/mtp_ref.py, models/demos/blackhole/mimo_v26/tests/test_mtp_ref.py

## E261007-0542-8b2
- date: 2026-10-07T05:42:28
- commit: 6ce28192ecf
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: CORRECTION (coordinator): briefs/COMMON.md listed 'bfp8 attention weights default' among decisions already made. The user's decisions covered only routed experts bfp8. bfp8 qkv/o is the coordinator's working assumption inherited from the study (o_proj is bf16 in the checkpoint, qkv fp8 128x128 block). Evidence so far: attention PCC with bfp8 vs bf16 weights differs by about 0.0001 on L1/L5 at S=128 (attn agent E261007-0247-b41 and earlier); device-vs-oracle layers 6-47 run median 2-4x above the bf16-vs-fp32 oracle floor, which excludes weight quantisation (E261007-0540-c3b). To be put to the user; a bf16 o_proj switch is kept available (cost ~0.2 GB/chip extra).
- evidence: -
- paths: bringup/mimo_v26/briefs

## E261007-0550-d25
- date: 2026-10-07T05:50:36
- commit: fcfc8f0f421
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: USER DECISIONS 2026-10-07: (1) spec decode drafter = the model card's 5-layer SWA DFlash-style drafter predicting 7 tokens per forward pass (model card README section 'Speculative Decoder'; block size 8 = 1 verified + 7 drafted; checkpoint dir dflash/), NOT the in-checkpoint MTP-3 (interpretation of 'MTP 7 token prediction': the card calls the DFlash drafter an MTP drafter; coordinator to confirm in the report). (2) bfp8 attention weights confirmed (keep bfp8 qkv/o; supersedes the open question in E261007-0542-8b2). (3) GSM8K threshold: user asked to use a different benchmark prompt set from the model card; the README publishes only agentic suites (DeepSWE, ProgramBench, Terminal Bench, OSWorld ...) with no runnable prompts or example messages, so no such set exists locally: clarification requested, GSM8K non-thinking n=128 remains the interim runnable proxy with threshold not yet set.
- evidence: -
- paths: bringup/mimo_v26/scorecard.json

## E261007-0553-ce8
- date: 2026-10-07T05:53:58
- commit: bde698a68ac
- branch: ssinghal/mimo
- host: coordinator
- track: decision
- status: inferred
- claim: USER DECISIONS 2026-10-07: drafter confirmed = DFlash-style 5-layer drafter in the checkpoint dflash/ dir (7 draft tokens, block 8). GSM8K gate set to 85%: scorecard gsm8k min_n=128, min_correct=109 on gsm8k_128_nothink.json (sha256 858aa02c...2190b), non-thinking. Supersedes the provisional n=64/55 template value.
- evidence: -
- paths: bringup/mimo_v26/scorecard.json

## E261007-0556-577
- date: 2026-10-07T05:56:30
- commit: d0d86174096
- branch: ssinghal/mimo-moe
- host: 41
- track: moe
- status: verified
- claim: Measured per-chip DRAM of routed-expert weights on a real 4x8 mesh (host 41, 4 real layers 1-4, free-DRAM probe with 64 MB filler tensors per chip before/after): free DRAM per chip 32576 MB before; 4 layers use 832 MB (+-64 MB) = 208 MB/layer (+-16) for BOTH ND-sharded (unified-op layout) and interleaved layouts (analytic bfp8 204 MB/layer). EXTRAPOLATED to 47 layers (linear): 9.5 GB (+-0.75 GB) = 29% of the 32.6 GB free per chip; weights freed completely afterwards
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_F_dram_h41_d0d86174096.log
- paths: models/demos/blackhole/mimo_v26/tt/router.py, models/demos/blackhole/mimo_v26/tt/router_select.py, models/demos/blackhole/mimo_v26/tt/router_kernels

## E261007-0556-b7f
- date: 2026-10-07T05:56:51
- commit: da7b28c0b67
- branch: ssinghal/mimo-moe
- host: 41
- track: moe
- status: verified
- claim: Router cost (1 card, real L5 gate, traced, per call): exact router at 64 rows 0.231 ms = matmul fp32 0.092 + sigmoid 0.002 + bias add 0.003 + topk 0.044 + index conversions 0.014 + gather 0.041 + normalize/typecast/RM 0.036; matmul core grid helps (1x8 0.180, 2x8 0.172 ms total). Port of the DSV4 router_select generic_op kernel (exact fp32 top-8, lowest index on ties, normalised bf16 weights, one core per row, T<=32): variant sel (32 rows) 0.066 ms, sel + matmul grid 1x8 (32 rows) 0.031 ms = 7.4x cheaper than the exact router at 64 rows (~0.2 ms/layer, ~9.4 ms/token over 47 layers). Agreement of sel_mm1x8 over ALL 47 layers on real ffn_in (L1-5 bf16 prefill dump, L6-47 device-captured dump, 2048 tokens each, real gate/bias): set match vs fp32-CPU 0.9966 total (exact router 0.9966); expert sets identical to the exact router for 100.00% of tokens (L7 0.9995, all other layers 1.0000); weights PCC vs exact 1.000000; all flips have fp32 gap < 0.01; 2 runs bit-identical on every layer. Host 46 failed device init once (FW init timeout 'try resetting the board') at 05:35, host_health still said OK; not reset by me
- evidence: /mnt/tt-data/ssinghal/mimo-logs/moe_F_router_agree_h41_6b0f7f3ccc7.log
- paths: models/demos/blackhole/mimo_v26/tt/router.py, models/demos/blackhole/mimo_v26/tt/router_select.py, models/demos/blackhole/mimo_v26/tt/router_kernels

## E261007-0557-8b9
- date: 2026-10-07T05:57:55
- commit: 9f2672231cf
- branch: ssinghal/mimo
- host: 46
- track: infra
- status: inferred
- claim: Host .46 added to bad_hosts.txt as SUSPECT after two anomalies by two agents (FW init failure at device open while host_health said OK; a 48-layer run stuck at ~2500% CPU for 18 min). Not a proven fault; no reset was done (not mine to reset); skip until a clean canary. host_health.py cannot detect FW-init failures (defect).
- evidence: -
- paths: models/common/bringup_kit/bad_hosts.txt
## E261007-0559-bd9
- date: 2026-10-07T05:59:42
- commit: 34b6fa4ef62
- branch: ssinghal/mimo-asm
- host: 35
- track: asm
- status: verified
- claim: MERGE CANDIDATE ssinghal/mimo-asm @ 34b6fa4ef62 (ssinghal/mimo merged in, incl. moe_decode.py): re-check after merge, s2048b1 CH=512 L0..L5 traced chunked prefill: hidden L5 PCC 0.99972, logits 0.99945 top1 1/1, replay bit-identical, hand-off decode min 0.99986 over L0-L5 (decode MoE uses spread_pad_routing for the padding rows). Contents: tt/prefill.py, tt/kv_pool.py, layer.chunk(), model layout kwarg, tests test_prefill_handoff.py / test_prefill_scale.py / test_capture_device.py, notes/kv_pool.md.
- evidence: /mnt/tt-data/ssinghal/mimo-logs/asm_pf_merged_34b6fa4ef62.jsonl
- paths: models/demos/blackhole/mimo_v26/tt
