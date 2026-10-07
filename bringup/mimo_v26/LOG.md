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
