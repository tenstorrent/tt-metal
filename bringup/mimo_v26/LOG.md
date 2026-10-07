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
