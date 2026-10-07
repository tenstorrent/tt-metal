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
