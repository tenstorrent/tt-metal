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
