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
