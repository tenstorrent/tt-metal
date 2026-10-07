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
