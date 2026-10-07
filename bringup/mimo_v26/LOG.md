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
