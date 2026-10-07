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
