# Decode optimizations from tt-metal PR 60161 (gpt-oss-20b): what transfers to DeepSeek-V4.1-Flash (measured 2026-10-14, .44)

Reference: PR 60161 turns on only for a 1x4 Blackhole mesh, TP=4, 1 token per device (batch 1); prefill and every other mesh/batch stay on the old path. Its
gain (18.35 -> 2.47 ms/token) comes from DRAM-streaming kernels that read only the 4 routed experts, a packed 6 KB all-reduce fused with residual + RMSNorm, and an
LM-head + top-32 sampling kernel. Four read-only reviews of the PR against our code and the L3 decode op table (MoE layer, B=16, traced 985 us: moe_compute 361,
attention 222, shared expert 111, mHC ~110, MoE glue + reduce_scatter ~107, MoE all_gather 25; programs run back to back, ~27 us of gaps) ranked ideas; each was then
measured before being kept. Result: none gave a speed-up here.

| idea | measurement | verdict |
|---|---|---|
| sharded multi-core q-norm (attention `_qkv_fused`) | op 16.0 -> 5.1 us alone, 10 us with the slice + 2 reshards (tests/test_qnorm_microbench.py); PCC 0.99998 vs the old op, bit-exact no. Layer op table: kernel time -5.7 us but +2 ops -> traced 984.9 -> 987.6 us | no gain, reverted |
| kv-norm sharded | 9.4 -> 9.0 us with reshards | no gain |
| router matmul on more cores / lower fidelity | 22.5 us on every grid 1x4..8x8 (compute-bound at HiFi4 fp32-acc); HiFi2 / LoFi 18 us but not bit-exact (expert selection must stay exact) (tests/test_router_microbench.py) | rejected |
| bias add folded into `router_select` kernel | bit-exact (T=16/32/64, shared and per-row bias) but select 31 us vs add + select 15 us (soft-float on the scalar core) | rejected, reverted |
| `all_reduce_async` instead of reduce_scatter + all_gather (attention output, [32,5120] bf16, axis of 8 devices) | pair 39.8 us net (2 links, production); all_reduce_async 36.8 us (PCC 1.0) but needs L1 width-sharded input/output: with the DRAM <-> L1 reshards 43.9 us. 3-4 links fail (fabric allows 2). (tests/test_allreduce_microbench.py) | no gain as a drop-in, <= ~3 us if neighbours kept L1 sharded |
| custom packed all-reduce (PR `boundary.py`) | not built: payload is 327 KB/device (160 KB real rows), not the PR's 6 KB; the pair is ~3x above its bandwidth bound so fixed latency dominates, gain unknown | open item, needs a kernel prototype |
| LM head / sampler / embedding fusions | review estimate 1-2 ms of a ~100 ms spec round; unmeasured | not started |

Take-aways: the PR's headline gain is batch-1 routing overhead we do not have (we already read only active experts, expert-parallel over 32 chips). Remaining big
levers are outside the PR: moe_compute (361 us on the slowest chip, 1.3-1.6x imbalance vs median), mHC (~110 us), Engram host prep (serial, in the 4.5-7 ms host gap),
and for long context KV sequence-sharding (capacity, not short-context latency).
Baseline B=32 plain demo (gsm8k_b32, ctx 512) at 682cba8558d: TTFT 574 ms, decode 57.3 ms/token = 17.44 tok/s/user (log dsv4-logs/base/perfbase_b32.log).
Tooling: dsv4-logs/base/perf_optable.sh <tag> <layer> [env] runs tests/test_layer_op_table.py + tests/op_table_build_v2.py (now tolerates empty duration cells).
