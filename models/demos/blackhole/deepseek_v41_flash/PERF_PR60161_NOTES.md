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

## KV sequence-sharding over the 8 mesh columns: communication cost (tests/test_kvshard_comm_microbench.py, .44, T=16 users = 32 tile rows, 2 links)
Replicated KV today (attention is column-local, no cross-column traffic). With the cache split by position every column needs q for all 64 heads and the partial
attention outputs must be merged: q all_gather [32,4096]->[32,32768] 47.0 us + partial-output reduce_scatter [32,32768]->[32,4096] 39.8 us + max/sum all_gather
[32,128] fp32 20.0 us = **~107 us per attention layer, ~8.5 ms/token over 40 layers** (communication only; the extra partial-softmax math and the indexer top-k merge come on top).
Verdict: a capacity feature, not a latency one - gate it to long contexts (it frees KV DRAM 8x and divides the indexer work by 8), keep the replicated layout at <= 131072.
Design must shrink the traffic first (send only the T real rows instead of 32 padded rows, bf16 -> keep q for 8 heads and exchange partial (o, m, l) instead of gathering q,
or shard the heads of the 64 across columns differently).

## Per-round top-2 logits in the spec round (SpecDecoder.forward) - measured 2026-10-15, not kept
`ttnn.topk(k=2)` on the fp32 [T,16160] column-shard logits + all_gather costs 2.45 ms + 46 us per round in a trace (tests/test_top2_microbench.py); only the demo's
near-tie report reads it (the vLLM server never does). Making it optional (DSV41_SPEC_TOP2=0) gave:
- 4-layer model (DSV41_LAYERS=0-3, B=32, k=3): round 41.1 / 41.0 ms with top-2 vs 38.2 / 38.2 ms without (-2.9 ms, repeatable);
- full 40-layer model: round 113.6 / 114.3 ms (two runs) with vs 114.1 ms without - no measurable difference (A/A spread ~0.7 ms), accepted drafts/round 2.353 / 2.353 vs 2.382.
So the saving is hidden at full size (something else bounds the round there) and it is not claimed. Code reverted.
Side finding: the k=3 spec round of the 4-layer model takes ~38 ms, i.e. roughly a third of the 114 ms round is fixed cost independent of the layer count (head + sampling +
accept/commit + drafter + host feed/readback, Engram prep); 4 -> 40 layers adds ~75 ms. That fixed part, not the layers, is the next thing to profile.
Demo spec runs: dsv4-logs/base/perf_top2_{on,on2,off}.log, perf_t2L4_*.log; launcher dsv4-logs/base/perf_demo.sh <name> <spec k> "<extra env>".

## KEPT: non-blocking per-round readback (tt/spec_model.py SpecRunner._readback, DSV41_READBACK_NB, default on) - measured 2026-10-15
Host profile of a k=3, B=32 spec round at 40 layers (temporary timers, demo on .44): device wait 103.4 ms; host: readback 4.2-4.7, Engram row gather 2.7, upload 2.3, page
setup 1.3, hash 0.8 (= ~11-12 ms, i.e. 10 % of the 114 ms round, device idle). The readback did one blocking ttnn.to_torch per mesh row; issuing the 4 row-leader
reads with ttnn.from_device(blocking=False) and waiting once (microbench tests/test_readback_microbench.py: 0.93 -> 0.35 ms, identical data) cuts the in-round host
readback 4.19 -> 2.33 ms.
- Round time (demo, 40 layers, B=32, k=3, same code, flag on/off): 112.9 ms vs 114.4 ms; earlier pair 112.4 ms (on) vs 113.6 / 114.3 / 114.6 ms (off, three runs): ~ -1.5..-2 ms per round (~1.5 %).
- Exactness: DSV41_READBACK_CHECK=1 reads every round's pack both ways and asserts equality: >= 100 rounds of the full 40-layer demo identical, no assertion; first token spec==plain 32/32 in all runs.
  (Spec acceptance varies 2.32-2.38 between identical-code runs, so acceptance cannot be used as an exactness check here.)
Rejected in the same session: column-sharded Engram-row upload (DSV41_ROWS_SHARD wired into the spec path): upload 3.1 -> 7.96 ms, round slower (host-side column slicing is expensive).
Remaining host items: Engram gather 2.7 ms (LUT dequant could save ~1.5), upload 2.3 ms, page setup 1.3 ms; moving the Engram hash + row gather onto the device (kernels exist: engram_hash.py, engram_table.py) would remove ~8 ms but is a large change.
