# Gemma 4 26B-A4B Mesh Plan

Target host: 8x Blackhole p150b loudbox.

Initial mesh: `1x8`, tensor parallelism along the width axis.

## Decode Plan

| Component | Ownership |
| --- | --- |
| Residual stream | Replicated logical hidden vector after TP allreduces. |
| Dense projections | Q/K/V, shared MLP gate/up, and LM head are column-parallel where output width is large. |
| Output/down projections | Attention O, shared MLP down, and expert down use row-parallel sharding followed by TP allreduce. |
| Sliding KV | 8 KV heads map naturally to 8 TP devices, one KV head per device. |
| Full KV | 2 KV heads are fewer than TP; current correctness path assigns/replicates needed KV heads by Q-head group. |
| Experts | Current path shards expert intermediate across TP; replication remains a candidate if DRAM allows and latency improves. |
| CCL | Allreduce after row-parallel attention output, MLP down, expert down, embedding all-gather, and non-sampled LM output. |

## Memory Estimate

Checkpoint metadata reports about 51.6 GB BF16 source weights. Across 8 p150b cards, the raw average is about 6.45 GB per card before padding, caches, replicated weights, and TTNN tensor-cache overhead. Each p150b board reports 16 GB GDDR, so a TP-sharded BF16/dequant correctness path is plausible; fully replicated experts in BF16 are not plausible without careful accounting.

KV cache for the demo `max_seq_len=4096` is small relative to weights:

Sliding layer logical K+V per token: `2 * 8 * 256 * 2 bytes = 8192 bytes`.

Full layer logical K+V per token: `2 * 2 * 512 * 2 bytes = 4096 bytes`.

For 25 sliding and 5 full layers at 4096 tokens, logical BF16 KV is about 0.94 GB before paging/padding/replication.

## Open Topology Questions

1. Whether full-layer KV replication across Q-head groups is faster than sharded ownership on Blackhole for batch=1 decode.
2. Whether expert weights should remain TP-sharded or move to active-expert replicated/packed layout.
3. Whether Gemma4 needs a nonzero trace region size on 8x p150b.
