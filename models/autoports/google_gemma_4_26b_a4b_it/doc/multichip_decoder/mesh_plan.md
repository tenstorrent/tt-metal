# TP4 plan, before implementation

Target: four Blackhole P300c ASICs, one 1x4 mesh, FABRIC_1D, Linear CCL,
initial num_links=1. All four open/close successfully. Shape-faithful reduce-scatter
[1,1,32,2816] passes exactly in BF16 and FP32 (collective_smoke.log).
Baseline: OptimizedDecoder at a9259624f2 / documentation 2e3a1779d3.
This is a candidate plan, not an optimality or completion claim.

| Tensor | Global logical shape | Local physical shape / mapping |
| --- | --- | --- |
| residual | [1,1,S,2816] | candidate replicated or hidden-sharded [1,1,S,704] |
| sliding Q | 16x256 | 4x256 |
| sliding KV | 8x256 | 2x256; cache [pages,2,32,256] |
| full Q | 16x512 | 4x512 |
| full KV | 2x512 | 1x512, head0 on ranks0/1, head1 on ranks2/3 |
| packed sliding QKV weight | [2816,8192] | [2816,2048] column shards |
| packed full QKV weight | [2816,10240] | [2816,3072], duplicated KV across rank pairs |
| attention output weight | [4096 or8192,2816] | [1024 or2048,2816] row shards |
| expert gate/up | [128,2816,2x704] | pad I=768; each rank [128,2816,2x192], paired packing |
| expert down | [128,704,2816] | [128,192,2816], zero padded K |
| shared gate/up | [2816,2x2112] | pad J=2176; [2816,2x544] |
| shared down | [2112,2816] | [544,2816] |
| router/norm/RoPE | original | replicated; router top8 identical on all ranks |
| page table/positions | original logical indices | replicated; identical physical-page ownership per rank |

Sparse experts retain all128 checkpoint experts partitioned over intermediate
width; only gate-selected active experts execute decode. Weight dtypes start
from optimized policies (BFP8 sliding gate/up, BFP4 full gate/up and both down).
Do not replicate full experts: it wastes capacity and single-user bandwidth.
EP4 is an alternative but creates active-expert imbalance and dispatch traffic;
it remains unmeasured. DP does not help the batch1/concurrency1 target.

H=2816/4=704 is tile aligned. I and J padding stays inside MLPs and has zero
down weights. Public logical S remains arbitrary; inherit optimized chunk/tail
and prefix orchestration. Every layer must consume its own output layout.
Allreduce occurs after score-weighted local expert reduction, never per expert.

## Collective candidates

Let A = physical_rows *2816*bytes_per_element; each logical width shard is A/4.
Ring-equivalent traffic below is per-rank algorithmic volume, excluding headers;
actual Linear topology timing must be measured.

| Boundary family | Residual before/after | Next consumer | Nominal bytes/rank | Dtype | Persistence / decision |
| --- | --- | --- | --- | --- | --- |
| local matmul + allreduce | replicated/replicated | post norm | 1.5A | FP32 and BF16 controls | common ping-pong semaphores; reference candidate |
| reduce-scatter delayed gather | replicated/sharded | distributed post norm + residual | .75A plus small norm stats | FP32/BF16 | gather only before next column projection |
| fused gather-matmul | sharded/local partial | next row projection | .75A | op-supported | test packed projection contract |
| fused matmul reduce-scatter | replicated/sharded | distributed post norm | .75A | op-supported | persistent output/semaphores; check BH support |
| hidden-sharded residual end to end | sharded/sharded | distributed input norm | stats + gathered projection inputs | FP32/BF16 | adapt both shared/routed post norms; gather only at test boundary |

Measure producing and consuming operations together. Immediate RS->AG alone
does not reject a sharded residual. No performance rejection is earned yet.
Common Attention1D does not express Gemma4 Q/K/V head RMSNorm, tied global KV,
partial RoPE and post-attention normalization directly. Common MLP1D differs
in GeGLU and the independent shared/expert post norms. Reuse baseline attention
math and GPT-OSS sparse experts/CCL families with explicit local dimensions.

## Capacity target

Retain262144 logical tokens. BFP8 tile is1088 bytes. Local K+V payload is
2*262144*2*256*1088/1024=285212672 bytes per sliding layer and
2*262144*1*512*1088/1024=285212672 per full layer. Global KV pair replication
makes full cache reduction2x, sliding4x. Full-stack weight/cache accounting and
reserved trace/activation peak are still required; no capability reduction.
