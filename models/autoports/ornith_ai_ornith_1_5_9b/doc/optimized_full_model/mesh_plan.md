# Full-model TP4 topology and policy

The target is four P300c Blackhole chips (IDs0–3), mesh1×4, FABRIC_1D,
Linear CCL topology, two links. The full-model starting point is ebd521f7.
The model is dense: 24 linear-attention and8 full-attention layers, hidden4096,
vocabulary248320. No EP/MoE applies. The complete32-layer model is measured;
profiling alone reduces the stack to real layers0/3.

| Tensor/boundary | Global shape/ownership | Per-device shape and placement | Selected action |
|---|---|---|---|
| Embedding | [248320,4096], hidden TP4 | [248320,1024] BF16 row-major DRAM | Gather once at model entry |
| Decode residual | [1,1,B,4096],1≤B≤32 | Replicated across ranks, BF16 L1 width-sharded8×4, [32,128] core shards | Carry exactly through every layer and final norm |
| Packed QKV/gated linear projections | Output TP4, local attention heads | Existing BF8 DRAM-sharded weights; HiFi2 | Preserve optimized multichip policy |
| MLP gate/up/down | Column TP4 then row TP4 | Existing BF8 weights, LoFi | Preserve selected separate decode gate/up |
| Full KV | Eight layers, local KV head per rank | BF8 paged K/V, BF16 updates | Preserve pool ownership, page tables and context262144 |
| Linear state/history |24 layers | FP32 recurrence, BF16 tiled pooled convolution history | Preserve inactive-row protection and native batch path |
| Final norm | Complete4096 hidden on each rank | Local RMSNorm in existing residual shard; HiFi2 FP32 accumulation | No new collective or interlayer conversion |
| LM head | [4096,248320], output TP4 | [4096,62080], BF8 DRAM-width-sharded over8banks; bank width7776 | Three readers/bank, K-block2, output target64cores/per_core_N31, BF16 output |
| Sampler logits | Four disjoint vocabulary shards | [1,1,32,62080] BF16 | Physical32 candidates/shard; semantic greedy k1 |
| Candidate collective |32 values+indices per user/rank |128 candidates after TP4 gather | Preserve common sampler; no full-vocabulary gather |
| Feedback/positions | Stable native32-token input plus slot positions | Device token write via tt_out_tok; device plus_one positions and seed | Nonblocking split model/sampling replay |
| Page tables | Scheduler-owned logical mappings | Persistent signed device tables | Only admission/explicit changed-table command refreshes |

The replica residual is the selected stage5 contract, not a stage7 replacement.
Stage5 compared a stack-compatible hidden-fractured stream through distributed
norm, residual addition and subsequent projections, including persistent/fused
AG/MM/RS families. Its [topology ledger](../optimized_multichip_decoder/topology_decisions.md)
and [work log](../optimized_multichip_decoder/work_log.md) remain authoritative.
No decoder weight, activation, fidelity, KV, collective or residual setting changes.

For the terminal column-parallel head, each rank already owns complete hidden
input and computes a distinct vocab shard. Row-parallel head+reduce-scatter would
create vocab-sized partial results and extra communication without removing an
existing hidden gather. Full-vocab all-gather is the rejected forced-argmax
candidate; split-greedy only gathers32 candidates. There is no terminal row
projection followed by a hidden consumer to fuse with AG/MM/RS.

`LMHead1D` from the pinned tt-transformers checkout is used directly, with all
weights loaded before capture. Its `lm_head_dtype` controls output dtype and is
BF16; its LazyWeight explicitly retains BF8. The complete original HiFi2,
FP32-destination compute config is supplied unchanged. One interleaved BF8 head
copy remains for arbitrary-height prefill; the decode kernel admits one tile
row only. The additional sharded copy is included in the updated full-context
allocator measurements. No logical context or batch limit is lowered.

The whole-vocab one-reader candidate exceeded L1 before dispatch. Four-way
local-vocab splitting made it legal and bit-identical, but slower. Multi-reader
whole-vocab geometry is legal and faster. Geometry artifacts preserve reader
counts1–4, K blocks2/4/8/16, output targets64/110, exact L1 blockers, and successful
adapted split/whole variants. The selected policy stays BF8/HiFi2 throughout.
