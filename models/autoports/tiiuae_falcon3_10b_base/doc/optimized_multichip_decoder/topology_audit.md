# Operation-topology audit

Status: completed before local tuning; final actions and measured outcomes are
appended below.

Measured graph: real Falcon3 layer 20, logical batch 32, one decode token at
position 17, TP=4 on the fixed 1x4 p300c mesh. Each device owns 3 Q heads, one
K/V head, 768 attention output channels, and a physically padded 6,144-channel
MLP shard. Projection weights are BFP4_B/LoFi; residuals and CCL payloads are
BF16; K/V cache is BFP8_B.

## Decode operation sequence

| Region | Current sequence and movement | Material issue or candidate | Constraint | Action |
|---|---|---|---|---|
| Entry and input norm | reshape; optional row pad; DRAM-to-width-sharded L1; 32-core sharded RMSNorm | Preserve the residual layout across layer boundaries and remove entry conversion after layer 0 | Public logical batch must remain distinct from 32-row tile padding | Measure a stack-native sharded residual API; gather/convert only in the comparison harness |
| QKV projection | residual-grid to QKV-grid reshard; one rank-grouped packed QKV DRAM-sharded matmul; sharded-to-L1-interleaved; `nlp_create_qkv_heads_decode` | Q/K/V packing is already selected; advisor layout and wider legal K blocks remain candidates | Per-rank packed order must remain `[Q_r,K_r,V_r]`; N=1,280 | Seed fresh TP-local shard advice, then compare against selected four-core DRAM-sharded path |
| Decode RoPE | position typecast and two embeddings; batch-32 path performs transpose/repeat/slice/neg/concat plus two multiplies and add for Q and K | Replace spelled-out rotate-half graph with a dedicated/fused QK RoPE op | Falcon3 uses rotate-half ordering, heterogeneous per-user positions, Q heads=3 and KV heads=1 | Search dedicated ops and adapt head/cos/sin layout; PCC and traced whole-layer timing decide |
| Cache and attention | two paged updates; explicit-config paged SDPA decode; DRAM-to-height-sharded conversion; `nlp_concat_heads_decode` | SDPA and BFP8 cache are already selected; minimize head-layout repair | Cache updates require BF16 inputs into BFP8 cache; logical batch/page-table semantics fixed | Retain unless profiler or a legal fused head path wins |
| O projection boundary | head-layout to two-core working shard; DRAM-sharded `[768,3072]` matmul; reshard to residual grid; BF16 async all-reduce (reported as RS+AG); residual add | Compare persistent-buffer minimal all-reduce and safe fused/lower-movement families; avoid restoring old layout inside a lower-movement timing window | TP4 Ring/two links; clean teardown and next-open health are correctness gates | Reuse prior AutoFix failure evidence; run only safe adapted candidates supported by current APIs |
| Post-attention norm | 32-core sharded RMSNorm, then residual-grid to gate/up-grid reshard | A phase-specific grid is intentional if it wins the whole MLP | Norm and both projections must consume compatible layouts | Fresh advisor seed plus existing geometry control |
| Gate/up | two same-input `[3072,6144]` DRAM-sharded matmuls; SILU fused into multiply | Pack weights into one `[3072,12288]` projection, split on device, and compare full MLP | Packed path must retain BFP4/LoFi and count split/layout cost | Implement candidate; compare real-weight PCC and traced prefill/decode against tuned split path |
| Down boundary | reshard; `[6144,3072]` DRAM-sharded matmul; reshard to residual grid; BF16 async all-reduce; residual add | Persistent CCL and lower-movement contract as above | Padded MLP channels must remain mathematically zeroed by down weights | Measure coherently with next-layer layout |
| Layer exit | residual-grid to L1 interleaved; reshape; L1-to-DRAM | Remove terminal conversion and next layer's reciprocal conversion | Preserve logical `[1,batch,1,3072]` API or expose an explicit stack-native equivalent | Measure a two-layer stack-compatible path; conversion for host PCC stays outside the measured layers |

## Prefill operation sequence

Prefill keeps large activations DRAM interleaved, uses packed QKV, fused head
split, dedicated rotary embedding, SDPA, concatenate-heads, row-parallel O plus
BF16 all-reduce, separate gate/up with SILU fused into multiply, and down plus
BF16 all-reduce. Rows above 1,024 are internally chunked; logical lengths 17,
31, 1,025, and 32,768 remain valid. The principal graph candidate is therefore
packed gate/up. Prefill program-grid/block sweeps from the input stage remain
the control. The only changed projection width, packed gate/up, was rerun end
to end in both DRAM and L1-sharded unpack forms.

## Collective/layout families

| Family | Residual before -> after | Approximate batch-32 bytes/rank per boundary | Ops | Persistent buffers | Current evidence/action |
|---|---|---:|---|---|---|
| Entry replicated all-reduce | replicated width-sharded L1 -> replicated width-sharded L1 | 294,912 Ring bytes | local matmul + allocation-bearing AR + add | implicit per call | Stable control, but slower than the persistent family |
| Selected persistent async all-reduce | same replicated contract | same payload; allocation/setup removed from replay | local matmul + async AR + add | one shared 786,432-byte/device intermediate and two 440-byte/device semaphores | Isolated and complete-decoder A/B passed; selected and watcher-clean |
| Reduce-scatter carry-forward | replicated input -> hidden-fractured residual | 98,304 local result; avoids immediate AG | local matmul + RS + local add + distributed norm | explicit RS buffers | Prior isolated boundary PCC 0.999794 and 1.00765x speedup, but poisoned fabric after process exit despite cleanup; AutoFix failed |
| Fused matmul-reduce-scatter | replicated input -> hidden-fractured residual | same lower-movement payload | fused MM+RS | explicit intermediate/output | Prior adapted one/two-link attempts hung; standalone MM+RS passed |
| Gather-input/local-output | hidden-fractured input -> output-fractured or replicated later | decomposition-dependent | AG+MM fused, optional later AG | explicit AG output | Prior TP4 attempt hit source-proven hardcoded four-transfer receiver accounting; current source was rechecked and retains the incompatible accounting |

No collective is needed *between* layers when the replicated residual is carried
forward. The optimized contract target is: replicated across the TP mesh,
BF16, width-sharded in L1 on the residual grid, physical decode rows tile-padded
internally, with logical batch retained separately. Any host-visible DRAM output
is an outer harness boundary, not part of the measured decoder stack.

## Final actions and outcomes

| Audited item | Action taken | Evidence and decision |
|---|---|---|
| Entry/exit residual movement | Added stack-native residual-return/consume APIs and measured two independently materialized decoders sharing one persistent pool without restoring DRAM between them | 0.710055 ms versus 0.778981 ms, PCC 1.0, 1.0971x; selected contract has zero inter-layer collectives |
| Packed QKV | Preserved the already fused rank-grouped projection and retested its program family through fresh shard advice | Existing four-core DRAM-sharded QKV remains fastest; exact advisor family raised decode to 0.641203 ms |
| QKV/O/gate/down reshards and program grids | Ran fresh compiler advice, exact candidate, O-only hybrid, an adapted legal 8x6 O retry, then recaptured the final dedicated-RoPE/stack graph with production cos/sin shapes and applied every feasible choice as one 96-core residual family | The corrected capture has 25 ops, 22 choices, two spills, and only concat is unfixable. The coherent candidate passes PCC and measures 0.451892 ms; its own two-layer residual measures 0.819200 ms versus 0.888366 ms with a DRAM boundary, but the full candidate is 15.45% slower than default |
| Decode RoPE primitive cluster | Replaced it with two dedicated rotary-embedding ops for tile-aligned batches; adapted non-tile batches to the owned explicit path; applied compiler-advised L1 rotary layouts in the corrected full-family candidate | Final dedicated 0.391425 ms versus exact final-graph explicit 0.514621 ms; corrected advisor RoPE candidate 0.451892 ms; heterogeneous batch 2 passes |
| Persistent collective resources | Added explicit persistent intermediate buffer, two semaphores, and sub-device manager for BF16 decode async all-reduce | Isolated 0.022101 ms versus 0.056970 ms, PCC 0.9999974; final graph persistent 0.391425 ms versus 0.437456 ms; two decoder instances share one pool cleanly |
| Collective topology/link placement | Retimed Ring/Linear and one/two-link families on the final graph | Ring/two-link 0.391425 ms selected; Linear/two-link 0.397344 ms; Ring/one-link 0.408873 ms |
| Activation/CCL dtype and fidelity | Retimed BFP8 CCL; BFP8/HiFi2 and BF16/HiFi4 weights; attention-only and MLP-only HiFi2 | BFP4/LoFi weights plus BF16 CCL are fastest; the initial BF16 L1 error was adapted from 8 to 24 down cores before rejection |
| Same-input gate/up | Implemented one packed projection, retried with an L1-sharded unpack, then reran both on the current dedicated-RoPE/direct-output/persistent-CCL graph | Both pass PCC; current graph is 0.392700 ms with DRAM unpack and 0.397105 ms with L1 unpack versus the 0.391425 ms separate-projection default |
| Public output staging | Added direct DRAM materialization for unpadded rows; adapted smaller batches after discovering physical/logical row mismatch | Batch-32 candidate reached 0.391468 ms and became default; batch-1, batch-2, and non-aligned suites pass through owned safe slicing |
| Lower-movement RS/distributed residual | Retained the coherent old layout through the old complete boundary; consulted existing AutoFix and post-exit health evidence | 1.00765x isolated at PCC 0.999794, but clean-exit hardware correctness failed after repair attempts; rejected by failed `$autofix`, not by an immediate API error |
| Fused matmul-RS / AG-matmul | Reused one/two-link, standalone/integrated, source, and hang evidence from the immediately preceding completed stage | Matmul-RS integration hangs although standalone passes; AG-matmul TP4 accounting is hardcoded for four transfers; no safe decoder-local candidate remains |

## Final inter-layer contract

Full-model bringup should call `decode_forward_from_residual` for an already
stack-native tensor and should defer `materialize_decode_output` until the final
outer boundary. The tensor carries replicated values, while each TP rank holds
BF16 L1 width-sharded physical shape `[1,1,32,3072]` on the 32-core residual
grid. Logical batch and sequence position remain separate metadata. Decoder
layers in one sequential stack must share the first layer's
`DecodeAllReduceResources`; non-owning layers are released before the owner.
This avoids gather, reshard, all-reduce, and DRAM round trips between decoder
layers; each decoder retains its two required internal row-parallel async
all-reduces.
