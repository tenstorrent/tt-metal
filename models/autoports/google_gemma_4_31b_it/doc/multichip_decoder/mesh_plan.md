# Stage 04 mesh plan (before implementation)
Baseline: optimized_decoder.OptimizedDecoder at 87140921cfc65695b74f0981b3c19b4182e46e61.
Hardware: P300_X2, four Blackhole ASICs, 1x4 linear fabric, two links available.
FABRIC_1D mesh open/close passed (mesh_smoke.log). Target TP4 across all ASICs.
Dense model; no experts or routing. Preserve 262144 logical tokens and batches1/8/16/32.

| Tensor | Global | Per device | Mapping / initial program |
|---|---|---|---|
| Residual | 5376 | 1344 sharded or5376 replicated | BF16;7 or8 L1 shard cores |
| Sliding Q/K/V | 8192/4096/4096 | 2048/1024/1024 | column TP; packed4096; K5376,28 senders,Kblock6,2 readers |
| Full Q/K(tied V) | 16384/2048 | 4096/512 | column TP; packed4608;8 senders,Kblock3,2 readers |
| Sliding O | 8192x5376 | 2048x5632 | row TP; zero output padding256;8 senders,Kblock8,2 readers |
| Full O | 16384x5376 | 4096x5632 | row TP;8 senders,Kblock16,2 readers |
| Gate/up | 5376x21504 | 5376x5376 | column TP;28 senders,Kblock6,1 reader (21 tiles/bank) |
| Down | 21504x5376 | 5376x5632 | row TP;8 senders,Kblock7,2 readers |
| KV sliding | pages,16,32,256 | pages,4,32,256 | BFP8 local heads,2176 bytes/token/device |
| KV full | pages,4,32,512 | pages,1,32,512 | BFP8 local heads,1088 bytes/token/device |
| RoPE/page tables/positions | request metadata | replicated | absolute positions; shuffled private pages |
| Hidden norms | 5376 | 1344 or5376 | distributed statistics for shard variant; head norms replicated |

DRAM weights width-sharded over8 banks with tile-rounded bank widths. Prefill starts with baseline MinimalMatmul M4/K8/N16; tune from measured evidence.
Local KV at262144 tokens:570425344 sliding /285212672 full bytes/device/layer. Full-stack memory accounting must precede acceptance; no context reduction is authorized by this estimate.

## Collective candidates
Let A=M*5376*2 bytes (M padded to32 for decode); TP4 ideal per-rank all-reduce payload1.5A and reduce-scatter0.75A; implementation/link scheduling adds overhead.

| Family | Residual before/after | Next consumer | Bytes per boundary | Storage/dtype |
|---|---|---|---|---|
| Local MM + full reduction | replicated/replicated | local post norm + residual |1.5A | BF16 first, BFP8 trial; ping-pong semaphores |
| RS + delayed gather | sharded/sharded | distributed post norm, residual, distributed input norm, then gather for next column MM |0.75A RS +0.75A delayed AG + stats | BF16 residual; BF16/BFP8 communication trial |
| Fused AG-MM | sharded/column output | head split or gate multiply |0.75A AG | persistent buffers; inspect supported dtype/geometry |
| Fused MM-RS | partial/sharded | distributed post norm |0.75A | persistent buffers; adapt projection padding |
| Residual-sharded |1344/1344 | next layer directly | no boundary gather | norms reduce stats; gather only column input |

Choose winner only after coherent whole-layer measurements. Immediate RS->AG is only replicated control, not evidence against delayed gather. TP2+DP2 leaves half bandwidth unused at batch1 and duplicates full weights; 2D TP adds collectives to both axes on just4 devices. Neither is selected initially. Common Attention1D expects Meta RoPE and differs in tied raw KV/value norm and Gemma post-projection norms; use model-local baseline math with common CCL/distributed norm primitives. MLP1D is configurable GELU but its integrated reduction contract does not include Gemma post-MLP normalization; retain explicit boundary to compare layouts.

## Selected path, after bounded comparisons

The final implementation selects TP4 on `FABRIC_1D_RING` with two links and `Topology.Ring`. Replicated BF16 residuals enter/leave both phases; decode post-projection norms remain in 8-core L1. Column and row ownership, local heads, and page metadata remain as in the original plan. Gate/up shards are padded5376→5632 independently (global21504→22528), with matching zero down-projection rows; useful channels remain21504. Gate/up now use2 DRAM readers and down uses Kblock11. Full-attention QK uses Kblock21 (the complete168-tile input is sharded over8 senders,21 tiles each);3 readers was measured slower. Sliding QKV keeps28senders/K6/2readers. Output projection padding5632→logical5376 is sliced before reduction.

The row boundary is local MM → RS → AG → local norm/residual; prefill communication uses BFP8 (with BF16 restored before norm), decode uses BF16. Both collectives have explicit barriers and cyclic global semaphores supplied by `TT_CCL`. Prefill CCL operations own transient outputs. The final decode path preallocates RS staging/output and AG output buffers independently for each projection and B1/8/16/32 signature; see the final persistent-buffer controls in work_log.md. Buffers remain alive with the decoder and are overwritten before consumption on every replay. Stats remainBF16 in the experimental sharded alternative.

At4096/128/B1, the Ring replicated candidate costs approximately11.27ms+128*0.437ms for sliding attention. The adapted fused AG/sharded candidate with BF16 projection output passes PCC but costs10.98ms+128*1.095ms and retains an additional interleaved projection copy. Its BFP8 projection-output variant fails prefillPCC(.98495); switching only those outputs toBF16 restores.999939. This controlled precision result and the measured decode loss reject that family for the selected workload. Fused RS with sharded residuals also passes after matching Ring fabric but costs~870us decode. No boundary gather is charged to the sharded test comparison inside measured time.

`candidate_layer0_*.json`, `candidate_layer5_*.json`, `sliding_fused_ag_bf16.json`, `sliding_fused_ring.json`, and `sliding_*16.json` retain coherent whole-layer alternatives. Padded6144, smaller gate sender grids, K14/N8 prefill, and full QK3-reader candidates do not beat the chosen path. Largest pref K21 hits a recorded L1 limit; four DRAM readers violate the op's range. These are measured/configuration-limited rejections, not assumptions based on the old implementation.

The corrected packed-prefill control writes BFP8 directly, matching split projection output dtype. It still loses about 3 ms at B32 over 4096 prefill plus 128 decode steps, while winning less than 0.5% at B1. The balanced split policy is retained. See the corrected-control table in work_log.md and `candidate_layer*_finalpacked8*.json`; the earlier BF16-then-cast packed control is superseded for this decision.
