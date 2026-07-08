# DiffusionGemma perf-optimization campaign (#47465 → goal 100 tok/s)

Optimization unit = **traced** denoise step over the 256 canvas + commit (dg-08 methodology).
Config: Blackhole QB2 `(1×4)` TP, tuned true-sparse MoE (`DG_SPARSE_MOE_TUNED=1`, HiFi2).

## Baseline (2026-07-08, tuned, 30 layers)

| path | ms | note |
|---|---:|---|
| denoise step — **traced** | **233.4** | the ranking metric |
| denoise step — eager | 720.8 | trace = **3.1× faster** (dispatch overhead is 68% of eager) |
| commit — eager | 129.0 / token | not yet traced; per-token autoregressive |
| prefill TTFT (18-tok prompt) | 607.9 | |

Traced 233 ms ≈ device-FW sum (276 ms/dev projected), i.e. trace closes the eager dispatch gap;
the eager op-breakdown therefore maps directly onto the traced step.

## Op-topology audit (share of the traced denoise step)

From the 2L+6L→30L device-FW decomposition (`whole_gen_opprofile/`):

| bucket | share | ~ms of 233 | where |
|---|---:|---:|---|
| MoE + attention **Matmul** | 35% | ~82 | 5 matmuls in `sparse_moe.sparse_experts_forward` (dispatch, gate/up/down, combine) + attn proj |
| **layout / glue** | 28% | ~65 | `build_capacity_dispatch` typecast×4 / scatter / gather / slice / reshape; tilize↔untilize; sharded↔interleaved |
| **elementwise / reduce** | 22% | ~51 | BinaryNg, Unary, Reduce (activation, routing, entropy-accept) |
| LayerNorm | 6% | ~14 | |
| TP collectives | 4% | ~10 | AllGather / ReduceScatter |
| diffusion token-select (ArgMax) | 4% | ~10 | per-step, fixed |

Permute cumsum-artifact is **gone** (1.8%) with the capacity-dispatch MoE — the old #47465
`SparseMatmul+Permute` breakdown is obsolete.

## Prioritized levers (to be applied + measured one at a time, traced before/after)

1. **layout/glue in `sparse_moe.py` (28%)** — collapse redundant `typecast`s (idx→uint32 *and* →float32),
   fuse the scatter/gather dispatch-matrix build, avoid tilize↔untilize round-trips. Lowest risk, in-repo.
2. **elementwise fusion (22%)** — fuse activation + routing-weight multiplies (BinaryNg/Unary chains).
3. **Multiple Command Queues** — overlap input writes / output readback with compute (tt-enable-tracing skill).
4. **commit path** — commit is eager 129 ms/tok; trace + batch it (batched-decode work) — likely the largest
   full-generation lever if a block commits many tokens. Audit block-time split (denoise Σ vs commit) next.
5. **datatype sweep** — bf8 experts (DRAM 11.6→~5.8 GiB/chip, faster matmul) if fidelity holds (datatype-sweep skill).

Roofline: per denoise step re-reads all resident weights (13.1 GiB/chip, 88.6% MoE experts) over the full
256 canvas — weight traffic, not incremental KV, sets the floor. `100 tok/s` needs the block-time split first.

## Log
- 2026-07-08: baseline captured (traced 233 ms/step); op audit from whole_gen_opprofile; levers prioritized.
