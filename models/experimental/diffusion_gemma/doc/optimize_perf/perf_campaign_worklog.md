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

## Block-time split (2026-07-08) — commit dominates, not denoise

| phase | per 256-token block | note |
|---|---:|---|
| denoise | ~11.2 s | ≤48 steps × 233 ms traced |
| **commit** | **~31–35 s** | **256 sequential single-token decode-appends** (one full 30L forward each) |
| ⇒ block | ~43 s → ~6 tok/s | commit is ~75% of block time |

**So the #1 lever is the commit, not the denoise step.** After denoise the 256 canvas tokens are
known; populating their KV is algebraically a *causal prefill of 256 tokens at start_pos*, not 256
autoregressive decodes.

## Batched commit — 24.8× prize, but currently numerically broken

`verify_commit_batching.py --num-layers 30` (cloned KV caches, seq vs batched, per-layer PCC):

- **speedup = 24.82×** (commit 35.1 s → **1.41 s**). This alone would take the block ~43 s → ~13 s ⇒ ~6 → ~20 tok/s.
- **PCC FAIL: 232/240 K/V checks fail.** Signature: **layer 0 K/V = 0.99999 (exact), then error compounds every layer** (L1 K 0.89, L5 0.41, L11 0.04, L24 V = −0.10). Classic accumulation.
- Diagnosis: layer-0 *written K/V* is correct, so embedding + K/V projection are right; the divergence is in the **attention output** (hidden state passed layer→layer). Suspects in `commit_batched.py`: the `build_device_commit_causal_mask` prefix+canvas causal pattern, canvas-Q RoPE offset, or the read-back-then-SDPA numerics — not the KV write.

### Two paths forward (next iteration)
1. **Reuse the proven `chunked_prefill.py` (PCC 0.9997)** for commit — it already does correct causal +
   sliding-window prefill at a non-zero `start_pos` (`chunk_start_idx` + `paged_fill_cache`). Commit ≡ prefill
   the 256 canvas tokens at `start_pos`; correct-by-construction, sidesteps the `commit_batched.py` mask bug. **Preferred.**
2. Debug `commit_batched.py`'s attention layer-by-layer (compare batched vs sequential attention output at layer 0,
   isolate mask/RoPE/SDPA). Higher risk.

Do NOT enable batched commit until PCC ≥ 0.997 (KV correctness gates block-to-block coherence).

## Log
- 2026-07-08: baseline captured (traced 233 ms/step); op audit from whole_gen_opprofile; levers prioritized.
- 2026-07-08: block-time split → commit (~31 s, 256 sequential decodes) dominates. Batched commit verified 24.8× faster but KV PCC fails (232/240, layer-0-exact then compounds). Next: route commit through chunked_prefill (preferred) or debug commit_batched attention.
