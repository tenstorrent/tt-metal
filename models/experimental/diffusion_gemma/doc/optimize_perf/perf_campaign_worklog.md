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

## CORRECTION (2026-07-08 session B) — the batched commit is NOT buggy; `verify_commit_batching` is an invalid gate

The "batched commit numerically broken / attention output is the bug" diagnosis above is a
**misdiagnosis** and is retracted. Isolated on-device probes settle it decisively:

**1. The batched attention is correct.** New isolation probe `probe_attn_only.py` = `verify_commit_batching`
with `layer.enable_moe_block=False` forced on **both** paths, so the layer tail (shared_mlp + per-row
RMSNorms) is identical and the *only* remaining difference is attention:

| probe | worst K/V PCC | speedup | verdict |
|---|---:|---:|---|
| attention-only, 4 layers | **0.9977** (L3) | 13.8× | attention PASSES |
| attention-only, 30 layers | 0.494 (L29); ≥0.99 through L8, 0.96 @ L11, 0.86 @ L15, 0.71 @ L19 | 16.7× | see note |

Layer-0 attention KV is 0.99999 (and prior single-layer `probe_commit_l0attn.py` = 0.99992). The 30-layer
decay to 0.49 is **not an attention bug**: with MoE off the batched (prefill masked-SDPA + chunked RoPE) and
sequential (decode flash-SDPA + per-user RoPE) are different **bf16** kernels, and deep-network residual
feedback amplifies their tiny per-layer differences. ⇒ **No two non-bit-identical commit implementations can
meet `--pcc 0.997` at `--num-layers 30`.** The 0.997/30L threshold measures bf16 chaos-amplification, not correctness.

**2. The full-model failure is the MoE, and the *sequential* MoE is the defective one.** With MoE on (baseline,
`DG_SPARSE_MOE_CAPACITY=32`): 4L worst 0.612 (L3 V), 30L 232/240 fail. The torch-oracle gate
`probe_moe_vs_torch.py` (re-run this session, bit-exact layer-0 input, routing agreement 0.9969):

| MoE output vs HF torch oracle | PCC |
|---|---:|
| **batched** (`sparse_experts_forward`) | **0.856** ← higher = correct kernel |
| sequential (`_commit_experts_decode_forward`, decode `sparse_matmul` nnz=8) | 0.579 |

**VERDICT (device): BATCHED matches torch; SEQUENTIAL is the buggy kernel** — reproducing the 2026-07-04
resolution in `commit_batching.md`. `_commit_experts_decode_forward` is a near-verbatim copy of the shared
`models/demos/gemma4/tt/experts/decode.py::decode_forward` (same `sparse_matmul` + reshape/transpose), so the
inaccuracy is the gemma4 decode sparse-MoE kernel, which the batched path deliberately avoids (it uses the
accurate prefill/capacity-dispatch MoE).

**Consequence:** `verify_commit_batching.py` compares the **correct** batched path against the **defective**
sequential reference, so its FAIL is expected and meaningless (already declared a non-gate in
`commit_batching.md`). The correct gate is `probe_moe_vs_torch.py`. **No change was made to `commit_batched.py`:
it is not defective, and forcing `verify` to pass would require corrupting the fast path to reproduce the
defective sequential MoE.** Ruled out along the way: swapping the batched MoE to the sequential decode MoE is
impossible (decode `sparse_matmul` requires batch=1 — `sparsity volume must == 128`); `DG_SPARSE_MOE_CAPACITY=256`
(drop-free) overflows L1 in the tuned MoE matmul (EC=32768).

**Recommendation:** gate commit correctness on `probe_moe_vs_torch.py` (batched↔torch) and a *single-layer*
attention probe, not on `verify_commit_batching.py` (batched↔sequential, 30L, 0.997). Separately, the defective
gemma4 decode sparse-MoE still governs the paged/vLLM sequential commit (#47488) and is worth fixing there.

## Log
- 2026-07-08: baseline captured (traced 233 ms/step); op audit from whole_gen_opprofile; levers prioritized.
- 2026-07-08: block-time split → commit (~31 s, 256 sequential decodes) dominates. Batched commit verified 24.8× faster but KV PCC fails (232/240, layer-0-exact then compounds). Next: route commit through chunked_prefill (preferred) or debug commit_batched attention.
- 2026-07-08 (session B): retracted the "attention bug" diagnosis. `probe_attn_only.py` (MoE off, both paths) → attention-only KV PCC 0.9977 @ 4L (PASS), 0.494 @ L29 (bf16 prefill-vs-decode compounding, not a bug). `probe_moe_vs_torch.py` re-run → batched 0.856 vs torch, sequential 0.579 → **sequential MoE is the defective kernel**. `verify_commit_batching.py` is an invalid non-gate (correct-batched vs defective-sequential; also unreachable at 0.997/30L for any non-bit-identical pair). No `commit_batched.py` fix; recommend gating on `probe_moe_vs_torch.py`.
