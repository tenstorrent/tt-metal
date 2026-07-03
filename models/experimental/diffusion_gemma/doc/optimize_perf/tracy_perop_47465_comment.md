## Per-op + inter-op-gap breakdown (Tracy-enabled build)

Follow-up filling the two gaps from the earlier comment. I built a **Tracy-enabled tt-metal** (`ENABLE_TRACY=ON`, in a git worktree so the shared `build_Release` is untouched) and profiled one denoise step:

```
prof_denoise_step.py --num-layers 2 --canvas-length 256 --iters 3 --no-trace   # reduced 2-layer real-ckpt DiffusionGemma, (1,4) QB2 P150x4 mesh
python -m tracy -r -p            # enriched op report: per-op DEVICE FW/KERNEL duration + OP TO OP LATENCY
```
The enriched report carries the `DENOISE_START..DENOISE_END` signposts, so the per-step region is isolated exactly. All numbers below are **device 0** (TP=4, per-chip).

### Gap #1 — per-op device decomposition (the ~137.5 ms/layer)
Inside the denoise region the per-layer device time is **almost entirely one op**:

| op | device-fw | %fw | kernel | n | avg fw | note |
|---|---:|---:|---:|---:|---:|---|
| **Permute (K‑transpose)** | **130.5 ms** | **97.4%** | 0.2 ms | 150 | 870 µs | pure NOC/DRAM move, ~0 math |
| BinaryNg | 1.5 ms | 1.1% | 1.3 ms | 1347 | 1.1 µs | |
| Unary | 1.3 ms | 1.0% | 1.2 ms | 213 | 6.2 µs | |
| Concat | 0.6 ms | 0.4% | 0.1 ms | 513 | 1.1 µs | |
| (LayerNorm / Slice / Softmax / Matmul …) | <0.1 ms each | | | | | |

Whole profiled run (incl. prefill+warmup) device compute for context: **Permute 415 ms (64.5%)**, **SparseMatmul = MoE experts 163 ms (25.4%, real FLOPs, kernel≈fw)**, ArgMax 30 ms (4.6%), rest <2%.

**Root cause:** `denoise_attention` (`tt/diffusion_attention.py:295`) materializes Kᵀ with a standalone
`ttnn.permute(k_group, (0,1,3,2), memory_config=DRAM_MEMORY_CONFIG)` **per KV-group, per layer** — ~0 math, ~870 µs of pure NOC/DRAM data movement each, ×150 in the step. It's a hand-rolled attention (GQA head clone+concat → permute Kᵀ → matmul → softmax → matmul), every intermediate round-tripping DRAM.

### Gap #2 — inter-op (dispatch/host) gaps
Eager execution is **dispatch-bound**: `OP TO OP LATENCY` is **54% of wall-time inside the denoise region** (160 ms gap vs 134 ms device-fw) and **81% whole-run** (2752 ms gap vs 644 ms fw). This is exactly what the Metal trace removes — the same 2-layer step **traced = 331 ms** vs eager ~3.4 s. Worst host-side stalls (eager): `Tilize` (avg 41 ms gap/op), `ReshapeView` (743 µs), `Embeddings` (42 µs).

### Takeaway / next
For the **traced** production path (gaps already gone), the single dominant device cost is the **attention Kᵀ permute (97% of per-layer fw)** — near-pure data movement from one line. Optimizing now: eliminate the explicit Kᵀ materialization (fuse QKᵀ via SDPA / keep in L1, drop the GQA clone+concat DRAM round-trips). Artifacts + repro scripts under `models/experimental/diffusion_gemma/doc/optimize_perf/` (`tracy_perop_*.txt`, `tracy_agg_*.py`).
