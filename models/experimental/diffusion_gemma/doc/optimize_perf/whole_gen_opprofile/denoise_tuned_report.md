## DiffusionGemma 26B-A4B — full per-phase op-level profile (latest tuned config)

Complete op-level breakdown of **all three block-generation phases — PREFILL, DENOISE, COMMIT** —
on the current code (tuned true-sparse MoE), superseding the earlier denoise-only comment.

**Setup.** Blackhole QB2, `(1×4)` TP mesh, tuned config (`DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1`,
HiFi2 experts, `DG_DEDUP_ARGMAX=1`), canvas 256, tracy device profiler, eager, signpost-segmented.

**Method.** Full-30-layer device logs are ~39 GB and can't be post-processed on the shared box
(the `earlyoom` watchdog kills the pandas pass), so each phase is profiled at **2 and 6 layers**
(small, earlyoom-safe), each op's device-FW time is **linearly decomposed** (fixed + per-layer) and
**projected to 30 layers**. Device-FW is summed over the 4-device mesh and reported **per device**.

---

### Per-phase overview

![phases](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/phase_comparison_30L.png)

| phase | 30L device-FW (ms/dev) | MoE path | dominant ops |
|---|---:|---|---|
| **PREFILL** (prompt) | ~1034 | **dense** gemma4 expert path | SparseMatmul 48% + Permute 47% |
| **DENOISE** (per step, ≤48/block) | ~276 | **capacity-dispatch** (true-sparse) | Matmul 35%, elementwise/glue ~50% |
| **COMMIT** (per-token seq probe) | ~376 (4-tok) | decode sparse-MoE | Transpose 35% + SparseMatmul 34% |

**The three phases run three different MoE code paths** — this is the headline. PREFILL uses the dense
gemma4 expert path (the old `SparseMatmul` + `cumsum`-`Permute` FW-artifact, still ~95% of prefill);
DENOISE uses the true-sparse capacity-dispatch path (Matmul-based, the Permute artifact **gone**);
COMMIT's live default is the **batched** single-prefill commit (≈ one denoise-step-equivalent).

Wall-clock note: at 48 traced steps a block is **~14.3 s → 17.9 t/s (model-faithful)**; ~68% of the
*eager* denoise step is dispatch overhead that trace removes (eager 720 ms → traced 233 ms/step).

---

### 1. PREFILL — dense path, SparseMatmul + cumsum-Permute dominate

![prefill](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/prefill_op_breakdown.png)

| op | ms/dev (30L) | share |
|---|---:|---:|
| SparseMatmul (dense experts) | 495.5 | 48% |
| Permute (cumsum FW-artifact) | 485.8 | 47% |
| ReduceScatter / AllGather (TP) | 16.2 | 2% |
| Matmul / BinaryNg / Unary / LayerNorm | ~25 | 3% |

Prefill runs the **gemma4 dense expert path**, so the `cumsum(dim=2)` **Permute artifact is still here**
(~47%) — the same artifact the true-sparse denoise path eliminated. Prefill happens once per prompt
(TTFT ≈ 0.6 s for an 18-token prompt), so it is not the steady-state bottleneck, but it is the place
the dense-MoE + Permute cost still lives.

### 2. DENOISE — the per-step unit (true-sparse capacity-dispatch MoE)

![denoise](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/denoise_op_breakdown.png)
![share](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/denoise_time_share.png)

| op | ms/dev (30L) | share | scaling |
|---|---:|---:|---|
| Matmul (MoE experts + attn proj) | 96.9 | 35% | per-layer |
| BinaryNg / Unary / Reduce (elementwise) | 61.4 | 22% | per-layer + per-step |
| Slice / Concat / Tilize / Untilize / Permute (layout glue) | ~77 | 28% | per-layer |
| LayerNorm | 17.1 | 6% | per-layer |
| TP collectives | 11.9 | 4% | per-layer |
| ArgMax (diffusion token-select) | 11.7 | 4% | **per-step (fixed)** |

By function: **MoE+attn Matmul 35% · layout/glue 28% · elementwise/reduce 22%.** The compute core is
only ~35%; **~50% of the denoise step is data-movement + elementwise around the capacity-dispatch MoE**
— the next optimization frontier. The old `SparseMatmul`/cumsum-`Permute` dominance is gone (Permute 1.8%).

![layer scaling](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/layer_scaling.png)

### 3. COMMIT — live path is batched (≈ one denoise-step); legacy sequential shown for reference

![commit](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/commit_seq_op_breakdown.png)

The **live default commit is batched** (`select_commit_fn`, default on since 2026-07-04): the 256 canvas
tokens are written into the KV cache as **one causal prefill-append reusing the sparse MoE — ≈ one
denoise-step-equivalent**, verified torch-correct and **24.8× faster** than the legacy path
(`verify_commit_batching` 35.1 s → 1.41 s @30L). So op-wise, the production commit ≈ the DENOISE
breakdown above.

The chart shows the **legacy sequential per-token decode-append** breakdown (the profiled reference):
Transpose 35% + SparseMatmul 34% (the gemma4 decode sparse-MoE). This is what each single-token
decode costs; the live batched path replaces 256 of these with one prefill.

---

### 4. Memory — DRAM is 89% MoE experts

![dram](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/dram_by_component.png)

Per-chip weight DRAM **13.1 GiB** (of 31.87 usable): MoE experts (128×) **88.6%** (11.6 GiB), attention
4.1%, embedding/LM-head 2.6% each. All 128 experts are resident (A4B = 4B *active*, 26B *resident*).

### 5. Denoise device-grid utilization

![core util](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/denoise_core_util.png)

Most denoise ops light up ~110 / 130 worker cores (~85% of the grid); the sharded↔interleaved layout
conversions use ~77 (~60%) — consistent with the "layout glue" cost above.

---

### Reproduce

```bash
# per-phase profile at N ∈ {2,6} layers (full-30L device log is too big for -p on the shared box)
cd tt-metal-tracy
PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole \
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
python -m tracy -r -p -v --op-support-count 20000 \
  models/experimental/diffusion_gemma/doc/optimize_perf/prof_denoise_step.py \
  --num-layers 6 --canvas-length 256 --iters 1 --no-trace --commit-tokens 4

# all-phase charts (PREFILL/DENOISE/COMMIT projections) + DRAM/util:
python .../whole_gen_opprofile/build_report_full.py OUT <csv2>:2 <csv6>:6
python .../whole_gen_opprofile/build_mem.py         OUT <30L-load-log> <csv6>

# interactive (ttnn-visualizer): --performance-path generated/profiler/reports/<ts>
```
