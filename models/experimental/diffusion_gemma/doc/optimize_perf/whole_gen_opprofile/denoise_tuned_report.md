## DiffusionGemma 26B-A4B — denoise op-level profile (latest **tuned true-sparse MoE**)

Full op-level re-profile of the denoise step on the **current** code (tuned true-sparse
capacity-dispatch MoE), superseding the earlier `SparseMatmul + cumsum-Permute` breakdown —
which no longer reflects how denoise runs.

**Setup.** Blackhole QB2, `(1×4)` TP mesh, tuned config (`DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1`,
HiFi2 experts, `DG_DEDUP_ARGMAX=1`), canvas 256, tracy device profiler, eager (`--no-trace`).

**Method.** The full-30-layer device log is ~39 GB and cannot be post-processed on the shared
box (the `earlyoom` watchdog SIGTERMs the pandas pass). So the denoise step is profiled at
**2 and 6 layers** (small, earlyoom-safe logs), each op's device-FW time is **linearly
decomposed** into a fixed (per-step) + per-layer term, and **projected to 30 layers**:
`t_op(L) = a_op + b_op·L`. Cross-checked against the **directly-measured** 30-layer step time.

---

### 1. Headline — the denoise op profile has changed

The true-sparse MoE (gather active experts → **batched Matmul** → scatter) replaced the dense
`SparseMatmul` path in denoise, and the giant `cumsum(dim=2)` **Permute FW-artifact is gone**
(Permute is now **1.8%**, was ~46%). There is **no single dominant op** anymore.

| metric | value |
|---|---:|
| measured 30L denoise step (eager) | **869 ms/step** (TTFT 1893 ms) |
| 2L / 6L step (eager) | 89 ms / 195 ms → ~26.5 ms/layer + ~36 ms fixed |
| projected 30L denoise **device-FW** | **276 ms/device** |
| ⇒ on-device compute share of wall-clock | **~32%** (the other ~68% is eager dispatch/host overhead, closed by trace in serving → ~7 t/s) |

---

### 2. Denoise op breakdown (30L projected, device-FW ms/device)

![op breakdown](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/denoise_op_breakdown.png)

| op | 30L device-FW ms/dev | share | scaling |
|---|---:|---:|---|
| Matmul (MoE experts + attn proj) | 96.9 | 35.1% | per-layer |
| BinaryNg | 30.0 | 10.9% | per-layer |
| Slice | 21.0 | 7.6% | per-layer |
| Unary | 20.1 | 7.3% | per-layer |
| LayerNorm | 17.1 | 6.2% | per-layer |
| ArgMax (token select) | 11.7 | 4.2% | **per-step (fixed)** |
| Reduce | 11.3 | 4.1% | **per-step (fixed)** |
| Scatter | 7.4 | 2.7% | per-layer |
| AllGather | 7.3 | 2.6% | per-layer |
| Concat / Tilize / Untilize | ~17 | ~6% | per-layer |
| Permute | 5.1 | 1.8% | per-layer |

### 3. By function — ~half of denoise is "glue", not matmul

![time share](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/denoise_time_share.png)

- **MoE + attention Matmul — 35.1%**
- **Layout / glue** (slice, concat, tilize, untilize, permute) — **27.9%**
- **Elementwise / reduce** (binary, unary, reduce) — **22.2%**
- LayerNorm 6.2% · TP collectives 4.3% · diffusion token-select (ArgMax) 4.2%

> **Takeaway:** the compute core (Matmul) is only ~35%; ~50% of denoise device time is data
> movement + elementwise **around** the capacity-dispatch MoE. The next optimization frontier is
> the gather/scatter/layout glue, not the matmul itself.

### 4. Layer scaling → 30L projection

![layer scaling](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/layer_scaling.png)

Per-layer ops (Matmul, LayerNorm, collectives, layout) scale with L; per-step ops (ArgMax
token-select, final Reduce) are fixed. `★` = the 30-layer projection.

### 5. Per-phase device time

![phase comparison](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/phase_comparison.png)

---

### 6. Memory — DRAM is 89% MoE experts

![dram by component](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/dram_by_component.png)

| component | per-chip DRAM (GiB) | share |
|---|---:|---:|
| MoE experts (128×) | 11.60 | **88.6%** |
| Attention | 0.54 | 4.1% |
| Token embedding | 0.34 | 2.6% |
| LM head | 0.34 | 2.6% |
| MoE router | 0.02 | 0.2% |
| **total** | **13.10** | of 31.87 GiB usable/chip |

All 128 experts are resident (A4B = 4B *active*, but 26B *resident*). This is why the full-30L
tracy profiler buffer OOMs: 13 GiB weights + trace region + a large op-support buffer > 32 GiB.

### 7. Device grid utilization (denoise)

![core util](https://raw.githubusercontent.com/tenstorrent/tt-metal/diffusion-gemma-function/models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile/figs/denoise_core_util.png)

Most denoise ops light up ~110 / 130 worker cores (~85% of the grid); the
sharded↔interleaved layout conversions only use ~77 (60%) — consistent with the "glue" cost above.

---

### Reproduce

```bash
# tuned denoise profile (per-layer count N ∈ {2,6}); full 30L device log is too big for -p
cd tt-metal-tracy
PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole \
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
python -m tracy -r -p -v --op-support-count 20000 \
  models/experimental/diffusion_gemma/doc/optimize_perf/prof_denoise_step.py \
  --num-layers 6 --canvas-length 256 --iters 1 --no-trace --commit-tokens 4

# charts
python .../whole_gen_opprofile/build_report.py OUT 869 <csv2>:2 <csv6>:6   # op breakdown
python .../whole_gen_opprofile/build_mem.py    OUT <30L-load-log> <csv6>   # DRAM + util

# interactive (ttnn-visualizer): point --performance-path at the tracy report dir
ttnn-visualizer --performance-path generated/profiler/reports/<ts>
```
