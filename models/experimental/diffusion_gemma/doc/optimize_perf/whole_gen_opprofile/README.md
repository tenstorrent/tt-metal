# Op-level device profile of a whole generation (#47465)

Status: current for the profiling method and the attribution traps; **provenance-only for every op
mix**, all captured on MoE paths since deleted (dense-128, then token-gather).
Owns: the sum-of-device-FW overlap trap; the reduced-layer (2L/6L) projection method and the profiler
op-buffer ceiling that forces it; the per-device normalization of mesh device-FW; the attention
K-transpose Permute and its removal. Absorbs the deleted `denoise_tuned_report.md`,
`moe_transpose_investigation.md` and `tracy_perop_47465_comment.md`.
See also: [refuted list](../../REFUTED.md), [optimize_perf hub](../README.md).
Over the 100-line cap: one op-profile document for three merged files, carrying two open contradictions,
six traps and two repro pins, none cut for length.

## The attribution trap — read this before quoting any op share

On the `(1,4)` mesh the per-op device-FW windows **OVERLAP ~1.5–1.74x** (concurrent programs), so
**sum-of-device-FW EXCEEDS wall time and is valid only for RELATIVE op share, never as a cost.** Two
headline op shares in this tree are instances:

- **42.1% `PermuteDeviceOperation` [6-D `{0;3;2;1;4;5}`]** — an FW-overlap artifact worth ~1% of real
  time: eager per-op puts `cumsum(dim=2)` at **0.178 ms** against the full dispatch at 0.809 ms and ONE
  expert matmul at **4.17 ms**. Origin: `ttnn.cumsum(mask, dim=2)` in `build_capacity_dispatch` — a
  cumsum along a non-last dim tilizes as permute → cumsum → permute, emitting 4 Permutes per call. It is
  **NOT** the expert-group reshape: `ttnn.reshape [1,1,EC,H] <-> [1,E,C,H]` is a zero-copy metadata view,
  no device op, eager 0.006 / 0.003 ms. Established by `ttnn.graph` op-capture with fast-runtime-mode OFF
  plus eager per-op timing at real shapes; probes at
  `~/dg-agent-runs/{moe_graph_probe,dispatch_graph_probe,cumsum_microbench}.py`.
- **97.4% attention K-transpose `Permute`** — `denoise_attention` materialized Kᵀ with a standalone
  `ttnn.permute(k_group, (0,1,3,2), memory_config=DRAM)` per KV-group per layer, ~0 math and ~870 µs of
  pure NoC/DRAM movement each, x150 per step = **130.5 ms = 97.4% of per-layer device-fw**. Removed by
  fusing Kᵀ into `matmul(transpose_b=True)`, commit **`981820808bc`**; the denoise path now uses that
  fusion (`diffusion_attention.py:331-333`) and the per-phase profile records the denoise Permute at
  **1.8%**.

> **OPEN CONTRADICTION (unexplained):** the K-transpose Permute is answered three ways — a real dominant
> cost worth removing (`tracy_perop_47465_comment.md`, merged here); the archetype of an FW-overlap
> artifact, "the old Kᵀ 97% permute that gave 0% when optimized" (this file's own correction); and a
> real, bit-exact win but **only on the SDPA fallback path** (`moe_transpose_investigation.md`, merged
> here). The three cannot all be right and none was reconciled.

**VALIDATED MICRO-LEVER, deliberately not landed.** Replacing `cumsum(dim=2)` with a lower-triangular
matmul `L[S,S] @ mask[S,E]` (inclusive cumsum, zero permutes) is **0.0149 ms, 92% faster, PCC 0.999995**,
bit-exact because counts <= 256 are exact in bf16 — but it saves ~0.16 ms/layer, ~0.24 s/block out of
~200 s, i.e. ~0.1% end-to-end and ~0% under traced overlap, and the dispatch feeds integer routing-column
indices where a dtype slip is a correctness bug.

## The expert-major transpose is necessary data movement (negative result)

**REFUTED:** replacing `ttnn.transpose(gate,1,3)` / `(up,1,3)` in `Gemma4Experts.prefill_forward` →
`_process_prefill_chunk` with a zero-copy reshape is bit-exact but a perf **WASH**, and was reverted:
`PermuteDeviceOperation` dropped 130.4 → 8.9 ms (n 150 → 54) but `UnaryDeviceOperation` rose 1.3 → 122.7
ms at the same n=213, **total device-fw unchanged (133.9 → 133.8 ms)**, `eager_ms_per_step` unchanged
(339.30 → 339.26 ms at 2 layers).

**GENERAL TRAP:** `ttnn.reshape` here is a **LAZY VIEW** — the reorder the transpose was doing is not
eliminated, only deferred to the consuming op (the GeGLU `Unary`), so total work is conserved and a
Python transpose→reshape rewrite **relocates** data movement rather than removing it. (It was legal at
all because for the denoise canvas every prefill chunk is exactly one tile group — `chunk_len ==
TILE_SIZE == 32` ⇒ `group_size == 1` — so the swapped dims have size-1 neighbours; verified PCC=1.0,
max|diff|=0 by `diag_moe_transpose.py`. `diag_verify_moe.py`, cited as the full-MoE verifier, is not
present.) **CONCLUSION:** the transpose is intrinsic (the expert-major reorder feeding the down
`sparse_matmul`), so a real reduction needs a kernel-level change — a `sparse_matmul` variant emitting the
down-projection's layout, or a fused MoE — which lives in shared `ttnn`/gemma4, outside the
no-shared-edits scope. The investigated path is no longer default for either phase: denoise is
`tt/concat_moe.py`, prefill is the ragged zero-drop top-8 path.

## Method — reduced-layer 2-point fit

**HARD PROFILER CONSTRAINT:** the on-device profiler op buffer (`PROGRAM_SUPPORT_COUNT`, default 1000 →
~3.3k ops captured) **cannot hold a 30-layer forward (~30k ops)** — a direct `--num-layers 30` Tracy run
silently drops device timing after the first contiguous ~3.3k ops (verified as prefill plus under one
denoise step, with all 67k commit ops empty). The buffer is raised with
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`, which sizes **DRAM and not L1**, so it is safe at low layer
counts. Separately, full-30-layer device logs are ~39 GB and cannot be post-processed on the shared box —
the `earlyoom` watchdog kills the pandas pass.

So each repeating unit is profiled at two small layer counts that fit the buffer, linearly extrapolated
to 30 layers, and composed: `whole_gen = prefill(30L)*1 + denoise_step(30L)*48 + commit_token(30L)*256`.
**The 2-point fit is well-conditioned** because op **counts** scale exactly linearly (SparseMatmul x3
from 2 to 6 layers, ArgMax constant = per-step overhead) and at this short context (prompt + canvas ≈ 288
< the 1024 sliding window) sliding- and full-attention layers attend to identical positions, so per-layer
cost is kind-independent; the 6-layer point still includes a full-attention layer (layer 5) as a check.

**Device-FW is summed over the 4-device mesh and reported PER DEVICE** — state this or the numbers are 4x
off. **The trustworthy per-phase metric is the device-busy SPAN** (max FW-end minus min FW-start, cycles
/ 1.35 GHz AICLK), validated against warmed wall clock exactly: denoise 2L span 352 ms vs wall 352.6 ms,
6L 916.9 vs 917.5 ms, commit 6L 110.3 vs 110.8 ms.

**HONESTY RULES.** Eager-under-Tracy timing exists only to attribute per-op device time and the span
figures are profiler-inflated — the actual serving speed is the traced path, never these spans. And the
prefill span is **COLD** (JIT-compile-polluted, uncached, no warm-up before the measured region): use
prefill device-FW (~1 s one-time, negligible), never its span.

### Runs and reproduction (env: see [plan](../../../plan.md))

From the Tracy build worktree `/home/zni/tt-metal-tracy` — the Tracy-enabled tt-metal (`ENABLE_TRACY=ON`)
is built in a git worktree so the shared `build_Release` is untouched — via
`python -m tracy -r -p -v --no-trace`:

| run | cmd | buffer | coverage |
|---|---|---|---|
| 2L | `prof_denoise_step.py --num-layers 2 --canvas-length 256 --iters 1 --commit-tokens 8` | 8000 | prefill/denoise 100%, commit 94.6% |
| 6L | `prof_denoise_step.py --num-layers 6 --canvas-length 256 --iters 1 --commit-tokens 4` | 22000 | 100% all phases |

`--commit-tokens` exists so the commit phase fits the buffer (256 single-token decode-appends would be
over 200k ops); each commit token is an independent single-token decode, so per-token cost is
canvas-independent and is scaled x256. The full per-phase invocation adds
`PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000` and `--op-support-count 20000`; the
`DG_SPARSE_MOE`/`DG_SPARSE_MOE_TUNED`/`DG_DEDUP_ARGMAX` flags it pinned at measurement time no longer
exist. **SIGNPOST REPRO** for the per-step region: `prof_denoise_step.py --num-layers 2 --canvas-length
256 --iters 3 --no-trace`, then `python -m tracy -r -p`; the enriched report carries the
`DENOISE_START..DENOISE_END` signposts so the region is isolated exactly, and all numbers are device 0
(TP=4, per chip). **REGENERATION:** `compose_whole_gen_opprofile.py --csv2 <csv> --n2 2 --ncommit2 8
--csv6 <csv> --n6 6 --ncommit6 4`; the chart builders `build_report_full.py` and `build_mem.py` named by
the merged per-phase report are **not present**. Interactive view: `ttnn-visualizer
--performance-path generated/profiler/reports/<ts>`.

**ARTIFACTS:** `whole_gen_op_breakdown.txt`, `whole_gen_summary.json` and `figs/` are committed; the
multi-GB raw `ops_perf_results*.csv` are intentionally not, and the `phase_op_agg_{2L,6L}.csv` listed by
earlier versions of this file are **not present** either. The Tracy per-op tables
`tracy_perop_denoise_region.txt` / `tracy_perop_wholerun.txt` and the aggregators
`tracy_agg_{denoise_region,enriched,full,signpost}.py` live under `doc/optimize_perf/`.

## Dispatch is what tracing removes

Eager execution is dispatch-bound: `OP TO OP LATENCY` is **54% of wall time inside the denoise region**
(160 ms gap versus 134 ms device-fw) and **81% whole-run** (2752 ms gap versus 644 ms fw); the same
2-layer step is **331 ms traced versus ~3.4 s eager**. Worst host-side eager stalls, if dispatch ever has
to be attacked again: `Tilize` at an average 41 ms gap per op, `ReshapeView` 743 µs, `Embeddings` 42 µs.
Consistent with the per-phase measurement that ~68% of the eager denoise step is dispatch overhead trace
removes (eager 720 ms → traced 233 ms/step).

## Per-phase op mix (provenance — captured on the deleted token-gather MoE)

**HEADLINE that still matters: the three block-generation phases run THREE DIFFERENT MoE code paths, so a
per-op profile of one phase says nothing about another.** Phase split by device-FW: **DENOISE 94.02%,
COMMIT 5.74%, PREFILL 0.24%** — the 48x denoise loop dominates, so the whole-generation op mix is
approximately the denoise-step op mix. Per-phase 30L device-FW per device: **PREFILL ~1034 ms**
(SparseMatmul 495.5 ms / 48% + the cumsum Permute FW-artifact 485.8 ms / 47%, ReduceScatter/AllGather
16.2 ms / 2%, Matmul/BinaryNg/Unary/LayerNorm ~25 ms / 3%; TTFT ~0.6 s for an 18-token prompt);
**DENOISE ~276 ms per step**;
**COMMIT ~376 ms for a 4-token probe** (Transpose 35% + SparseMatmul 34%). Prefill now defaults to the
ragged zero-drop top-8 path, so neither the dense path nor that Permute artifact is on it.

| DENOISE op at 30L | ms/dev | share | scaling |
|---|---:|---:|---|
| Matmul (MoE experts + attn proj) | 96.9 | 35% | per-layer |
| BinaryNg / Unary / Reduce | 61.4 | 22% | per-layer + per-step |
| Slice / Concat / Tilize / Untilize / Permute (layout glue) | ~77 | 28% | per-layer |
| LayerNorm | 17.1 | 6% | per-layer |
| TP collectives | 11.9 | 4% | per-layer |
| ArgMax | 11.7 | 4% | **per-step (fixed)** |

**By function: Matmul 35% / layout-glue 28% / elementwise-reduce 22%** — the compute core is only ~35%
and ~50% is data movement plus elementwise. That is the finding that motivated the norm and glue work.

**COMMIT:** the live default is the **batched** commit (`select_commit_fn`, default on since 2026-07-04)
— the 256 canvas tokens are written into the KV cache as ONE causal prefill-append reusing the MoE,
roughly one denoise-step-equivalent, verified torch-correct and **24.8x faster** than the legacy path
(`verify_commit_batching` 35.1 s → 1.41 s at 30L). The commit chart in `figs/` is the **legacy sequential
per-token decode-append**, kept only as the profiled reference for what a single-token decode costs.

**DRAM CENSUS:** per-chip weight DRAM is **13.1 GiB of 31.87 usable** — the 128 MoE experts are **88.6%
(11.6 GiB)**, attention 4.1%, embedding and LM-head 2.6% each; all 128 experts are resident (A4B = 4B
active, 26B resident). **GRID UTILIZATION:** most denoise ops light up ~110 of 130 worker cores (~85% of
the grid) while the sharded↔interleaved layout conversions use ~77 (~60%), consistent with the
layout-glue cost. **SETUP for this section:** Blackhole QB2, `(1x4)` TP mesh, HiFi2 experts, canvas 256,
tracy device profiler, eager, signpost-segmented.

> **OPEN CONTRADICTION (unexplained):** this profile's honesty note asserts "the model runs the full 48
> denoise steps; HF early-halt is a no-op under #48291", while
> [winter borrow](../winter_borrow_20260727.md) measured the halt firing at `denoise_steps_per_block =
> [9, 2, 2]` and other reports measured `[9,17,2]/48` and `K=10–43`. Not explained.

## Two corrections that outlive their subject

**PROFILING-CONFIG TRAP.** The whole-gen op mix was captured by `prof_denoise_step.py`, which does not set
the tuned-MoE flag, so it profiled the auto-config matmul path where the gate/up expert matmul is 13x
slower than what production ran — both #47465 headliners were therefore untuned or artifact. The
tuned-versus-auto numbers are owned by [OPT-004](../opt004_matmul_geometry.md) (measured here: gate matmul
4.17 → 0.318 ms; full MoE 10.06 → 2.90 ms/layer, 3.47x, PCC 0.99967).

**CAPACITY TRAP.** `prof_step_breakdown.py` hard-coded `capacity=32` for its isolated MoE row while the
real denoise layer used the zero-drop production default `capacity=256`. Capacity 32 is not a valid
production comparator — real routing can load one expert with 156–256 canvas tokens, silently dropping
41–84% of routed assignments. With the profiler fixed to the effective capacity and a route-drop assertion
added, the zero-drop MoE measures **9.11–9.21 ms/layer** (~61% of the contemporaneous 14.8 ms layer)
rather than 2.63 ms / 18%, and **the claim that the residual 11.8 ms was attention+CCL is WITHDRAWN**.

**The compact-ragged result.** It reduces the selected router+MoE component 9.63 → 5.62 ms/layer and the
full 30L traced fixed-48 block 19.7898 → 14.2743 s (12.936 → 17.934 tok/s, **+38.6%**), but stayed off by
default because its changed expert matmul geometry produces a different committed trajectory against a red
strict HF gate. The **EXACT** compact mode (reduction-compatible expert K blocks, baseline combine matmul
contract restored) reproduces the baseline committed SHA and the pre-change seed-0/1 committed agreements
while keeping a smaller gain: 19.7827 → 18.8807 s/block, 12.941 → 13.559 tok/s (**+4.8%**); the +38.6%
fast-reduce mode is diagnostic only.

The "~14.3 s/block → 17.9 t/s at 48 traced steps" wall clock quoted by the merged per-phase report is
superseded by the concat MoE and the full-canvas norm — [winter borrow](../winter_borrow_20260727.md),
[l1 residency](../l1_residency.md); the single current per-step cost is owned by the
[optimize_perf hub](../README.md). The ~235 GB/s denominator is owned by the
[non-MoE roofline](../nonmoe_roofline/README.md) and the weight-byte floor by the
[work log](../work_log.md).
