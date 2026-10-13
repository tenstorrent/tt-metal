# WIP plan: `math_thread_row_sums`: benchmarks, docstring, reply to review

Temporary file: delete before the PR merges. Answers pavlejosipovic's third review comment on PR #59695.

## The review comment

The reviewer asked:
1. Which exact workload and configuration make the MATH-thread row sums faster: Q/K/V shape, chunk sizes, `reuse_kv`,
   math fidelity, `math_approx_mode`, `fp32_dest_acc_en`? The PR gives Blackhole latencies but not the fidelity.
2. Matched PACK-vs-MATH profiles at the same settings, ideally MATH/PACK busy or stall cycles.
3. Why Blackhole only? Blackhole has roughly 2× Wormhole's packer bandwidth, so why does moving work *off* PACK help
   there? The reviewer's premise: SDPA originally moved the row reduction onto PACK because MATH was the bottleneck.

## Decision (already made)

`math_thread_row_sums` **stays opt-in**. Whether it helps depends on fidelity and shape. It also costs some accuracy
wherever it's on: one 16-bit DEST accumulator per row, compared with the pack thread's L1 sums (SDXL PCC
0.9972 → 0.9968). The work below supplies the numbers for the docstring and the reply. It doesn't change the
default.

## What's already known

Sources: `amorrison/high-batch-optim`, `models/demos/blackhole/pplx_embed_4b/doc/NEGATIVE_RESULTS.md` §65 and the
`doc/POSITIVE_RESULTS.md` row "batched SDPA row sums on the math thread (09-30)".

- **Workload:** pplx-embed-4b batched prefill.
  - Q `[B, 32, 512, 128]`, K/V `[B, 8, 512, 128]`, bfp8, non-causal, `output_heads_concat=True`.
  - `reuse_kv=True`, q128/k512 (one K chunk). Grid 12×8 at bs8, 12×10 at bs16/32.
  - **LoFi**, `math_approx_mode=False`, `exp_approx_mode=True`, `fp32_dest_acc_en=False` (streaming kernel),
    `packer_l1_acc=True`.
- **Per-thread zones** (bs8 shape, compute only, one q128 unit). The per-row-group figures cover 8 score tiles:

  | | PACK row sums (before) | MATH row sums (after) |
  |---|---|---|
  | MATH Q·Kᵀ 2×4 subblock | ~690 | ~690 |
  | MATH `SUB` (mostly the `tile_regs_acquire` wait on PACK) | 725–750 | 265–420 |
  | PACK `EXP` (SFPU issued from the pack thread) | 500–540 | 500–540 |
  | PACK `SUB_EXP` packs (incl. 8 L1-acc row-sum packs) | 455–465 | 121–168 |
  | normalize (per unit) | 2,887 | 3,615 |
  | **unit total** | **20,302** | **18,766** |

  Probes (wrong output): no row-sum packs −11%, no exp −19%.
- **History:**
  - `666d33c8bc6` (2025) put the partial row sums on PACK via L1 accumulation. That kernel ran `exp_tile` on the
    **math** thread (`compute_common.hpp:398`).
  - The streaming kernel (`08a5992cf1d`, 2026-03) moved exp onto the **pack** thread (`exp_packthread_tile`,
    `compute_streaming.hpp:614`). Since then PACK carries exp *and* the row-sum packs.
- **Hypothesis to confirm:** which thread paces depends on MATH's matmul cycles per score tile vs. PACK's exp and
  row-sum cycles per score tile.
  - LoFi, `d=128` (Q·Kᵀ inner dim 4 tiles), approximate exp: PACK is the bottleneck. Measured.
  - HiFi2/HiFi4 (about 2×/4× matmul cycles), larger `d`: MATH is likely already the bottleneck, so the extra
    normalize work lands on it and the flag loses. Not measured.
  - `exp_approx_mode=False`: costlier exp on PACK, which pushes back toward PACK being the bottleneck. Not measured.
- **Blackhole only, as a matter of scope:** the batched normalize matmul (`matmul_block` with `rt_dim = sbh`,
  `kt_dim` = row stride, then a 1×1 re-init to avoid the V-matmul hang) was written and validated against Blackhole's
  LLK only. Wormhole has its own no-MOP matmul (`bb3abd495f3`), so a port is plausible but unmeasured. Packer bandwidth
  isn't the limit: the removed cost is pack-*thread* time (SFPU exp issue plus ~57-cycle single-tile L1-acc packs), not
  bytes.
- **Speedups recorded so far disagree:**

  | Source | bs8 | bs16 | bs32 |
  |---|---|---|---|
  | §65 | −5.9% | −7.0% | −5.7% |
  | PR body (K/V in L1: 202.9 / 312.8 / 570.6 → 188.1 / 286.8 / 528.5 µs) | −7% | −8% | −7% |
  | nanobind docstring | 6% | 7% | 7% |

  Re-measure and use one set everywhere.

## Benchmarks

### Tooling

On `amorrison/high-batch-optim` under `models/demos/blackhole/pplx_embed_4b/perf_tools/`:
- `bench_common_traced.py`: traced device time.
- `bench_sdpa_floors.py`: the model's reuse_kv q128 config, all-DRAM / K-V-L1 / all-L1 placements.
- `bench_sdpa_zones.py`: per-unit device-profiler zones, plus `--parse`.
- `sdpa_kernel_variants.py`: patched kernel trees, `zones` / `zconly`.

Check them out into the working tree uncommitted (`git checkout origin/amorrison/high-batch-optim -- <paths>`).
Run them from a directory with no `ttnn/` tree (kernel lookup tries the cwd first), with one `TT_METAL_CACHE` per
variant.

**Porting needed before the runs.** The variants script patches the kernels as they were *before* this PR's refactors:
`REUSE_KV` and `SDPA_SUM_ON_PACK` were defines then. Now they're compile-time args, and the define is gone.
- [ ] Drop the `sumpack` variant. `math_thread_row_sums=False` is now the same-binary control.
- [ ] Check that the `zones` / `zconly` / `nosum` / `noexp` regex patches still apply to the current
      `compute_streaming.hpp`. Fix the anchors where they don't.
- [ ] Update the `DM_STUB` reuse detection: it reads `#ifdef REUSE_KV`, which should become the compile-time arg.
- [ ] Add CLI arguments to `bench_sdpa_floors.py` / `bench_sdpa_zones.py` for `math_thread_row_sums`,
      `math_fidelity`, `exp_approx_mode` and the shape (`B`, `NQH`, `NKH`, `S`, `d`, q/k chunk, grid).

### Matrix

Every arm runs flag off vs. flag on, in the same binary. For each, record traced µs/call (cold clock, same chip,
alternating rounds) and, from `zconly`, the per-thread zones: MATH `SUB` / acquire wait, PACK `EXP`, PACK `SUB_EXP`,
normalize, unit total.

**A. pplx-embed-4b (the motivating workload).** q128/k512, `reuse_kv=True`, `exp_approx_mode=True`, bs8 on 12×8,
bs16/32 on 12×10, K/V in L1 (bs8/16) and all DRAM (bs32).
- [ ] LoFi: re-measure the speedup and settle the inconsistent numbers above.
- [ ] HiFi2: tests whether MATH takes over as the bottleneck.
- [ ] HiFi4.

**B. Head dimension**, at the pplx batch-16 shape with LoFi. Change `d` only, keeping one K chunk.
- [ ] `d` = 64, 128, 256. Larger `d` means more Q·Kᵀ cycles per score tile on MATH.

**C. Exp mode.**
- [ ] pplx bs16, LoFi, `exp_approx_mode=False`.

**D. SDXL 1024×1024 on Blackhole** (`models/demos/stable_diffusion_xl_base/tt/model_configs/model_configs_1024x1024BH.py`,
`tt_attention.py`). These are real single-K-chunk shapes at the accuracy-sensitive end. Grid 11×10, q128,
`exp_approx_mode=False`, `math_approx_mode=False`, `fp32_dest_acc_en=False`.
- [ ] Self-attention in down/up blocks 2: 1024 tokens, k1024 (one K chunk), 20 heads, `d=64`, **LoFi** (the model's
      setting). Confirm the batch (CFG) in the model.
- [ ] The same shape at **HiFi2**. The model uses HiFi2 for the down-block-1 / up-block-0 self-attention, but those
      blocks run 4096 tokens with k512, so the flag doesn't apply there.
- [ ] Cross-attention: 77-token K, k128 (one K chunk), LoFi.

**E. Batch 1 (pplx).**
- [ ] q256/k512 on 8×8 and packed q192/k512 on 11×8, LoFi. Does the gain hold with about one unit per core?

**F. Wormhole.** Out of scope unless the reply should argue for or against a port. If so, it would mean porting the
batched normalize to the WH no-MOP matmul first.

### Accuracy (each arm in A–D)

- [ ] PCC and max error vs. fp32 torch, flag off vs. on.
- [ ] Per-row denominator error statistics vs. `S`: the docstring's "S=1024 std 0.007 → 0.012" claim.
- [ ] SDXL module PCC tests at the model config with the flag forced on, to repeat the 0.9972 → 0.9968 data point:
      `test_module_tt_transformermodel`, `test_module_tt_crossattnmidblock2d`, `test_module_tt_crossattnupblock`.

### Expected outcome

Two tables:
1. the speedup by (fidelity, `d`, exp mode);
2. per-thread zones at LoFi vs. HiFi2 showing where the bottleneck moves.

If HiFi2 doesn't move it, revisit the hypothesis before writing the reply.

## Docstring update (`sdpa_nanobind.cpp`, `math_thread_row_sums`)

Keep: opt-in, the eligibility conditions (Blackhole, `fp32_dest_acc_en=False`, one K chunk, no attention sink;
ignored otherwise) and the accuracy paragraph. Change:
- [ ] **When it helps, by regime:** low fidelity (LoFi), small head dim, approximate exp, i.e. when the pack
      thread (which runs the exp) paces the softmax. At HiFi2/HiFi4 or large head dims the math thread is likely
      already the bottleneck and the flag is neutral or slower. Fill in from tables A–C.
- [ ] **Speed numbers** with their full config: "Blackhole, LoFi, `exp_approx_mode=True`, 32/8 heads, s=512,
      d=128, `reuse_kv` q128/k512, batch 8 / 16 / 32: X / Y / Z%". Use one measurement set.
- [ ] Optionally one line on the HiFi2 result, e.g. "at HiFi2 the same shape is ±N%".
- [ ] Also update the PR description: state the fidelity and config, and use the same speedup numbers as the
      docstring.

Draft (fill in the placeholders):

> math_thread_row_sums (bool): Defaults to `False`. Blackhole, `fp32_dest_acc_en=False`, a single K chunk
> (`k_chunk_size >= s`), no `attention_sink`; ignored otherwise. Compute each softmax denominator from the exp'd
> scores with a matmul on the math thread instead of accumulating it on the pack thread. The streaming kernel runs the
> exp on the pack thread too, so at low fidelity, where the math thread's Q·Kᵀ is cheap, the pack thread paces the
> softmax and this helps. At higher fidelity or larger head dims the math thread is already the busier one and this
> does not help. Speed (LoFi, `exp_approx_mode=True`, 32/8 heads, s=512, d=128, `reuse_kv` q128/k512, batch
> 8 / 16 / 32): {X} / {Y} / {Z}% faster per call; {HiFi2 result}. Accuracy: {existing paragraph}. Turn it on for
> throughput-bound, low-fidelity batched prefill whose accuracy you have checked with it on (e.g. embedding models);
> leave it off for models with tight PCC margins or at HiFi2 / HiFi4.

## Reply to the review comment (draft, fill in after the runs)

> Thanks, it's a fair question. The answer depends on fidelity, which the PR should have stated.
>
> **Workload / config:** pplx-embed-4b batched prefill on Blackhole P150. Q [B, 32, 512, 128], K/V [B, 8, 512, 128],
> bfp8, non-causal, `reuse_kv` with q128/k512 (one K chunk), 12×8 at bs8 and 12×10 at bs16/32. **LoFi**,
> `math_approx_mode=False`, `exp_approx_mode=True`, `fp32_dest_acc_en=False` (streaming kernel), `packer_l1_acc=True`.
> Speedup: {X / Y / Z%} at bs8 / 16 / 32.
>
> **Why PACK and not MATH here:** the earlier move of the row sums to PACK (#22312) was made when the kernel ran exp
> on the math thread. The streaming kernel (#38838) moved exp onto the pack thread for FPU/SFPU overlap, so PACK now
> carries the exp and the L1-accumulated row-sum packs. Zones for one row group (8 score tiles) at this config: MATH's
> Q·Kᵀ subblock takes ~690 cycles; PACK spends 500–540 on exp plus 455–465 on the row-sum packs before it packs the
> next half, so MATH waits 725–750 cycles in `tile_regs_acquire`. With the sums on MATH, PACK's packs drop to 121–168
> and MATH's wait to 265–420; normalize grows 2,887 → 3,615 cycles per unit, and the unit drops 20,302 → 18,766.
> {Matched zone table, flag on / off, LoFi and HiFi2.}
>
> **Fidelity:** at HiFi2 {result: e.g. MATH's Q·Kᵀ roughly doubles, MATH becomes the bottleneck, and the flag is
> ±N%}. That is why it is opt-in rather than automatic, along with the accuracy cost (one 16-bit DEST accumulator per
> row; SDXL PCC 0.9972 → 0.9968). The docstring now says when to use it.
>
> **Blackhole only:** a scope choice, not an architectural claim. The batched normalize matmul relies on Blackhole's
> `matmul_block` behaviour (`rt_dim` over the row group, `kt_dim` as the row stride, and a 1×1 re-init afterwards) and
> was only validated there. Packer bandwidth isn't what this relieves: the removed cost is pack-thread time, issuing the
> SFPU exp and ~57-cycle single-tile L1-accumulate packs, not packed bytes. A Wormhole port on the WH no-MOP matmul is
> plausible but unmeasured. {Or: WH result if run.}

## Results (fill in on hardware)

| Arm | Shape | Fidelity | Exp approx | Flag | µs/call | Δ | MATH wait | PACK EXP | PACK SUB_EXP | Normalize | Unit | PCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
