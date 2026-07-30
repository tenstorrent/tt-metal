# DG_NORM_FULLCANVAS default-flip gate — SUPERSEDED 2026-07-30, the flag SHIPPED and was DELETED

> **This gate's verdict was overturned, and the gate itself is now history.** The full-canvas norm is
> the only path since 2026-07-30; `DG_NORM_FULLCANVAS` no longer exists. Three things closed it:
>
> 1. **The premise was void.** This gate ran with `DG_SPARSE_MOE=1`, i.e. on the token-gather denoise
>    MoE deleted in `7417bd7d69d` because it does not let the trajectory converge at all — which is
>    exactly why it describes BOTH its arms as "coherent-then-degenerate". `committed_match = 0.145`
>    was measured between two arms sitting on a broken baseline.
> 2. **The magnitude was never measured.** "~2e-6/norm, PCC 0.999998" came from a bench that reports
>    PCC > 1.0 elsewhere in its own table. The real delta was 5.73 bf16 ULP — four orders of magnitude
>    larger — and its cause was ttnn's rmsnorm defaulting to bf16 partial accumulation, not the
>    `block_h` difference this document blames. With fp32 accumulation the two shapes are
>    **bit-identical**: 0 of 69,206,016 elements over 96 device slices.
> 3. **The full-scale evidence is the opposite of the small-scale evidence.** The "27% shorter answers"
>    was a 10-question artifact (−10% at 71, gone at 198). The 198-question run with the norm on scored
>    **71.21%** against **66.67%** for the previous full run on the same questions, with 0 empty
>    replies and 0 responses over the 2% non-Latin threshold.
>
> Everything below is the original 2026-07-27 gate, kept as the record of how it was decided the first
> time. Do not cite its numbers as current.

## Original gate (2026-07-27) — verdict at the time: KEEP OPT-IN

The dg-08 L1-residency pass landed `DG_NORM_FULLCANVAS` (full-canvas RMSNorm, +15.8% @48 traced)
**opt-in, default OFF**, because its output is not bit-identical to the chunked-norm default
(`l1_residency.md`). This gate answers: **should the default be flipped ON?** Rule (from the request):
flip only if the diffusion decisions **hold within the #48291 bar vs the current chunked-norm
default**; else keep opt-in.

## Verdict: KEEP OPT-IN (default OFF). The gate FAILS decisively.

`decision_agreement.py` (the dg-07/dg-05 harness that produced the bfp8 decision) run chunked (current
default) vs full-canvas on ONE injected-noise block, **everything pinned except `DG_NORM_FULLCANVAS`**:
30 layers, 16 denoise steps, seed 0, fixed seeded initial canvas + fixed per-step renoise tokens, CLEAN
ARGMAX sampling (gumbel_noise=None → deterministic), production MoE (`DG_SPARSE_MOE=1`,
`DG_SPARSE_MOE_TUNED=1`), non-degenerate prompt (~200 non-EOS committed tokens — not an all-EOS
constant-vs-constant trajectory).

| metric (full-canvas vs chunked default) | value | bar | rejected-bfp8 reference |
|---|---:|---:|---:|
| **committed clean-argmax match** | **0.145** | ≥ 0.95 | 0.227 (**full-canvas is WORSE**) |
| mean per-step Gumbel argmax agreement | 0.544 (min 0.144) | high | ~0.95 step-0 for bfp8 |
| mean accept/renoise IoU | 0.504 (min 0.0) | high | 0.501 (≈ identical) |
| mean per-step entropy PCC | 0.659 (min 0.259) | high | 0.631 (≈ identical) |
| mean sampled-canvas agreement | 0.889 (min 0.770) | — | — |

`committed_match = 0.145 ≪ 0.95` — **~85% of committed tokens differ from the current default**, and on
the two sensitive metrics (entropy PCC, accept IoU) full-canvas is statistically **indistinguishable
from the already-rejected bfp8 experts lever** (0.659/0.504 vs 0.631/0.501). So the decisions do NOT
hold within the #48291 bar vs the chunked default → **keep opt-in.**

> **TWO CORRECTIONS 2026-07-30 — read before citing this gate.**
>
> 1. **The magnitude below is wrong.** "PCC 0.999998 / a ~2e-6 bf16 reduction-ORDER difference" came
>    from `bench_norm_fullcanvas.py`, which reports PCC > 1.0 elsewhere in its own table and so never
>    had that resolution. Measured directly at the shipped shape
>    (`tests/test_device_norm_fullcanvas.py`, QB2): the weighted norms differ on 19.43% of elements,
>    rel p99 1.14e-2, **max 2.24e-2 = 5.73 bf16 ULP**. Four orders of magnitude larger. The
>    `block_h` mechanism named below is nonetheless REAL; the amplification story still holds, it
>    just starts from a few ULPs rather than from 2e-6.
> 2. **This gate's own premise is void.** It ran with `DG_SPARSE_MOE=1, DG_SPARSE_MOE_TUNED=1`, i.e.
>    on the token-gather denoise MoE deleted in `7417bd7d69d` because it does not let the trajectory
>    converge at all (entropy plateaus ~0.46 against a 0.005 halt threshold). Both arms sat on that
>    baseline — which is exactly why both trajectories here are described as
>    "coherent-then-degenerate". The same mistake voided the pad-fix revert. **`committed_match =
>    0.145` cannot be cited for or against the flag until it is re-measured on the current path**,
>    where early halt fires (100% halted, mean 16.1 steps) — the very condition this document names
>    as the one that would make the flip safe.
>
> What has NOT changed: the flag is still default OFF, and the burden of proof is still on it. A
> 71-question GPQA prefix (2026-07-29: score 76.06% -> 78.87%, guard 2 -> 1, degenerate 2 -> 1,
> drift-any 3 -> 4 with 1 fixed / 2 new) is neither of the two things this document says a flip
> requires, and on the project's own >2% non-English gating metric it moves 0/71 -> 1/71.

## Why a 2e-6/norm change flips 85% of committed tokens

This is #48291 chaos-amplification, not a full-canvas bug. Per-norm PCC is 0.999998 (a ~2e-6 bf16
reduction/accumulation-ORDER difference between `block_h=8` and 8×`block_h=1`). But diffusion commits
the CLEAN ARGMAX with **no temperature/top-p cushion**, and the backbone already argmax-agrees with HF
only ~50%, so it sits on a knife-edge: a 2e-6 logit perturbation, compounded over 30 layers × 16 steps
and fed back through the entropy-budget accept/renoise loop (IoU 0.50 → different positions commit vs
re-noise each step), cascades to ~85% different committed tokens. This is exactly the bf16
chaos-amplification documented for batched commit (`commit_batching.md`: "no two non-bit-identical bf16
kernels meet 0.997 at 30L") and for bfp8 experts.

## Honest caveat: "different", not proven "worse"

Both trajectories are **coherent-then-degenerate** (the #48291 signature); neither is clearly better:
- chunked head: "diffusion language model is a generative model that produces text by iteratively
  refining a sequence of random noise or tokens into a coherent structure, rather than generating it
  word-by-word like traditional autoregressive models." → degenerates.
- full-canvas head: "diffusion language model is a type of generative model that creates text by
  iteratively refining a sequence of random noise into coherent language through a gradual process,
  rather than predicting the next word in a sequence like traditional models." → degenerates.

So full-canvas is a **different point in the space of equally-(un)faithful bf16 outputs**, not a
validated regression. But the request's rule is "hold vs the current chunked default within the
#48291 bar", and changing 85% of the shipping default's committed tokens fails that rule decisively —
a change that large must not be a silent default. **Landing stays opt-in.**

## What flipping the default would actually require

1. An **absolute** HF-vs-TT decision-fidelity comparison (dg-05 `demo/replay_hf_tt.py`) showing
   full-canvas argmax-agrees with HF **as well as** chunked does (both ~50% under #48291) — i.e. that
   the flip changes *which* equally-faithful output, not *whether* it is faithful. The TT-vs-TT gate
   here cannot distinguish that (both TT paths are ~50% vs HF), and per the request the TT-vs-chunked
   rule already gates the flip OFF.
2. Ideally, **#48291 itself resolved** (the fp32-attention backbone fix). Once the argmax has a cushion,
   the 2e-6 reduction-order difference would no longer flip committed tokens, `committed_match` would
   approach 1.0, and the flip would be safe — at which point early-halt also fires and the whole
   step-count/perf picture changes anyway.

(SUPERSEDED -- see the header: it shipped on 2026-07-30 and the flag is gone.) Until then: `DG_NORM_FULLCANVAS` remains opt-in (default OFF); the +15.8% @48 / +23.3% @12 traced win is
available to anyone who opts in and accepts the non-bit-identical (but coherent) output.

## Artifacts
- `norm_fullcanvas_flip_agreement.json` — the full compare output (committed to this dir).
- `doc/datatype_sweep/decision_agreement.py` — the harness (reused; `run` chunked/full-canvas + `compare`).
- Replay trajectories `traj_{chunked,fullcanvas}.pt` (per-step argmax/accept/entropy/committed) — in the
  run scratchpad, not committed (large-tensor artifact policy). Reproduce:
  `DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 [DG_NORM_FULLCANVAS=1] python .../decision_agreement.py run
  --num-layers 30 --max-denoising-steps 16 --seed 0 --output <path> --label <chunked|fullcanvas>` then
  `... compare --ref <chunked> --cand <fullcanvas>`.
