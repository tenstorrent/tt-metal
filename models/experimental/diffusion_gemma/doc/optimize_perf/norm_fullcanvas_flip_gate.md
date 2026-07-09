# DG_NORM_FULLCANVAS default-flip gate — decision fidelity (dg-05 method, #48291) → KEEP OPT-IN

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

Until then: `DG_NORM_FULLCANVAS` remains opt-in (default OFF); the +15.8% @48 / +23.3% @12 traced win is
available to anyone who opts in and accepts the non-bit-identical (but coherent) output.

## Artifacts
- `norm_fullcanvas_flip_agreement.json` — the full compare output (committed to this dir).
- `doc/datatype_sweep/decision_agreement.py` — the harness (reused; `run` chunked/full-canvas + `compare`).
- Replay trajectories `traj_{chunked,fullcanvas}.pt` (per-step argmax/accept/entropy/committed) — in the
  run scratchpad, not committed (large-tensor artifact policy). Reproduce:
  `DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 [DG_NORM_FULLCANVAS=1] python .../decision_agreement.py run
  --num-layers 30 --max-denoising-steps 16 --seed 0 --output <path> --label <chunked|fullcanvas>` then
  `... compare --ref <chunked> --cand <fullcanvas>`.
