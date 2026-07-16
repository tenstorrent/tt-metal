# DiffusionGemma decision fidelity

QB2 TP=4 correctness work for
[tt-metal #48291](https://github.com/tenstorrent/tt-metal/issues/48291).

## Status: DECIDED — TT is at the intrinsic bf16 floor; the strict gate is mis-specified

After the retained `gelu_pytorch_tanh` fix, the canonical seeded replay scores
committed `0.99609375` (seed 0) but `0.91406250` (seed 1) against the HF-bf16
trajectory, and the strict gate (`committed_match > 0.95` **and** every
active-step entropy PCC `> 0.95` **and** terminal accept-IoU `> 0.95`) stays red
on seed 1. A decisive **bf16-floor self-consistency control** — running the
**same HF model** in fp32 vs bf16 with identical injected noise (zero TT kernels)
— settles why, and it is not a TT defect.

### The decisive control (fp32 vs bf16 of the *same* HF model, identical noise)

| Comparison | seed 0 committed | seed 1 committed |
| --- | --- | --- |
| **HF-fp32 vs HF-bf16** — intrinsic bf16 floor, no TT | **0.86328** (35 differ) | **0.91406** (22 differ) |
| HF-fp32 vs TT — TT vs the fp32 *ideal* | 0.86328 (35) | **0.98047** (5) |
| HF-bf16 vs TT — **the current gate** | 0.99609 (1) | 0.91406 (22) |

(8-step fixed gate schedule. Numbers re-derived from the raw `.pt` artifacts.)

Read across the rows:

- **The gate bar is below the bf16 architectural floor.** Running the reference
  model in fp32 vs bf16 already misses committed `> 0.95` on both seeds
  (`0.863` / `0.914`). The block-diffusion loop commits the clean argmax with no
  temperature cushion, so a bf16-scale perturbation of the logits bifurcates the
  trajectory into a *different but equally valid paraphrase*. No bf16
  implementation — TP=4 or single-device — can match the fp32 ideal to `0.95`,
  because **the reference cannot match itself in fp32 to `0.95`.**
- **TT is at or better than that floor.** `HF-fp32 vs TT` equals
  `HF-fp32 vs HF-bf16` on seed 0 (both `0.863`) and is *better* on seed 1
  (`0.980` vs `0.914`). The gate scores TT against a bf16 reference that is itself
  a chaotic draw, penalizing TT for the reference's own bf16 rounding.
- **Per-step entropy PCC is ill-conditioned, proven by the same control.**
  fp32-vs-bf16 entropy PCC collapses toward zero at converged steps (8-step
  seed 0 steps 5–7 ≈ `-0.004` negative; at the 48-step budget the same steps
  collapse to small-positive `~0.02–0.05` — identical ill-conditioning), while
  the absolute entropy error there is `≤ ~1.1` nat and falls to `~1e-3` at full
  convergence. As a step converges the
  per-token entropy profile goes near-constant, its variance → 0, and PCC is then
  dominated by rounding noise. A strict per-step entropy-PCC bar is unreachable in
  principle — the reference fails it against itself in fp32.

### The floor holds at the full production 48-step budget (not an 8-step artifact)

The gate uses a fixed 8-step schedule with early-halt disabled; production runs up
to 48 steps with early-halt (`entropy_stop_threshold = 0.005`). Re-measuring the
fp32-vs-bf16 floor at the production config (`measure`-style run, both sides
early-halt):

| seed | fp32 halt | bf16 halt | committed (fp32 vs bf16) |
| --- | --- | --- | --- |
| 0 | 7 steps | 8 steps | **0.86719** |
| 1 | 14 steps | 15 steps | **0.91406** |

Both trajectories **converge and halt** (7–15 steps, `halted=True`), and the floor
is essentially unchanged from the 8-step numbers (`0.863` → `0.867`, `0.914` →
`0.914`). So the floor is a production-scale property of the bf16 diffusion
trajectory, not an artifact of the 8-step gate truncation. Artifacts:
`/tmp/dg48291_hf_{fp32,bf16}_prod48_seed{0,1}.pt`.

A **direct TT run at the production 48-step config** (canonical prompt, seed 1,
early-halt on, device P150x4, `DG_SPARSE_MOE=1`) confirms TT is at/above the floor
— non-degenerate (36 content tokens), both halted:

| Comparison (production 48-step) | committed | differ |
| --- | --- | --- |
| **TT vs HF-fp32 (the ideal)** | **0.99219** | 2 |
| HF-fp32 vs HF-bf16 (floor) | 0.91406 | 22 |
| TT vs HF-bf16 (the gate) | 0.91016 | 23 |

TT tracks the fp32 ideal at `0.992` — *closer than the bf16 reference itself*
(`0.914`). TT's decoded output is essentially identical to fp32's
(*"…creates text by starting with a sequence of random noise and iteratively
refining it into a coherent sentence through a learned denoising process."*); the
gate fails only because it scores TT against the bf16 reference, which wandered to
a different valid paraphrase. Artifact: `/tmp/dg48291_tt_prod48_seed1_fixed.pt`.

### Output quality: coherent and correct at the converged production commit

Decoded committed text (prompt: *"Explain what a diffusion language model is in
one sentence."*), production 48-step config:

- **seed 0** — fp32 *" diffusion language model is a generative model that creates
  text by starting with a sequence of random noise and iteratively refining it
  into a coherent sentence through a learned denoising process."*; bf16 identical
  except a leading *"A"*. The 34/35 committed "misses" are **entirely a one-token
  left-shift alignment artifact** — the sentences are character-identical after
  "A".
- **seed 1** — fp32 *"…starting with a sequence of random noise … coherent
  sentence through a learned denoising process."*; bf16 *"…produces text by
  starting with random noise or a corrupted sequence … through a learned denoising
  process."*; both correct, fluent, equivalent definitions. TT (8-step) stays
  close to fp32.

Positional token-match — committed OR non-EOS — is corrupted by paraphrase
alignment. Non-EOS committed agreement makes this stark: seed 0 `fp32-vs-bf16 =
0.0000` (the leading-"A" shift misaligns every non-EOS position despite identical
text), seed 1 `fp32-vs-TT = 0.8571` vs `bf16-vs-TT = 0.4054`. It is the wrong
fidelity metric for a paraphrase-generating diffusion model.

## Decision

- **Achievable decision-fidelity floor on current bf16/TP=4 kernels = the
  intrinsic bf16 floor**: committed `≈0.86–0.98` vs the fp32 ideal (chaotic per
  seed), the *same* floor HF-bf16 itself achieves, and unchanged at the production
  48-step budget where the trajectory converges. This clears a usable product
  bar — coherent, correct text on every seed measured; the full-model prompt→text
  RUN (#47464) already validated production generation on QB2.
- **The strict #48291 gate as specified is unreachable by any bf16
  implementation, including the HF reference**, and positional `committed_match`
  / per-step-entropy-PCC vs a single chaotic bf16 reference are the wrong fidelity
  metrics for a paraphrase-generating diffusion model.
- **Recommendation:**
  1. **Product-accept** the current output: TT is at or above the bf16 floor and
     produces coherent, correct generations.
  2. **Re-specify the gate** to sound, reachable criteria: measure fidelity to the
     **fp32 ideal** (not a chaotic bf16 draw); gate on **output quality /
     alignment-robust** agreement rather than positional token equality; and use
     `tests/trajectory_pcc.py:sound_entropy_step_fidelity` as a **converged-step
     diagnostic** (well-conditioned where raw PCC is not) rather than a per-step
     pass/fail bar. That metric is a conditioning fix, **not** a reachable gate:
     at genuine transition-step divergence the reference fails it against itself,
     which is the floor, not a defect. Flipping the production gate's pass criteria
     is a correctness-policy change **left for owner sign-off** — the production
     gate here is deliberately unchanged and still red.
  3. If bit-closer-to-fp32 fidelity is ever required (it is not for product),
     **fp32 MoE backbone precision is a separate owned effort on the shared
     Gemma-4 backbone** — blocked today by `ttnn.topk` `TT_FATAL` on FLOAT32 and
     fp32 experts exceeding the QB2 DRAM budget; DiffusionGemma must not edit
     `models/demos/gemma4/`.

## Reproduce

The bf16-floor control (needs the DiffusionGemma venv + ~104 GB host RAM for the
fp32 model; **no TT device** — HF runs on CPU):

```bash
PYTHONPATH=$TT_METAL_HOME python \
  models/experimental/diffusion_gemma/doc/decision_fidelity/measure_bf16_floor.py \
  --stage-artifact /tmp/dg48291_tanh_seed1.pt --checkpoint $DG_CKPT
```

`demo/replay_hf_tt.py` gained `--hf-dtype {bfloat16,float32}` (fp32 forbidden under
`--stage-gate`) so the fp32 reference trajectory can be regenerated directly. Run
the production gate itself with `DG_SPARSE_MOE=1`, `--stage-gate`,
`--noise-mode seeded`, `--max-denoising-steps 8`.

## Retained fixes

- All DiffusionGemma MLP paths select `ttnn.GeluVariant.Tanh`, matching checkpoint
  `gelu_pytorch_tanh` (the legacy boolean selected FastLut erf-GELU). Raises
  canonical seed-0 committed `0.9296875 -> 0.99609375`. `DG_GELU_TANH=0` retained
  for bisects.
- Sparse-matmul FP32 intermediate circular-buffer sizing follows the actual
  intermediate format.
- Denoise sparse MoE defaults capacity to the canvas length (the old capacity of
  32 dropped 41–84% of active routes per layer).
- `tests/trajectory_pcc.py:sound_entropy_step_fidelity` — well-conditioned
  per-step entropy diagnostic: requires the absolute tolerance on both branches
  (catches affine/offset errors) and PCC only where the reference profile has
  structure (avoids the degenerate-denominator false alarm). CPU-tested.
- `demo/replay_hf_tt.py:--hf-dtype` + `doc/decision_fidelity/measure_bf16_floor.py`
  — the reproducible bf16-floor self-consistency control.
- The stage gate stops its entropy minimum at common all-accept and records the
  raw all-step minimum separately.
- An opt-in Blackhole routed-expert HiFi4 FP32-full-DST candidate
  (`DG_SPARSE_EXPERT_FP32_FULL_SYNC=1`) improves seed-0 committed
  `0.9296875 -> 0.93359375` but regresses seed-1 active-step entropy; opt-in only.

## Remaining risk

- Floor characterized on one canonical prompt. The fp32-vs-bf16 floor is confirmed
  across **5 seeds** (committed `0.863 / 0.914 / 0.891 / 0.871 / 0.957`, mean `0.899`;
  4 of 5 below the 0.95 committed bar) at 8-step, and at the 48-step production budget
  on 2 seeds — so the "reference can't reliably match itself" result is seed-robust,
  not a 2-seed fluke. A broader prompt sweep would further tighten the "usable bar"
  claim; TT-side rows measured on seeds 0/1 (device).
- TT's fidelity-to-floor is directly measured at both the 8-step gate
  (TT-vs-fp32 `0.980`) and the 48-step production budget (TT-vs-fp32 `0.992`),
  so it is no longer inferred. (A first 48-step device attempt collapsed to
  all-EOS because the run used the default "Once upon a time" prompt, not the
  canonical one — HF collapsed identically, confirming it was prompt/invocation,
  not the device path; the corrected run is the one recorded.)
- The production gate's pass criteria are unchanged pending owner sign-off on the
  re-specification; the gate stays red against the (mis-specified) bf16 reference.
- **Inherited (non-dg-05) HARD-RULE gate failure:** the branch carries committed
  shared-`gemma4` edits (`tt/experts/operations.py` +1 `deallocate`, #47464;
  `tt/model.py` sharded-terminal) that pre-date this stage and fail
  `check_no_shared_gemma4_edits.sh`. This session made zero shared edits; the
  cleanup is owned by #47462 / #47464 / sharded-terminal and is flagged for owner
  action.

See `work_log.md` and the root `AUTOFIX.md` / `AUTODEBUG.md` for the full
experiment ledger and refutations.
