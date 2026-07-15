# DiffusionGemma decision fidelity

QB2 TP=4 correctness work for
[tt-metal #48291](https://github.com/tenstorrent/tt-metal/issues/48291).

## Status: DECIDED — TT is at the intrinsic bf16 floor; the strict gate is mis-specified

After the retained `gelu_pytorch_tanh` fix, the canonical seeded eight-step replay
scores committed `0.99609375` (seed 0) but `0.91406250` (seed 1), and the strict
gate (`committed_match > 0.95` **and** every active-step entropy PCC `> 0.95`
**and** terminal accept-IoU `> 0.95`, measured **against the HF-bf16 trajectory**)
stays red on seed 1. A decisive **bf16-floor self-consistency control** — running
the **same HF model** in fp32 vs bf16 with identical injected noise (zero TT
kernels) — settles why, and it is not a TT defect.

### The decisive control (fp32 vs bf16 of the *same* HF model, seeded 8-step)

| Comparison | seed 0 committed | seed 1 committed |
| --- | --- | --- |
| **HF-fp32 vs HF-bf16** — intrinsic bf16 floor, no TT | **0.86328** (35 differ) | **0.91406** (22 differ) |
| HF-fp32 vs TT — TT vs the fp32 *ideal* | 0.86328 (35) | **0.98047** (5) |
| HF-bf16 vs TT — **the current gate** | 0.99609 (1) | 0.91406 (22) |

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
  (`0.980` vs `0.914` — TT tracks the fp32 ideal more closely than the bf16
  reference does). The gate scores TT against a bf16 reference that is itself a
  chaotic draw, penalizing TT for the reference's own bf16 rounding.
- **Per-step entropy PCC is ill-conditioned, proven by the same control.**
  fp32-vs-bf16 entropy PCC collapses and goes **negative** at converged steps
  (seed 0 steps 5–7 ≈ `-0.004`), while the absolute entropy error there is
  ~`1e-4`–`0.1`. As a step converges the per-token entropy profile goes
  near-constant, its across-position variance → 0, and Pearson correlation is
  then dominated by rounding noise. A strict per-step entropy-PCC bar is
  unreachable in principle — the reference fails it against itself in fp32.

### Output quality: all three trajectories are coherent and correct

Decoded committed text (prompt: *"Explain what a diffusion language model is in
one sentence."*):

- **seed 0**, all three: *"A diffusion language model is a generative model that
  creates text by starting with a sequence of random noise and iteratively
  refining it into a coherent sentence through a **learned / gradual** denoising
  process."* — differ only by a synonym at one position.
- **seed 1**: HF-bf16 *"…starting with random noise or a blank sequence …
  through a reverse diffusion process."*; TT *"…starting with a sequence of
  random noise … through a denoising process."*; HF-fp32 close to TT. All are
  correct, fluent, equivalent definitions. The 22-position seed-1 "block"
  [15..36] is a **paraphrase-alignment** artifact (TT's shorter sentence shifts
  token alignment, then pads), not 22 wrong tokens.

Committed-match (position-wise token equality against one reference trajectory)
therefore **overstates** the practical defect: it penalizes valid paraphrases
that chaotic bf16 feedback produces, and it is corrupted by alignment shifts.

## Decision

- **Achievable decision-fidelity floor on current bf16/TP=4 kernels = the
  intrinsic bf16 floor**: committed `≈0.86–0.98` vs the fp32 ideal (chaotic per
  seed), the *same* floor HF-bf16 itself achieves. This clears a usable product
  bar — coherent, correct text on every seed measured.
- **The strict #48291 gate as specified is unreachable by any bf16
  implementation, including the HF reference**, and `committed_match` /
  per-step-entropy-PCC vs a single chaotic bf16 reference are the wrong fidelity
  metrics for a paraphrase-generating diffusion model.
- **Recommendation:**
  1. **Product-accept** the current output: TT is at or above the bf16 floor and
     produces coherent, correct generations.
  2. **Re-specify the gate** to sound, reachable criteria: measure fidelity to
     the **fp32 ideal** (not a chaotic bf16 draw); replace the ill-conditioned
     per-step entropy-PCC with the **variance-gated** metric
     (`tests/trajectory_pcc.py:sound_entropy_step_fidelity` — PCC only where the
     reference entropy profile has structure, absolute tolerance where it does
     not); and treat committed-match as a chaos-bounded diagnostic rather than a
     hard bar. This does **not** weaken fidelity — it stops measuring rounding
     chaos the reference itself exhibits. Flipping the production gate's pass
     criteria is a correctness-policy change and is left for owner sign-off.
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

The stage artifact is any `demo/replay_hf_tt.py --stage-gate` output (holds the
bf16 `hf_traj` + device `tt_traj` + inputs). `demo/replay_hf_tt.py` also gained
`--hf-dtype {bfloat16,float32}` (fp32 forbidden under `--stage-gate`) so the fp32
reference trajectory can be regenerated directly. Run the production gate itself
with `DG_SPARSE_MOE=1`, `--stage-gate`, `--noise-mode seeded`,
`--max-denoising-steps 8`.

## Retained fixes

- All DiffusionGemma MLP paths select `ttnn.GeluVariant.Tanh`, matching
  checkpoint `gelu_pytorch_tanh` (the legacy boolean selected FastLut erf-GELU).
  Raises canonical seed-0 committed `0.9296875 -> 0.99609375`. `DG_GELU_TANH=0`
  retained for bisects.
- Sparse-matmul FP32 intermediate circular-buffer sizing follows the actual
  intermediate format.
- Denoise sparse MoE defaults capacity to the canvas length (the old capacity of
  32 dropped 41–84% of active routes per layer).
- `tests/trajectory_pcc.py:sound_entropy_step_fidelity` — variance-gated entropy
  fidelity that is well-conditioned at converged steps (CPU-tested).
- `demo/replay_hf_tt.py:--hf-dtype` + `doc/decision_fidelity/measure_bf16_floor.py`
  — the reproducible bf16-floor self-consistency control.
- The stage gate stops its entropy minimum at common all-accept and records the
  raw all-step minimum separately.
- An opt-in Blackhole routed-expert HiFi4 FP32-full-DST candidate
  (`DG_SPARSE_EXPERT_FP32_FULL_SYNC=1`) improves seed-0 committed
  `0.9296875 -> 0.93359375` but regresses seed-1 active-step entropy; opt-in only.

## Remaining risk

- The floor is characterized on one canonical prompt at two seeds. A broader
  prompt/seed sweep would tighten the "usable bar" claim, but every trajectory
  measured produces coherent correct text and TT never falls below the bf16
  floor.
- The production gate's pass criteria are unchanged pending owner sign-off on the
  re-specification; the gate stays red against the (mis-specified) bf16 reference.

See `work_log.md` and the root `AUTOFIX.md` / `AUTODEBUG.md` for the full
experiment ledger and refutations.
