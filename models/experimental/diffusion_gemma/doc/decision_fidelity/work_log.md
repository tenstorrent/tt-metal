# Decision-fidelity work log

## Reproduction

Target: DiffusionGemma 26B-A4B-it, QB2 `P150x4`, TP=4.

Canonical probe:

```bash
DG_SPARSE_MOE=1 \
python -u models/experimental/diffusion_gemma/demo/replay_hf_tt.py \
  --checkpoint "$DG_CKPT" \
  --hf-checkpoint "$DG_CKPT" \
  --local-files-only \
  --prompt "Explain what a diffusion language model is in one sentence." \
  --seed 0 \
  --max-denoising-steps 8 \
  --noise-mode seeded \
  --stage-gate \
  --output /tmp/dg48291_stage_gate.pt
```

HF and TT receive identical initial canvas, FP32 Gumbel noise, and renoise token
IDs. SHA-256 hashes are saved in the artifact.

## Retained results

- Correct `GeluVariant.Tanh` activation, canonical seed 0: committed
  `0.99609375`, HF non-EOS `0.9714286`; per-step argmax
  `[0.9453, 0.9570, 0.9961, 0.9766, 0.9883, 0.9961, 0.9961, 0.9961]`.
- The same semantic fix improves seed 1 to `0.9140625`; combining it with
  full-DST experts reaches `0.92578125`, so multi-seed fidelity remains open.
- Post-fix traced seed 0 captures/replays all eight single-step traces and
  reproduces eager committed/non-EOS agreement exactly at
  `0.99609375 / 0.9714286`.
- Independent follow-up review clean-passed activation coverage: direct model,
  sparse/dense expert, shared MLP, self-conditioning, regular/chunked prefill,
  commit, and traced target paths all enter the DG tanh context; plain Gemma4
  remains outside it.
- Dense/ragged eight-step committed match: `0.92578125`.
- Original capacity-32 true-sparse committed match: `0.5625`.
- Zero-drop capacity-256 true-sparse committed match: `0.91796875`.
- Zero-drop true-sparse with HiFi4 FP32 full-DST expert projections:
  revision-locked committed match `0.9296875 -> 0.93359375`, HF non-EOS
  `0.4857143 -> 0.51428574`, minimum active-step entropy PCC
  `-0.00607 -> 0.61842`, and terminal-active IoU `1.0`.
- Seed-1 A/B also improves committed match (`0.8828125 -> 0.89453125`) but
  regresses active-step entropy after step 2, so this remains an opt-in
  candidate rather than the production default.
- Current same-revision canonical zero-drop sparse baseline after all retained
  semantic fixes: committed `0.9296875`, HF non-EOS `0.4857143`.
- Blackhole sparse-matmul FP32 CB regression: PCC `0.999998`, max absolute
  error `0.1633`.

The production capacity-32 path had maximum per-expert loads of 156–256 and
dropped 838–1711 of 2048 active routes in each layer.

## Localization

- HF LM head on TT final hidden preserves TT decisions: terminal path refuted.
- Exact HF routing on free-running TT hidden does not improve the trajectory.
- Every TT causal-prefill layer run from exact HF input writes K/V at
  PCC `0.99991–0.99996`; normal K/V drift is inherited from hidden drift.
- Exact HF prompt KV plus exact layer-20 input: eight-step committed match `1.0`.
- Exact layer-19 post-attention normalized branch: final committed match
  `0.99609375`, but intermediate entropy/accept decisions diverge.
- HF layer-19 attention+post-norm on live TT hidden: step-0 `0.9140625`.
- Device, BF16 left-fold, BF16 pairwise, and FP32 host TP sums all give branch
  PCC approximately `0.9427`.
- Injecting HF's exact per-step self-conditioning signal yields committed and
  HF non-EOS agreement `1.0`.
- Recomputing the exact HF softmax/embedding signal from the live TT previous
  logits does not reproduce that oracle (`0.91796875` committed), and combining
  it with full-DST experts regresses to `0.890625`. The exact-HF-signal result
  resets the feedback trajectory; it does not isolate a TT soft-embedding bug.
- Exact HF layer-0 input on every step passes the gate (`1.0`, minimum active
  entropy PCC `0.95884`), while injecting it only at step 0 regresses to
  `0.86328125`. The former is another feedback reset, not a DG input fix.
- The full-canvas exact-input+exact-KV ledger localizes the worst single-layer
  output to layer 18 (PCC `0.9974325`). Split branch PCCs are post-attention
  `0.9997648`, shared FF `0.9983796`, routed expert FF `0.9918051`, and post-FF
  `0.9965505`.
- Exact HF routing raises layer-18 routed FF to `0.9977631`, but the same
  canonical router on live free-running TT hidden regresses the trajectory to
  `0.9140625`. Exact prompt KV plus all live HF routers reaches only
  `0.92578125`.
- Selecting the checkpoint's actual tanh-GELU raises layer-18 shared FF
  `0.9983796 -> 0.9998859`, routed FF `0.9918051 -> 0.9937973`, exact-route
  routed FF `0.9977631 -> 0.9997175`, and post-FF
  `0.9965505 -> 0.9978347`.

The successful fixed-HF branch is a teacher-forced reset from a different
trajectory, not a production layer-19 fix.

## Gate corrections

`--stage-gate` now enforces seeded noise, canvas 256, eight denoise steps, the
full model, P150x4, `DG_SPARSE_MOE=1`, and no diagnostic HF injection. Its
entropy minimum ends at the first common all-accept step and separately records
the raw all-step minimum. CPU tests cover canonical-argument validation and
post-saturation exclusion.

## Refuted candidates

- Full/manual FP32 attention and selective FP32 SDPA intermediates.
- Broad or targeted FP32 RMSNorm, CCL, router, sparse combine, and expert reduce.
- Full-DST FP32 sparse combine, denoise LM head, and shared MLP. Each improved
  at least one single-step continuous metric but reduced eight-step committed
  agreement to `0.92578125`, `0.87109375`, and `0.91015625`, respectively.
- Restricting full-DST FP32 routed experts to layers 0–19 or 20–29. The former
  reduced eight-step committed agreement to `0.875`; the latter regressed the
  single-step decision metric.
- Score-first compact routing and BF16 sort routing.
- HF router on live TT hidden as a full-trajectory correction.
- All-reduce-before-route-combine.
- Raw layer-19 attention replacement, live-TT HF post-norm, targeted layer-19
  HiFi4/FP32 o_proj+norm, and TP sum parenthesization changes.
- Canonical FP32 post-LM-head softcap/decision path: exact device and host
  implementations both regress committed agreement to `0.91015625`.
- FP32 entropy and tanh+FP32-terminal combinations regress committed agreement
  to `0.9609375` and `0.91796875`, respectively.
- HF expert-ID-order BF16 left-fold combine: layer-18 exact-input routed-branch
  PCC changes only `0.9977631 -> 0.9977694`.
- HF FP32 route-weight-before-BF16-contribution: tanh+exact-route expert FF
  changes only `0.9997175 -> 0.9997222`, while post-FF slightly regresses.
- FP32-packed Gumbel addition and processed-logit clean argmax do not change
  seed-1 committed or entropy metrics.

## bf16-floor self-consistency control (decisive)

Run the SAME HF model in fp32 vs bf16 with identical seeded 8-step injected noise
(`doc/decision_fidelity/measure_bf16_floor.py`; zero TT kernels). This measures
the intrinsic sensitivity of the block-diffusion trajectory to a bf16-scale
perturbation of the model logits, independent of any TT implementation.

- fp32-vs-bf16 committed match: seed 0 `0.86328125` (35 positions differ, block
  `[0..34]`), seed 1 `0.91406250` (22 differ, block `[15..36]`). Both are BELOW
  the `0.95` gate — the reference cannot match itself in fp32.
- fp32-vs-bf16 per-step entropy PCC collapses and goes negative at converged
  steps (seed 0 steps 5–7 ≈ `-0.004`; across-position entropy std `≈0.007`,
  max |Δ| `≈1e-4`–`0.1`). The strict per-step entropy-PCC bar is ill-conditioned
  (near-constant vector → PCC dominated by rounding), proven against the
  reference itself.
- TT is at or better than the floor: `HF-fp32 vs TT` committed = seed 0
  `0.86328125` (== the bf16 floor), seed 1 `0.98046875` (BETTER than
  `HF-bf16 vs TT` `0.91406250` — TT tracks the fp32 ideal more closely than the
  bf16 reference does).
- Decoded text: fp32, HF-bf16, and TT are all coherent, correct one-sentence
  definitions on both seeds; committed-match misses are valid-paraphrase and
  token-alignment artifacts, not wrong tokens.
- Artifacts: `/tmp/dg48291_hf_fp32_seed0.pt`, `/tmp/dg48291_hf_fp32_seed1.pt`
  (fp32 HF trajectories); `/tmp/dg48291_tanh_seed{0,1}.pt` (bf16 HF + TT).

### bf16 floor at the production 48-step budget (not an 8-step artifact)

The 8-step gate disables early-halt; production runs up to 48 steps with
`entropy_stop_threshold = 0.005`. Re-running the fp32-vs-bf16 control at the
production config (`scratch hf_floor_prod48`): both sides converge and halt
(`halted=True`) at 7–15 steps, and the floor is unchanged.

- seed 0: fp32 halts at 7 steps, bf16 at 8; committed `0.867188` (34 differ). The
  fp32/bf16 sentences are character-identical except a leading "A" (one-token
  left-shift) — the 34 "misses" are pure alignment.
- seed 1: fp32 halts at 14 steps, bf16 at 15; committed `0.914062` (22 differ);
  both coherent, correct, equivalent definitions.
- Artifacts: `/tmp/dg48291_hf_{fp32,bf16}_prod48_seed{0,1}.pt`. So the intrinsic
  bf16 floor (`0.863`/`0.914` at 8 steps) is confirmed at production scale
  (`0.867`/`0.914`), where the trajectory actually converges.

Direct TT at the production 48-step config (seed 1, canonical prompt, early-halt,
device P150x4, `DG_SPARSE_MOE=1`; non-degenerate, 36 content tokens, both halted):
committed `TT-vs-fp32 = 0.99219` (2 differ), `fp32-vs-bf16 = 0.91406`,
`TT-vs-bf16 = 0.91016`. TT tracks the fp32 ideal more closely than the bf16
reference does, and TT's decoded text is essentially identical to fp32's. Artifact
`/tmp/dg48291_tt_prod48_seed1_fixed.pt`. (A first attempt collapsed to all-EOS
because it defaulted to the "Once upon a time" prompt — HF collapsed identically,
so it was prompt/invocation, not the device; corrected run recorded.)

### bf16 floor across 5 seeds (CPU-only, HF fp32 vs bf16, 8-step)

Broader-seed check of the intrinsic bf16 floor (same HF model fp32 vs bf16, identical
seeded 8-step noise, canonical prompt; canvas + noise generated deterministically per
seed, no TT). seed 0/1 reproduce the earlier `0.8633`/`0.9141` exactly (validation).

Full 3-way committed_match (fp32/bf16 = CPU floor; TT = device replay per seed, 8-step,
seeded, canonical prompt; seeds 0/1 reproduce the original table exactly):

| seed | HF-fp32 vs HF-bf16 (floor) | HF-fp32 vs TT | HF-bf16 vs TT (gate) |
| --- | --- | --- | --- |
| 0 | 0.8633 | 0.8633 | 0.9961 |
| 1 | 0.9141 | 0.9805 | 0.9141 |
| 2 | 0.8906 | 0.9141 | 0.8984 |
| 3 | 0.8711 | 0.8789 | 0.9062 |
| 4 | 0.9570 | 0.9453 | 0.9609 |
| **mean** | **0.8992** | **0.9164** | **0.9352** |

Two seed-robust conclusions:
- **The floor holds:** 4 of 5 seeds have fp32-vs-bf16 below the 0.95 committed bar (mean
  0.899); only seed 4 (0.957) is above. The reference cannot *reliably* match itself in
  fp32 to 0.95 — the floor is seed-dependent (chaotic per noise realization). No bf16
  implementation clears a committed-`>0.95`-on-every-seed bar, and even seed 4 fails the
  full strict gate (which also needs per-step entropy-PCC + accept-IoU `>0.95`, both shown
  to collapse).
- **TT is at/above the floor:** TT-vs-fp32 mean 0.916 ≥ floor mean 0.899; per-seed
  TT-vs-fp32 minus floor = `[0.0, +0.066, +0.023, +0.008, -0.012]` (at/above on 4/5; seed 4
  is 1.2% below, within the chaos noise). TT tracks the fp32 ideal at least as well as the
  bf16 reference does — it is NOT systematically worse. The `bf16-vs-TT` gate column
  (mean 0.935, 4/5 below 0.95) mirrors the floor.

Artifacts: `/tmp/dg48291_floor_5seed.pt` (fp32/bf16), `/tmp/dg48291_seed{2,3,4}.pt` +
`/tmp/dg48291_tanh_seed{0,1}.pt` (bf16+TT); harness `doc/decision_fidelity/floor_5seed.py`.

## Metric caveats and disclosures

- **Positional token-match is corrupted by paraphrase alignment.** Non-EOS
  committed agreement (over the reference's non-EOS positions): seed 0
  `fp32-vs-bf16 = 0.0000` (!), `fp32-vs-TT = 0.0000`, `bf16-vs-TT = 0.9714`;
  seed 1 `fp32-vs-bf16 = 0.4286`, `fp32-vs-TT = 0.8571`, `bf16-vs-TT = 0.4054`.
  The seed-0 `0.0000` is the alignment artifact in its purest form: the fp32
  output drops the leading "A" (a one-token left shift), so every non-EOS
  position misaligns even though the text is essentially identical. This is why
  neither positional `committed_match` nor non-EOS agreement is a valid fidelity
  metric for a paraphrase-generating diffusion model.
- **Converged-step entropy abs error is not tiny.** The self-consistency control
  reaches max |Δ entropy| up to ~`1.1` (seed 0 step 5) and ~`2.8` at transition
  steps (seed 1 step 2) — the reference vs itself in fp32. The degenerate-
  denominator PCC collapse and the genuine transition-step divergence coexist;
  `sound_entropy_step_fidelity` separates them but does not make the transition
  steps pass (the reference fails them too — that is the floor).
- **Inherited shared-`gemma4` gate failure (NOT dg-05).**
  `check_no_shared_gemma4_edits.sh` (vs the origin/main merge-base) is RED on
  `models/demos/gemma4/tt/experts/operations.py` (+1 `deallocate`, commit
  `bf98aaf2e23`, #47464) and `models/demos/gemma4/tt/model.py` (sharded-terminal,
  commit `a22107f0447`). This session (dg-05) made ZERO shared edits — the
  working tree and staged diffs are empty of `gemma4`. The violation is inherited
  from #47464 / sharded-terminal and is those stages' cleanup, but it is a live
  HARD-RULE gate failure on the branch and is flagged for owner action.

## Router top-8 expert SELECTION: bf16 vs fp32 (device-free proxy for TT vs HF)

"How much does TT's per-layer top-8 expert selection differ from HF?" measured at
the router level with `doc/decision_fidelity/measure_topk_overlap.py`: run the HF
backbone fp32 and bf16 on the *same* seeded canvas/noise (zero TT kernels),
capture every router's dense top-8 per step/layer, and compare index-set overlap,
functional weight-mass overlap (`sum_e min(w_fp32, w_bf16)` after renormalizing,
= 1 − TV distance), top-1 (dominant-expert) agreement, and rank-resolved drop.
TT's router is bf16 (`sparse_moe.py:921` softmax + `:923` topk) and tracks the
bf16 committed floor over 5 seeds, so fp32-vs-bf16 is a faithful device-free proxy
(TT flips *slightly more*: TT does bf16 softmax where HF-bf16 keeps fp32 softmax).

**STEP 0 (identical initial canvas — clean router-precision isolate), seed 0 / 1:**

| metric | seed 0 | seed 1 |
| --- | --- | --- |
| top-8 index set shared | **6.80 / 8** | **6.79 / 8** |
| exact top-8 set match | 0.436 (56% flip ≥1) | 0.448 (55% flip ≥1) |
| **weight-mass overlap (functional)** | **0.846** | **0.845** |
| top-1 dominant-expert agree | 0.818 | 0.808 |
| rank-0 (dominant) drop out of bf16 top-8 | **4.2%** | **4.4%** |
| rank-7 (weakest) drop | 36.4% | 35.4% |

Rank-resolved drop @step0 (fp32 rank0..7 → out of bf16 top-8), seed 0:
`[0.042, 0.053, 0.072, 0.098, 0.133, 0.184, 0.253, 0.364]`. **Flips are
tail-dominated**: the dominant expert survives in the top-8 ~96% of the time and
is the exact #1 ~81% of the time; the weakest (8th) expert — which carries ~1–2%
of the routing weight — flips ~36%. Only ~15% of routing *weight* lands on a
flipped expert.

**Depth compounding (per-layer @ step 0, same input):** the topk operator is
essentially exact where the hidden state is still shared — layers 0–4 weight-mass
`0.98–0.99`, top-1 `0.97–1.00`. Divergence grows with backbone depth as bf16
hidden-state rounding accumulates — layers 20–27 weight-mass `0.64–0.72`, top-1
`0.55–0.66` — then partly recovers at the output layers (28–29 `0.85–0.90`). So
the divergence is **bf16 backbone drift, not a faulty router**.

**OVERALL (full 8-step trajectory), seed 0 / 1:** top-8 shared `5.21 / 5.34` of 8;
weight-mass overlap `0.637 / 0.655`; top-1 agree `0.568 / 0.588`. This is *lower*
than step 0 because the two runs commit different tokens after step 0 — later-step
routing differs for the legitimate reason that the trajectories themselves have
diverged (the same #48291 paraphrase bifurcation), not because the router got less
accurate. Per-step confirms it: step 0 (identical input) mass `0.845`, steps 1–5
drop to `0.50–0.69` as trajectories split, steps 6–7 recover as they converge.

**Reading:** given the same input, TT-class bf16 preserves the *functional*
routing — 85% of the weight and 96% of the dominant expert — and only reshuffles
the near-tied tail (rank 6–8). This is the router-level face of the same bf16
chaos floor that governs committed decisions, and it is why the output stays
coherent. Artifacts `/tmp/dg48291_topk2_tanh_seed{0,1}.pt`; reproduce with
`measure_topk_overlap.py --stage-artifact /tmp/dg48291_tanh_seed{0,1}.pt`.

## Current conclusion

The tanh-GELU semantic fix clears the core seed-0 committed-decision bar at
`0.99609375` and is now the DG default. The strict aggregate gate stays red on
seed 1 **because the gate is mis-specified, not because TT is defective**: the
bf16-floor self-consistency control shows the diffusion trajectory is chaotic at
the bf16 rounding scale (fp32-vs-bf16 committed `0.863`/`0.914`, below `0.95`) and
that per-step entropy PCC is ill-conditioned at converged steps (negative for the
reference vs itself). TT sits at or above that intrinsic bf16 floor and produces
coherent, correct output. Decision + recommendation (product-accept; re-spec the
gate to the fp32 ideal + variance-gated entropy; fp32 MoE backbone as a separate
owned effort) are in `README.md`. The earlier claim of an inherent shared Gemma-4
BF16 ceiling is withdrawn; the floor is the intrinsic *diffusion-trajectory*
chaos under bf16, which the reference shares.
