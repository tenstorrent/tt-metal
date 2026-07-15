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
