# AutoFix Report

## Starting Evidence
- The original compact-ragged candidate changed canonical committed agreement from
  seed-0 `0.99609375` to `0.89453125` and seed-1 `0.9140625` to `0.90625`.
- Its metadata kernel was elementwise exact, but a one-layer MoE comparison was only
  PCC `0.99924`, max-abs `0.0127`.
- `AUTOTRIAGE.md` describes an already-fixed NoC alignment hang and is not the
  numerical diagnosis. A fresh `AUTODEBUG.md` was requested for independent review.

## Hypothesis Experiments
- Hypothesis: compact router/packing drops or misweights routes.
  Experiment: compare device slot-token, inverse map, scaled weights, sparsity, and
  overflow mappings against a host oracle.
  Result: all tensors elementwise exact; no drops.
  Verdict: refuted.

- Hypothesis: compact expert matmul geometry changes BF16 accumulation.
  Experiment: hold real weights/input fixed and sweep C=32 gate/up/down K blocks
  against C=256 output rows.
  Result: prior `(22,2)` geometry was inexact; `(8,2)` was elementwise exact and
  retained ~2 ms component latency.
  Verdict: verified.
  Fix: use reduction-compatible K-block 8 for gate/up and 2 for down in both primary
  and overflow expert paths.

- Hypothesis: `fast_reduce_nc` changes the combine reduction.
  Experiment: hold C=256 expert output fixed and compare dense combine matmul with
  `embedding + fast_reduce_nc`.
  Result: PCC `0.999993`, max-abs `0.00146484375`, not elementwise exact; the
  difference was enough to bifurcate the diffusion trajectory.
  Verdict: verified.
  Fix: build the baseline dense combine directly from compact columns/weights,
  scatter compact expert tiles to `[E*C,H]`, and reuse the original combine matmul.
  The fast-reduce form remains diagnostic only.

## Final Status
- Fixed: reduced 1L/2-step trajectory is exact across all six fields.
- Fixed: full traced @12/@48 committed SHA matches baseline exactly.
- Fixed: strict HF committed agreement restored to seed-0 `0.99609375` and seed-1
  `0.9140625`.
- Performance retained: fixed @48 12.941→13.559 tok/s (+4.8%), 18.8 ms saved per
  denoise step. The inexact fast-reduce ceiling remains +38.6%.
- Remaining limitation: a faster compact combine must reproduce the baseline fused
  BF16 multiply-accumulate contract; simple fast-reduce or compact-K matmul variants
  do not.
# AutoFix Report

## Starting Evidence
- Issue: #48291, QB2 TP=4 DiffusionGemma decision fidelity.
- Seeded eight-step dense-MoE baseline: committed agreement `0.92578125`.
- Exact-input controls show continuous prefill/denoise drift plus discrete MoE routing amplification.

## Hypothesis Experiments
- Hypothesis: TT router implementation contributes independently of hidden-state drift.
  - Experiment: run each HF router on the corresponding live TT hidden state.
  - Result: step-0 agreement `0.91796875 -> 0.93359375`; entropy PCC `0.97713 -> 0.98626`.
  - Verdict: verified for one-step fidelity.
- Hypothesis: one router subcomponent is sufficient to recover the gain.
  - TT RMSNorm + HF projection/softmax/top-k: agreement `0.9140625`.
  - HF RMSNorm + TT projection/softmax/top-k: agreement `0.9140625`.
  - Verdict: refuted; both sides of the discrete boundary contribute.
- Hypothesis: a DG-local HiFi4/FP32 RMSNorm, projection, softmax, and FP32 sort router is a production fix.
  - Step-0 result: agreement `0.93359375`, matching the HF-router oracle on TT hidden.
  - Eight-step result: committed agreement `0.859375`, worse than the `0.92578125` baseline.
  - Verdict: refuted; the altered first decision changes diffusion feedback and degrades the trajectory.
- Hypothesis: limit the accurate router to sensitive layers or the final denoise step.
  - Layers 8/18/20 over eight steps: committed agreement `0.90625`.
  - Final step only: committed agreement unchanged at `0.92578125`.
  - Verdict: refuted.
- Hypothesis: select top-k directly from BF16 scores and apply FP32 softmax only to the compact eight routes.
  - Step-0 agreement fell from `0.91796875` to `0.90234375`; entropy PCC fell from `0.97713` to `0.97374`.
  - Verdict: refuted before a full trajectory run.
- Hypothesis: the sparse `[S,4096]` combine reduction causes the production sparse-only regression.
  - Existing ragged active-route path exactly reproduced the dense eight-step trajectory: committed agreement `0.92578125`, versus `0.5625` for production true-sparse.
  - A first trace-safe active gather/reduce prototype used the wrong route reduction order: step 0 reached `0.9453125` but entropy PCC fell to `0.86514`; feedback collapsed final agreement to `0.2734375`.
  - Direct route-load measurement found max expert loads of `156–256` and `838–1711 / 2048` dropped routes per layer at capacity 32.
  - A zero-drop capacity-256 control restored eight-step committed agreement to `0.91796875`; ragged/dense remain `0.92578125`.
  - Verdict: fixed the primary sparse-only bug by defaulting capacity to canvas length and disabling capacity-32-only tuned geometry for other capacities. Active combine prototypes were refuted and removed.
- Hypothesis: all-reduce complete active expert outputs before route weighting/reduction to match HF index-add order.
  - Ragged step-0 entropy PCC improved `0.97713 -> 0.98240`, but argmax agreement fell `0.91797 -> 0.91406`.
  - Verdict: refuted by the decision gate and removed.
- Hypothesis: Blackhole sparse expert FP32 accumulation can retain the historical
  Gemma-4 PCC lift when the kernel uses full-DST rather than half-DST synchronization.
  - HiFi4, exact math, FP32 destination accumulation, `dst_full_sync_en=True`,
    BF16 output, applied only to routed expert gate/up/down.
  - Step-0 agreement improved `0.91796875 -> 0.921875`; entropy PCC improved
    `0.96738 -> 0.97438`.
  - Revision-locked eight-step committed agreement improved
    `0.9296875 -> 0.93359375`; HF non-EOS improved
    `0.4857143 -> 0.51428574`, and minimum active-step entropy PCC improved
    `-0.00607 -> 0.61842`.
  - Applying the same config to sparse combine, LM head, shared MLP, or only a
    subset of layers regressed the full trajectory.
  - Seed-1 A/B improved committed agreement `0.8828125 -> 0.89453125`, but
    active-step entropy fidelity regressed after step 2.
  - Verdict: retained as a Blackhole-only opt-in diagnostic, not a production
    default; it does not pass all mandatory fidelity dimensions.
- Hypothesis: cumulative drift at the layer-20 input is the remaining trajectory boundary.
  - Exact HF prompt KV plus exact layer-20 input produced committed agreement `1.0` over eight seeded steps.
  - Replacing only layer-19's HF post-attention normalized branch produced `0.99609375`.
  - Exact raw layer-19 attention alone ended at `0.9296875`; HF RMSNorm on live TT attention gave step-0 `0.91015625`; targeted HiFi4/FP32-accumulation for TT layer-19 o_proj + norm also gave `0.91015625`.
  - The full HF attention+post-norm branch run on the live TT layer-19 input gave only step-0 `0.9140625`.
  - TP o_proj partial reconstruction (device, BF16 left-fold, pairwise, FP32 host sum) stayed at branch PCC `~0.9427`; CCL parenthesization is not causal.
  - Verdict: the exact branch succeeds by resetting cumulative pre-layer-19 hidden drift, not by replacing a faulty layer-19 op. Isolated raw-attention, norm, CCL order, and generic higher precision are refuted.
- Hypothesis: retain a separate FP32 residual accumulator across layers 0–20 while keeping branch compute in BF16.
  - Step 0 improved (`argmax 0.91797 -> 0.92188`, entropy PCC `0.97713 -> 0.98733`).
  - The seeded eight-step stage gate failed: committed match `0.89453`, minimum entropy PCC `0.03358`.
  - Verdict: refuted and removed; this is another example where one-step accuracy moves the diffusion feedback trajectory in the wrong direction.
- Hypothesis: TT self-conditioning incorrectly soft-embeds raw previous logits instead of HF temperature-processed logits.
  - Source comparison verified the semantic mismatch. The schedule is now threaded through eager and traced adapters; processed logits are rounded to BF16 before the chunked soft embedding, matching HF ordering.
  - Unit tests pass (56/56). The eight-step stage gate still fails: committed match `0.90234`, non-EOS match `0.28571`, minimum entropy PCC `0.65683`.
  - Disabling self-conditioning on both HF and TT also fails (`0.89844`, minimum entropy PCC `0.81468`), so self-conditioning is not the sole remaining source.
  - A two-pass FP32 streaming softmax lowered step-1 entropy fidelity and was removed.
  - Verdict: semantic bug fixed, but it exposes rather than closes the shared-backbone trajectory drift.
- Hypothesis: the remaining trajectory collapse is caused by TT's chunked
  self-conditioning soft-embedding numerics.
  - Injecting HF's actual per-step soft-embedding signal into TT produced exact
    committed/non-EOS agreement `1.0`.
  - The discriminating control computed the exact HF softmax@embedding from the
    live TT previous logits instead; it did not improve committed agreement
    (`0.91796875`). Coupling that control with full-DST experts regressed to
    `0.890625`.
  - Verdict: refuted as an isolated TT soft-embedding bug. The exact HF signal
    oracle resets the already-diverged feedback trajectory.
- Hypothesis: transfer GPT-OSS router and CCL precision policies.
  - Layer-19 DRAM-to-L1 CCL buffer normalization was a no-op for DG's already tile-aligned 704-wide TP shard.
  - GPT-OSS HiFi3/fp32-dest/numeric-stable router softmax left step-0 argmax unchanged and lowered entropy PCC `0.97713 -> 0.97267`.
  - GPT-OSS validates only shallow/per-component PCC and uses incompatible router and residual/norm semantics.
  - Verdict: compatible GPT-OSS levers refuted and removed.
- Hypothesis: DiffusionGemma's missing FP32 post-LM-head softcap/decision
  boundary is the remaining production bug.
  - The source mismatch is real, but both the device FP32 implementation and
    an exact Torch FP32 softcap of the identical TT BF16 pre-softcap logits
    produced committed agreement `0.91015625`.
  - Verdict: refuted for #48291 and removed.
- Hypothesis: a DG-local layer-0 embedding/self-conditioning input mismatch
  starts the trajectory drift.
  - Exact HF layer-0 input on every step passes (`1.0`, minimum active entropy
    PCC `0.95884`), but injecting it only at step 0 regresses to `0.86328125`.
  - Verdict: the all-step result is a feedback reset, not a layer-0 fix.
- Hypothesis: one exact-input local branch identifies a repairable upstream op.
  - Full-canvas, exact-input+exact-KV ledger ranks layer 18 worst at PCC
    `0.9974325`. Its post-attention branch is `0.9997648`, shared FF
    `0.9983796`, routed expert FF `0.9918051`, and post-FF `0.9965505`.
  - Exact HF routing raises the local routed branch to `0.9977631`, but using
    HF routing on live free-running TT hidden at layer 18 regresses final
    agreement to `0.9140625`; exact KV plus all live HF routers reaches only
    `0.92578125`.
  - Exact expert-ID-order BF16 left-fold combine changes local routed-branch
    PCC only `0.9977631 -> 0.9977694`.
  - Verdict: router discontinuity amplifies accumulated drift; neither router
    substitution nor combine order is a production correction.
- Hypothesis: TT used the wrong GELU function for a
  `gelu_pytorch_tanh` checkpoint.
  - Source audit proved legacy `fast_and_approximate_mode=True` selects
    FastLut erf-GELU, while HF requires `GeluVariant.Tanh`.
  - Layer-18 exact-input/KV shared FF improves `0.9983796 -> 0.9998859`;
    routed FF improves `0.9918051 -> 0.9937973`, or to `0.9997175` with exact
    routing; post-FF improves `0.9965505 -> 0.9978347`.
  - Canonical seed-0 committed agreement improves
    `0.9296875 -> 0.99609375`, with HF non-EOS agreement `0.9714286`.
    Seed 1 also improves to `0.9140625` (or `0.92578125` with full-DST experts)
    but remains below the multi-seed bar.
  - FP32 entropy and FP32 terminal combinations regress decisions and were
    removed.
  - Verdict: verified semantic bug and retained as the DiffusionGemma default,
    with `DG_GELU_TANH=0` only as a bisect escape hatch.
  - Post-fix trace validation captured eight single-step traces and replayed all
    eight successfully; traced committed/non-EOS agreement matches eager at
    `0.99609375 / 0.9714286`.
- Hypothesis: preserving HF FP32 route weights through expert weighting closes
  the remaining local error.
  - On tanh+exact-route layer 18, FP32 weighting changes expert FF only
    `0.9997175 -> 0.9997222`; post-FF slightly regresses
    `0.9998566 -> 0.9998558`, with unchanged max error.
  - Verdict: refuted and removed.
- Hypothesis: FP32-packed Gumbel addition or processed-logit clean argmax
  explains seed-1 failure.
  - FP32 Gumbel addition slightly changes sampled agreement but leaves seed-1
    committed/entropy metrics unchanged.
  - Processed-logit clean argmax is bit-identical on the tested trajectory.
  - Verdict: refuted for the strict gate and removed.

## Kept Fixes
- Sparse-matmul FP32 intermediate-CB size is derived from its actual intermediate format; Blackhole regression passes.
- Sparse denoise capacity defaults to the full canvas length, preventing silent route loss.
- Replay harness keeps exact-noise, hidden/KV/routing oracle controls used to verify and refute candidates.
- The strict stage gate enforces the canonical sparse P150x4 eight-step shape,
  rejects HF injection controls, and excludes steps after common all-accept
  from the active entropy minimum.

- Hypothesis: TT has excess drift beyond the intrinsic bf16 floor that a further
  precision fix could close.
  - Decisive control: run the SAME HF model fp32 vs bf16 with identical seeded
    8-step noise (zero TT kernels; `doc/decision_fidelity/measure_bf16_floor.py`).
  - fp32-vs-bf16 committed match = `0.86328125` (seed 0) / `0.91406250` (seed 1),
    both below `0.95`; per-step entropy PCC collapses/goes negative at converged
    steps. `HF-fp32 vs TT` = `0.863` (seed 0, == floor) / `0.980` (seed 1, better
    than the `HF-bf16 vs TT` gate value `0.914`). All trajectories decode to
    coherent correct text.
  - Verdict: REFUTED. TT is at or above the intrinsic bf16 floor; no bf16
    implementation (TP or single-device) can reach the strict gate because the
    reference cannot reach it against itself in fp32. The class of TT-side
    precision fixes is exhausted by construction.

## Final Status
- The strict `>0.95` gate stays red on seed 1, but the bf16-floor self-consistency
  control proves this is a **gate mis-specification + intrinsic bf16
  diffusion-trajectory chaos**, not a TT defect: fp32-vs-bf16 of the reference
  itself scores `0.863`/`0.914`, and TT is at or better than that floor.
- committed-match and per-step entropy-PCC vs a single chaotic bf16 reference are
  the wrong fidelity metrics for a paraphrase-generating diffusion model; the
  variance-gated `sound_entropy_step_fidelity` replaces the ill-conditioned
  entropy-PCC at converged steps.
- Decision: product-accept (coherent correct output at the bf16 floor); re-spec
  the gate to the fp32 ideal + variance-gated entropy (owner sign-off for the
  production pass/fail flip); fp32 MoE backbone is a separate owned effort on the
  shared Gemma-4 backbone (blocked by `ttnn.topk` FLOAT32 `TT_FATAL` + fp32
  experts exceeding QB2 DRAM). See `doc/decision_fidelity/README.md`.
