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

## Kept Fixes
- Sparse-matmul FP32 intermediate-CB size is derived from its actual intermediate format; Blackhole regression passes.
- Sparse denoise capacity defaults to the full canvas length, preventing silent route loss.
- Replay harness keeps exact-noise, hidden/KV/routing oracle controls used to verify and refute candidates.

## Final Status
- Still failing the `>0.95` production trajectory gate.
- Higher one-step router fidelity is not a valid proxy for the multi-step diffusion trajectory.
- The next candidate must preserve the baseline trajectory while reducing continuous TP/BF16 residual drift; broad or selective router replacement is exhausted.
