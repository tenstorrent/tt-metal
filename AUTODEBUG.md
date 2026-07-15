# AutoDebug: DiffusionGemma decision fidelity (#48291)

Date: 2026-07-13
Scope: inspection only; no hardware reproduction was attempted.

## Executive status

No inherent fidelity ceiling is proven, and no production fix is currently validated.

The best-supported first repair experiment is a **DiffusionGemma-local attention precision
island** in `tt/diffusion_attention.py`, tested on the real production denoise path. The local
denoise attention does not inherit PR #48748's HiFi4/fp32-destination SDPA configuration and can
fall back to an unconfigured `matmul -> softmax -> matmul` implementation. This is a smaller and
more relevant intervention than editing shared Gemma4 or immediately changing the flash-SDPA
kernel.

Before selecting any fix, `demo/replay_hf_tt.py` must be corrected: despite claiming replayed
noise, it injects **all-zero Gumbel tensors and all-zero renoise token IDs**. That is useful for a
clean-argmax probe, but it is not the requested deterministic real-checkpoint stochastic denoise
trajectory.

## Direct observations

1. PR #48748 is present in this branch (`16d59b800ee`). It configures shared Gemma4 causal-prefill
   SDPA with HiFi4 and `fp32_dest_acc_en=True`, and shared prefill experts with HiFi4 and
   `fp32_dest_acc_en=False`.
2. The post-PR QB2 TP=4 causal probe still measured logits PCC `0.8405204` and argmax `3/5`
   (`terminals/381199.txt`). This proves the shared causal backbone remains materially different
   from the HF DiffusionGemma checkpoint on that prompt; it does not isolate the first bad module.
3. The historical non-degenerate one-step replay found clean-argmax misses at canvas positions
   `[2,3,4]`: HF chose token `496`, TT chose EOS `1`; both sides rejected those positions. The
   miss therefore existed in logits before entropy acceptance or renoise (`plan.md`, R0.5).
4. The later absolute 16-step comparison recorded only `43/256` committed agreement for the
   default chunked-norm path, mean per-step argmax agreement `0.541`, accept IoU `0.100`, and
   entropy PCC `0.027` (`doc/optimize_perf/l1_residency.md`).
5. `demo/replay_hf_tt.py` builds both HF and TT trajectories with zero Gumbel noise and zero
   renoise IDs. It does not generate and replay a seeded real Gumbel/renoise stream.
6. Shared Gemma4 prefill attention passes an explicit HiFi4/fp32-destination compute config.
   DiffusionGemma denoise attention passes no compute config to either fused SDPA or its fallback:
   - fused: `ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, **kwargs)`;
   - fallback: BF16-default QK matmul, softmax, and PV matmul.
7. HF eager attention performs softmax in fp32 and then casts probabilities to the query dtype.
8. The SDPA C++ factory consumes `fp32_dest_acc_en`, but its intermediate/statistics CB format is
   still hard-coded to `Float16_b`. Therefore `fp32_dest_acc_en=True` is not equivalent to fp32
   softmax statistics and fp32 PV throughout the operation.
9. The production-selected local true-sparse MoE defaults to HiFi2 with fp32 destination
   accumulation disabled. Its fixed capacity `C=32` silently drops assignments above capacity.
   Existing verification measured representative layer-0 routing, not every layer/step of a real
   denoise trajectory.
10. Device `ttnn.topk` accepts only `BFLOAT16` or `BFLOAT8_B`
    (`topk_device_operation.cpp:156-159`). A true fp32 on-device routing-selection experiment is
    not available through this op.
11. Encoder and decoder `layer_scalar` values in the real checkpoint are exactly equal for all
    30 layers. Ignoring the encoder copy is not the observed cause.

## Interpretations

- The causal PCC result shows that numerical drift exists before diffusion sampling, but it does
  **not** prove that MoE, attention, or a specific layer is the first cause.
- Plain Gemma4 diagnostics from the earlier investigation are consistent with small hidden-state
  drift accumulating until routing top-k amplifies it: a baseline near `0.932`, no improvement from
  fp32 attention/router/lm-head weight containers, and a diagnostic uplift near `0.97` from HF
  routing. Those runs used another checkpoint/prompt/path and are not DiffusionGemma denoise proof.
- The no-op `attention=fp32` result does not refute an fp32-attention-arithmetic fix. The checkpoint
  weights are BF16, and that override did not make SDPA softmax/PV or the residual stream fp32.
- The optimize skill's claim that “full-fp32 attention alone lifts logits PCC >=0.92” has no
  committed reproduction artifact. It should be treated as an unverified proposal, not evidence.
- Router top-k is likely an amplifier, not the first cause. CPU routing on already-drifted TT hidden
  states was worse, and fp32 device top-k is unsupported.

## Causal chain

1. Prompt prefill writes BF16 frozen K/V.
2. Each denoise layer computes local Q/K/V, attention, residual, shared MLP, router, and experts.
3. Small attention/norm/residual differences accumulate in the hidden stream.
4. BF16 router scores cross a top-k boundary; a continuous error becomes a different expert set.
5. Different expert outputs amplify later hidden/logit drift.
6. Clean argmax flips directly change the commit candidate.
7. Entropy drift separately changes accept/renoise positions, feeding a different canvas and
   self-conditioning signal into later steps.
8. The final clean argmax has no sampling cushion, so trajectory divergence becomes committed text.

The earliest target-specific divergence currently proven is the final causal-backbone output and
the step-0 denoise logits at positions `[2,3,4]`. A real DiffusionGemma per-layer capture has not yet
identified the first bad layer/op.

## Ranked hypotheses

### H1 — DiffusionGemma-local denoise attention arithmetic is the first actionable source

Confidence: medium; highest-priority experiment because it is local and directly testable.

Supporting code:
- PR #48748 improved shared prefill attention precision, but the local denoise implementation does
  not reuse its compute config.
- The QB2 denoise path historically selected the staged GQA fallback after an SDPA L1-CB clash.
- The fallback computes QK, softmax, and PV with default BF16 intermediates, whereas HF softmax is
  fp32.
- Existing fallback coverage is synthetic (`4` Q heads, `2` KV heads, sequence `32x64`,
  head dimension `32`, PCC only `>=0.99`), not the production 30-layer decision trajectory.

What remains unproven:
- Which layers currently select fused SDPA versus fallback after the full-canvas optimization.
- Whether attention is the first per-layer divergence on real DiffusionGemma inputs.

### H2 — BF16 residual/norm accumulation causes gradual hidden drift

Confidence: medium.

Supporting evidence:
- Plain Gemma4 teacher-forced individual layers were high PCC while free-running hidden states
  decayed with depth, consistent with accumulation rather than one broken expert op.
- `DG_NORM_FULLCANVAS` shows that tiny reduction-order changes can radically change the trajectory,
  although it was not better against HF.

This suggests a narrow fp32 output/residual island around the first divergent layers may help, but
the first divergent DiffusionGemma layer must be measured before choosing the island.

### H3 — Current true-sparse MoE adds production-only error

Confidence: medium-low as the original cause; medium as a current regression risk.

- The original #48291 evidence predates the current optimized sparse path, so sparse MoE cannot
  explain the original blocker.
- Current production uses HiFi2 local matmuls and capacity 32. Capacity overflow is silently
  dropped, and dense-vs-sparse PCC `0.9997` is not a decision-fidelity guarantee.

### H4 — Router/top-k precision itself is the root cause

Confidence: low.

HF-routing injection is useful localization, but not a fix. Device fp32 top-k is unsupported, fp32
weight containers add no checkpoint precision, and CPU routing on TT hidden did not recover the
plain Gemma baseline. Fix upstream hidden drift before considering a new fp32 router/top-k op.

### H5 — An upstream flash-SDPA kernel change is unavoidable

Confidence: unproven.

The C++ kernel's BF16 intermediate/statistics format is a credible limitation. However, the current
DiffusionGemma path may use the manual fallback, and a local fp32 manual-attention experiment has
not been run. Upstream work is justified only if the local production-shaped experiment proves no
legal local path can recover decisions.

## Focused verify/refute experiments

### E0 — Repair the acceptance gate first

Modify only the DiffusionGemma replay harness:

1. Generate one seeded initial canvas, one seeded fp32 Gumbel tensor per step, and one seeded random
   renoise-token tensor per step.
2. Reuse those exact host tensors for HF and TT via the existing replay hooks.
3. Use a non-degenerate prompt and at least 8 steps; disable early halt for candidate comparisons.
4. Record clean argmax, Gumbel sample, entropy, accept mask, canvas, and committed tokens per step.
5. Require input hashes in the artifact so two runs cannot silently compare different noise.

### E1 — Locate the first real denoise divergence

On step 0 (no self-conditioning), capture HF and TT per layer for the same real checkpoint, prompt,
canvas, prefix K/V, and positions:

- layer input and input RMSNorm;
- Q/K/V projection, per-head norm, and RoPE;
- QK scores, probabilities, PV output, output projection;
- post-attention residual;
- shared MLP output;
- router scores, top-k indices/weights;
- expert output and final layer residual.

Report global PCC plus positions `[2,3,4]`, and log whether each TT layer used fused SDPA or the
manual fallback. This experiment can falsify H1.

### E2 — Attention precision matrix at the production shape

Keep weights, canvas, noise, MoE, norms, and trace mode fixed. Compare:

1. current fused/fallback behavior;
2. fused SDPA with the same HiFi4/fp32-destination config as #48748;
3. forced manual attention with current BF16 intermediates;
4. manual attention with fp32 QK output, fp32 softmax, and fp32 PV accumulation/output;
5. candidate 4 plus fp32 QKV/o-projection accumulation (BF16 checkpoint weights remain unchanged).

Select only by E0's HF-vs-TT decision trajectory. A one-layer PCC improvement is insufficient.

### E3 — Residual precision island

If E1 shows attention output is close but post-residual drift grows, retain fp32 only across
attention/FF residual adds and RMSNorm input/output for a small layer window around the first
divergence. Expand the window only if the trajectory improves.

### E4 — MoE controls

- Compare production sparse MoE against dense MoE on the same captured real inputs at every layer.
- Report capacity overflow counts for every layer/step; compare capacity 32 versus 64.
- Compare local HiFi2 versus HiFi4 compute while keeping routing fixed.
- Use HF routing only as a diagnostic control, never as a candidate fix.

### E5 — Upstream decision

Pursue a TTNN kernel fix only if E2 proves:

- fused SDPA is the selected failing path;
- local manual fp32 attention proves the arithmetic is causal but cannot be a viable shipping path;
- current SDPA cannot represent the needed fp32 softmax/PV path because of its BF16 intermediate
  formats.

The upstream boundary would be TTNN SDPA intermediate/statistics and PV accumulation, not
`models/demos/gemma4`.

## Smallest intervention boundary

1. **Required test fix:** `models/experimental/diffusion_gemma/demo/replay_hf_tt.py`.
2. **First model candidate:** `models/experimental/diffusion_gemma/tt/diffusion_attention.py`,
   adding a local precision policy/A-B path without touching shared Gemma4.
3. **Second candidate if measured:** local residual/norm precision in
   `models/experimental/diffusion_gemma/tt/denoise_forward.py`.
4. **Only after proof:** TTNN SDPA kernel changes outside the model directory.

## Recommended first fix experiment

Implement E0 and one local `manual_fp32` attention candidate together:

- real seeded Gumbel and random renoise replayed identically through HF and TT;
- production 30 layers, TP=4, true-sparse tuned MoE, one non-degenerate prompt;
- step-0 per-layer capture plus an 8-step trajectory;
- baseline versus fp32 QK/softmax/PV in the local staged-GQA path;
- accept only if committed clean-argmax agreement, per-step argmax, entropy PCC/max error, and
  accept IoU all improve without a new capacity drop or trace failure.

This is the shortest experiment that can turn the current attention theory into a real
DiffusionGemma fix or refute it. It stays within the required local directory and avoids prematurely
claiming an unavoidable upstream kernel change.

## Other code issues not claimed as the short-prompt root cause

- Long-prefix sliding layers require HF-style bidirectional sliding visibility. The production
  default remains maskless unless explicitly enabled; this becomes semantically wrong once the
  frozen prefix exceeds the sliding window, though it does not explain the short-prompt miss.
- Reproducing the stated production path depends on `DG_SPARSE_MOE=1`; the code default remains
  the slower dense path.
- Fixed sparse capacity can silently drop hot-expert assignments; current production needs
  trajectory-wide overflow evidence, not only representative layer-0 evidence.
