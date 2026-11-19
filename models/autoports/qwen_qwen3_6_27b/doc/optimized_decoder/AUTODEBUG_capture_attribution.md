# Actual-input attribution: chunk versus direct GDN

Analyzed caller-produced `capture_chunk.json`, `capture_chunk/layer0_step{19,58,99}.pt`, and `capture_direct.json`, plus CPU-only real-weight counterfactuals. No device execution or decoder source edits were performed. The numeric gates remain unchanged.

## Conclusions supported by the captures

**Native recurrent-state attenuation accumulates, then sensitive heads and the downstream residual/MLP amplify it. MLP quantization and fused SiLU rounding are secondary at the captured chunk failures.** Direct FP32 recurrence removes the large steps58/99 dips but leaves separate upstream/prefix differences. These are two issues, not one unproven generic “instability.”

All three selected eager reruns are bit-identical to their ordinary trace replay for output, convolution state and recurrent state. The ordinary traced PCC values reproduce the earlier failed coordinates. This excludes an eager/trace difference at these samples.

| Metric | step19/user20 | step58/user30 | step99/user27 |
|---|---:|---:|---:|
| Native chunk output PCC | .993091 | .970396 | .975729 |
| Direct output PCC | .993753 | .998585 | .999277 |
| Native prior-state relative RMS error versus HF | 2.89% | 3.84% | 7.77% |
| Direct prior-state relative RMS error versus HF | 1.63% | 1.56% | 2.00% |
| Native current core versus own-input FP32 oracle PCC | .99999979 | .99999969 | .99999980 |
| Direct current core versus own-input FP32 oracle maximum error | 3.73e-9 | 1.49e-8 | 7.45e-9 |

Both candidates start from identical prefix state error: PCC .99991098, relative RMS 1.563%. Native's own-input updated-state error is approximately .11% per step. Direct's own-input updated state agrees to at most 2.38e-7 absolute, effectively PCC1. Thus the direct implementation's recurrence order/arithmetic is verified against the CPU formula at the observed inputs; further rewrites of that formula will not address its remaining .991549 minimum sequence PCC.

At step58, heads12 and14 account for 39.26% and45.04% of total gated-output squared error. Their HF core RMS values are .0004375 and .0003898. Head14 core PCC is .85454 and gated PCC .40128; this is measured amplification. Their true decay factors are .9998747/.9998413, yet native updated state versus its same-input CPU oracle has least-squares gains .9985205/.9989253. Their inherited TT state versus HF has gains .96124/.96540. At step99, head7 inherited-state gain is .91655. This is compatible with repeated matrix-operand truncation/attenuation; the captures do not identify which low-level unpack/exp/diagonal-matmul stage is responsible. **Do not introduce an empirical state multiplier**: it would change the recurrence and cannot correct the directional error.

Crossing states and inputs supports the state attribution. At step58, HF inputs with TT old state give 3.164% core relative RMS error; TT inputs with HF old state give .723%. For head14 those errors are 52.47% versus2.15%. At step99, whole-core errors are7.522% versus1.029%. Fixing only the current token's projections cannot repair the native path's accumulated prior-state error.

## CPU real-weight downstream controls

`tests/analyze_decode_capture.py` loads only the pinned checkpoint's three layer0 MLP weights and existing selected-user artifacts. It imports no TTNN. Results are saved in `capture_chunk_cpu_attribution.json`.

HF MLP recomputed from the captured HF post-attention norm reconstructs the HF decoder output at PCC at least .9999999869, providing a strong control despite tiny CPU batch-shape/reduction rounding differences.

| CPU counterfactual final output | step19 | step58 | step99 |
|---|---:|---:|---:|
| Exact BF16 HF MLP on captured TT post-attention norm + TT residual, versus HF output | .992151 | .972719 | .977539 |
| Same counterfactual, versus actual TT output | .999941 | .999937 | .999944 |
| CPU fused-SiLU variant, versus HF output | .992152 | .972743 | .977534 |

Replacing the TT MLP with exact HF computation on its actual input preserves the failure. MLP input sensitivity/residual cancellation explain most final-output amplification; MLP weight precision/arithmetic is not the primary cause at these coordinates. Suppressing the intermediate BF16 SiLU rounding changes these CPU PCCs by at most .000024. This does not prove a `round_silu` device control useless on other inputs, but it refutes it as the principal explanation of the captured .9704/.9757 dips.

Likewise, exact CPU HF gated RMSNorm on captured TT core/z agrees with the TT gated output at PCC .9999878/.9999875/.9999940. Its agreement with HF gated output remains .999185/.997754/.997877. Most GDN gate error already exists in core/z, rather than arising in the post-core fused arithmetic.

CPU one-user recurrence recomputation from captured HF inputs is not perfectly bit-equal to the original full-batch CPU HF output (maximum BF16 differences7.63e-6/4.88e-4/3.05e-5). Interpret oracle differences relative to that floor; do not claim the standalone oracle reproduces every CPU reduction bit. The very large crossed-state effects and direct-versus-own-input agreement exceed this qualification.

## Concrete next fixes/controls

1. Retain direct FP32 recurrence as the demonstrated accurate arithmetic control. A fast replacement must avoid repeated low-precision matrix reads of recurrent state; merely changing wave geometry, cache dtype or current projection fidelity does not establish that. Do not retain a faster native implementation that still fails the unchanged gate.
2. For the remaining direct failures, target the inherited prefix and upstream projections/norms. The prefix state already differs1.56% before decode. Compare direct decode with a more faithful prefix and with HF-parity convolution/input normalization; run the **same input fixture**. BF16 attention weights plus standalone small projections are justified as an attribution control, not automatically the final default policy.
3. The direct path's failed step19 should be inspected separately from repaired58/99. Its old-state and current-input errors are comparable (HF-input/TT-state core1.29%, TT-input/HF-state1.17%). No single current-op tweak is proved sufficient.
4. Treat MLP/fused-SiLU and output-norm controls as secondary unless new measurements contradict the CPU attribution above. These controls should not postpone state-fidelity work.

## Real-text S33 functional-state discrepancy

`compact_profile_fixed_bfp8_0.json` records optimized HF output PCC .999992 prefill/.999991 decode, versus functional output .998476/.999106 and functional recurrent-state PCC .06303984. That last comparison is after **prefill plus one decode**, not prefix state alone (`tests/run_decoder.py` functional-equivalence ordering). The record contains no HF persistent-state comparison. It cannot establish whether functional or optimized state is correct, nor whether the error originates in prefill or the next update. A few large erroneous heads could dominate global state PCC while output remains close, because a token reads only one projection of state.

The functional prefix computes a 32-row inverse using repeated matrix squaring/products, while optimized native uses dedicated GDN prep/scan. Highly correlated real-text inputs can expose numerical differences in this solve that independent random inputs do not. This is a plausible, **unverified** explanation; there is no captured HF/functional/native state tensor yet to substantiate it. Do not waive the .995 functional-state gate on output PCC alone.

Added `tests/run_state_reference.py` to compare HF and the selected TT implementation's output, convolution state, and recurrent state immediately after prefill, decode1 and decode2. It reports every user/head, state RMS, relative error, and gain; optional selected-user tensors permit pairwise inspection. It supports all existing `QWEN_DECODER=functional/fused/optimized` selections and normal candidate/policy variables. Run each implementation in its own serial device invocation:

```bash
QWEN_DECODER=functional python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_state_reference \
  --layer 0 \
  --activation-file models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/text_activations/text_l0_b32_s33.pt \
  --user 0 --user 27 --user 30 \
  --dump-intermediates models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/state_functional \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/state_functional.json
```

Repeat with fused/optimized and distinct output paths, keeping optimized policy identical to the failing profile. This is diagnostic evidence, not a substitute state acceptance reference. Existing stage gates are unchanged. Syntax compilation passed; the new state runner has **not** been executed on hardware by this agent.
