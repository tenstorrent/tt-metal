# CPU quantization noise budget

Status: the parent executed the study in a CPU-safe slot; all 20 cases in
[quant-noise-budget-v1.jsonl](quant-noise-budget-v1.jsonl) completed. Startup
numerical checks, finite-output gates, exact coverage and original-input/source
immutability pass. The fetched record SHA-256 is
`be73a15a60b5b89965412d8eca7c81b6a5d21c96e5712a417b436af807345b4a`.
All six recorded source pins matched local source bytes at this audit, before
any subsequent formatting. This is a representation-only study, not a Blackhole
emulator, performance measurement, rigorous error lower bound, or model-quality
evaluation.

## Measured result: diffuse attention retains a substantial noise budget

These seed-1240 rows use FP64 outputs and the original-input FP64 reference;
all entries are relative L2 percentages. `V corrected` and `KV corrected`
subtract the same exact FP64 `mean(Vq - V)` before output rounding.

| Codec | N | K only | V only | V corrected | KV | KV corrected |
|---|---:|---:|---:|---:|---:|---:|
| TT RNE BFP4 | 4,096 | 11.864 | 11.325 | 9.283 | 16.341 | 15.066 |
| TT RNE BFP4 | 32,768 | 11.855 | 11.337 | 9.301 | 16.360 | 14.969 |
| TT RNE BFP4 | 262,144 | 12.026 | 11.615 | 9.275 | 16.697 | 15.134 |
| Representative NVFP4 recipe | 4,096 | 9.774 | 9.298 | 7.505 | 13.346 | 12.260 |
| Representative NVFP4 recipe | 32,768 | 9.826 | 9.789 | 7.478 | 13.936 | 12.445 |
| Representative NVFP4 recipe | 262,144 | 9.915 | 9.752 | 7.701 | 13.930 | 12.529 |

The second seed at 4K and 32K supports the same band: across all five normal
input sets, TT KV error is 16.34–16.70%, versus 14.97–15.16% after correction.
TT K/V reconstruction errors are each 11.70–11.75%, giving root-sum-square
predictions around 16.58%, close to the measured attention errors. The
representative NVFP4 recipe reconstructs K/V at 9.50–9.52% and produces
13.35–14.04% KV error, or 12.10–12.75% after correction. This does not claim
those errors for SageAttention or an NVIDIA device kernel.

Final BF16 rounding does not explain the large errors: across the ten normal
codec/input cases and five quantized output scopes, it changes reported L2 by
at most 0.00615 percentage points. At 256K, TT KV is 16.6979% with BF16 output,
and corrected KV is 15.1358%. The unquantized normal-input BF16-output floor is
0.163–0.168%. Both output-precision metric sets are retained in the record.

### What the heuristic gets right, and what it does not establish

Measured `N * sum(P^2)` is 2.701–2.769, close to `e`; the corresponding
independent-noise value-centering proxy is 0.7936–0.7993. TT's actual retained
V-error amplitude is 0.7985–0.8204, broadly consistent but not identical.
The NVFP4 recipe gives 0.7640–0.8072. There are only five input sets and 64 query
rows per set, not a confidence interval or proof of independence.

The measured K-only and V-only output-error cosines are near zero
(−0.0173 to +0.0167 across both codecs), supporting root-sum-square reasoning
for these inputs. Noise is nevertheless not perfectly independent or unbiased:
representation gain is approximately 0.9945–0.9960, and error/input correlation
is about −0.043 to −0.047 for TT. The exact nonlinear cross term
`(Pq - P) @ (Vq - V)` alone has 1.38–1.42% relative L2 for TT, so it is not
literally absent even when its effect on the combined norm is small.

The first-order softmax model predicts the combined error magnitude closely:
its KV L2 differs from exact quantized attention by at most 0.061 percentage
points for TT and 0.071 for the NVFP4 recipe. But matching an error norm is not
the same as predicting every error element. TT's K-only error-vector cosine
is 0.9960–0.9965, with an 8.35–8.97% discrepancy relative to the **error vector**;
the first-order KV vector discrepancy is 10.44–10.68%. Retaining the explicitly
second-order joint term `dp_linear @ (Vq - V)` reduces that KV discrepancy to
6.06–6.51%. These residuals quantify the linearization's limits, rather than
turning it into an exact arithmetic model.

With uniform-zero Q, K-only error and corrected V/KV error are exactly zero in
the recorded FP64 calculations for every case. Corrected BF16 metrics exactly
match the unquantized BF16-output floor, 0.156–0.201% across the five input
sets. Uncorrected uniform V error still varies by seed: TT is 9.90–11.70%.
That is expected finite-dimensional variation in the ratio of mean-error and
mean-value norms, not a failure of the exact mean-correction identity.

The practical conclusion is bounded: **for these fixed codecs and normal
inputs, eliminating all hardware arithmetic error still leaves a large
representation error**. Subtracting the uniform value-error component helps,
but cannot account for the remaining nonuniform V noise or K-induced changes
to attention weights. This is strong evidence for the proposed noise-budget
explanation on this input family, not a lower bound across alternative codecs,
scales, transforms, residual representations, or model-derived activations.
No clipping/outlier/common-mode stress cases were added to this small study.

## Experiment

`quantization_noise_budget.py` reuses the pinned CPU codecs and reference in
`codec_limits_models.py`, without importing TTNN or calling a device. It starts
with original normal BF16 square Q/K/V, head dimension 128, one head, and 64
explicitly recorded sampled query rows. Defaults cover N = 4,096 and 32,768 with
seeds 1240 and 1241, and N = 262,144 with seed 1240. The optional 128-row setting
and second long-context seed cost additional CPU time.

Both K and V are grouped along D in groups of 16. The two default codecs are
qualified TT RNE BFP4 and the existing representative E2M1/E4M3/global-FP32
NVFP4 recipe. The latter is not a complete SageAttention algorithm or an NVIDIA
hardware execution. Q is unchanged BF16, with no Q7 preprocessing. All QK,
softmax, P, PV, correction and recurrent reference arithmetic is FP64; in
particular this study does not quantize P or emulate LoFi phase truncation.

Every codec/input combination measures six outputs against the original-input
FP64 reference: unquantized control, K-only, V-only, KV, and the V-only/KV
outputs minus the per-feature token mean of `Vq - V`. Each output has both FP64
and final-BF16 metrics. A second query mode replaces Q with exact zeros, so its
softmax weights are uniform. It uses the same original K and V.

The correction is subtraction of `mean(Vq - V)` after attention, before final
BF16 rounding. It is **not** centering original V and then requantizing it, and
does not assume these two pipelines are numerically interchangeable. Both V
and its represented values are available to a hypothetical preprocessing step;
the study uses their exact FP64 mean and does not model the implementation cost
or precision of computing/storing that mean.

## Why a context-independent error scale is plausible

For one query, let `O = sum_j p_j V_j`. Assume independent, centered additive
value errors with per-component variance `sigma_V^2`, independent of the
weights, and independent unit-variance V. Conditional on the weights:

```
Var(O)       = sum_j p_j^2
Var(delta O) = sigma_V^2 * sum_j p_j^2
```

Consequently a ratio of concentrated aggregate squared norms is approximately
`sigma_V`, even as context grows. A softmax-Jacobian linearization gives a
similar approximate `sigma_K` scale for small independent K noise with
unit-variance Q. If the resulting K and V output errors are also approximately
orthogonal, their combined relative error is roughly
`sqrt(sigma_K^2 + sigma_V^2)`. Reconstruction errors of 11.7% for each therefore
suggest about 16.5–16.6% combined attention error, not 11.7% divided by the square
root of the context length. The reference itself shrinks as the context grows.

Subtracting the token mean of value error replaces its weights by `p_j - 1/N`:

```
Var(delta O_centered) = sigma_V^2 * (sum_j p_j^2 - 1/N)
retained V-error amplitude = sqrt(1 - 1/(N * sum_j p_j^2))
```

For diffuse independent Gaussian logits of variance `tau^2`, large-N
`N * sum_j p_j^2` approaches `exp(tau^2)`. With unit variance, the retained
amplitude is `sqrt(1 - 1/e)`, about 0.795. An 11.7% value error would then become
about 9.3%; combining it with unchanged 11.7% K error suggests about 14.9% KV
error. The study records both this unit-Gaussian proxy and the proxy using its
actual measured weight concentration.

These are **heuristics, not lower bounds**. They assume centered weak noise,
independence, diffuse weights and concentration of aggregate norm ratios.
Shared exponents induce correlations; saturation can introduce gain bias;
K error can correlate with the logits and weights; and peaked attention or
large perturbations invalidate the simple linearization. A finite sampled
reference can also have unusual cancellation. The code therefore records
reconstruction gain/error-input correlation, weight concentration, measured
K-versus-V output-error cosine and the exact nonlinear cross term rather than
treating the root-sum-square estimate as an acceptance criterion.

The uniform-Q case is an exact mathematical control: K has no effect, all
uncorrected V error is `mean(Vq - V)`, and subtracting that mean recovers the
original mean V up to FP64 summation error. Final BF16 output rounding remains.
This exact identity does not imply a generally zero attention-quantization
floor for nonuniform weights.

## First-order check and provenance

For the actual measured logit perturbation `ds`, the model forms
`dp_linear = P * (ds - sum(P * ds))`. Predictions include `dp_linear @ V` for
K-only and that error plus `P @ (Vq - V)` for KV. An additional explicitly
labeled model retains `dp_linear @ (Vq - V)`, which is second order in the joint
K/V perturbation, although linear in the perturbed P. It is not a purely
first-order KV approximation. Predicted output L2, error-vector agreement and
cosine are compared to exact FP64 quantized attention.

JSONL records pin all transitive CPU-model sources, original BF16 input hashes,
query indices, reference hash and RMS, both output-precision metric sets,
reconstruction statistics, and source/input immutability gates. An independent
online FP64 reference is checked against the dense, query-batched reference.
Startup tests check dense agreement, the correction identity, uniform-Q
invariance and a small-perturbation finite-difference softmax derivative.

## Running

Queue only in a CPU-safe gap, with no concurrent device timing. From the repo
root, using an existing CPU Torch environment:

```sh
/opt/venv/bin/python -B experiments/sdpa-l2/bfp4-lofi-v2/quantization_noise_budget.py --label quant-noise-budget-v1
```

The driver fixes Torch intra-op threads to four and inter-op threads to one.
Defaults produce 20 codec/query-mode records, each with six output comparisons.
Query batches of eight bound score scratch at long context; full FP64 K/V and
their reconstructed versions still require several GB of transient CPU memory.
For TT-only execution add `--codecs tt_rne`. For just tiny startup checks use
`--label self-test --self-test-only`. A fresh label is required for result files;
JSONL is written beside the driver and echoed to stdout. CPU elapsed seconds
are bookkeeping only and must not be presented as attention TFLOP/s.
