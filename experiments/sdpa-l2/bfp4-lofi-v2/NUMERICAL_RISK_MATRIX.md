# Low-precision SDPA: numerical-risk matrix

Status: exploratory operator evidence, September 15, 2026. The four locked
SDPA implementations are unchanged. **K8/V4 is a safer asymmetric candidate
than K4/V8 under the tested sharp-logit stresses.** Compensation protects
long-context recurrent state; it cannot recover information removed by K4/V4.
No model-quality claim is supported yet.

## What the asymmetric model adds

[New driver](asymmetric_risk_models.py), [raw results](asymmetric-risk-models-v1.jsonl):
80 output records in 27.7 seconds, four CPU threads. H1/Q128/D128, seed 1240;
normal at 4K and nine distributions at 32K. Reference uses original BF16 inputs
and FP64 attention; final output is BF16. Tables below are generated from JSON,
with L2 displayed to three decimal places.

Q uses per-value RNE7. K4/V4 are native shared-16 RNE BFP4. K8/V8 mean
**per-value RNE5, native BFP8 RNA packing, then effective LoFi trunc5**, not
full-seven-bit BFP8 matmul. P uses effective trunc7 and a matched denominator.
The native eight-bit exp grid is modeled in IEEE FP32; QK, subtraction,
online rescaling and PV/state use FP64. Exact-exp controls use the same P rule.
This excludes device cheap-subtraction error, BF16 P spilling, recurrent BF16
rounding/compensation and SFPU-specific arithmetic. It is not a device prediction
to the last decimal, a BF16 simulation, or qualification.

32K native-exp output L2%, ordered to expose the asymmetric comparison:

| Input | K8/V8 | K4/V8 | K8/V4 | K4/V4 |
|---|---:|---:|---:|---:|
| Normal | 2.733 | 11.912 | 11.262 | 16.006 |
| Sparse outliers in Q/K/V | 4.046 | 46.427 | 16.834 | 48.138 |
| Q and K each scaled by 2 | 4.538 | 29.361 | 12.655 | 31.575 |
| Common Q +32 | 6.602 | 21.115 | 12.980 | 24.455 |
| Common K +32 | 44.235 | 77.690 | 45.408 | 78.018 |
| Same common K, centered before encoding | 2.334 | 11.578 | 11.149 | 15.813 |
| Common V +32 | 0.387 | 0.387 | 0.030 | 0.030 |
| Every 16th V channel scaled by 32 | 3.463 | 13.420 | 13.036 | 18.023 |
| Constant V = 1 | 0.000 | 0.000 | 0.000 | 0.000 |

These failures primarily arise before exp: with exact exp, outlier K4/V8 is
46.313% versus K8/V4 16.918%; scaled-QK is 29.606% versus 12.718%. Native exp
does not explain the asymmetry. On normal K8/V8, however, exact-to-native exp
raises L2 from 2.086% to 2.733%; at K4/V4 it changes 15.977% to 16.006%.
Native exp is therefore a more defensible tradeoff at the coarse point than
at the approximately 2% point. Small improvements on some stresses can be
error cancellation, not a better exponential.

## Risk matrix

| Stress / mechanism | K4 risk | V4 risk | What helps; what does not |
|---|---|---|---|
| IID normal | Coarse score perturbations; comparable to V4 in aggregate | Weighted-value quantization | Unbiased packing is essential; normal-only tests cannot choose the asymmetric format safely |
| Sparse outliers, large Q/K, close competing logits | Shared exponents erase smaller K components; errors in QK can change attention routing | Linear weighted-value error, without directly changing routing | Prefer more K precision first; inspect score error, attention entropy and top-two margins. Sensitivity is not monotonic in entropy: fully separated winners can be robust |
| Common K across tokens | Quantization destroys small token differences although the common component is mathematically irrelevant | No special V mechanism | Center K before quantization; compensation/FP32 state cannot undo lost score differences |
| Common Q | Its dot product with K is a real, key-dependent score signal; inaccurate Q/K perturb it | Usually secondary here | Q centering requires an accurate score correction; dropping the mean silently changes the operator |
| V channel outliers / correlated V errors | Existing score errors may be amplified by large V channels | Shared-16 packing sacrifices neighboring smaller channels; coherent V mean error survives averaging | Conditional V rotation/mean matching may help; denominator compensation does not remove V representation bias |
| Long contexts, diffuse or repeated blocks | Representation loss remains even with exact state | Small numerator updates can disappear in BF16 recurrence | FP32 recurrence or numerator **and** denominator compensation; denominator-only is not sufficient |
| Large common V / constant V | Weak test of score accuracy | Global L2 may be dominated by the common offset | Check dynamic residuals and BF16-output rounding floor; constant V is a structural normalization control, not model-quality evidence |

For common V, the BFP4 model returns BF16 output 32 throughout. Its 0.030% L2
is the small FP64 residual relative to the large offset, not proof that V4
preserves fine information. BF16 output rounding itself can erase that residual.
PCC is undefined for a constant output; do not replace it with a passing score.

## Device evidence: compensation is a different axis

Existing normal-input 32K/H10, Q256/K512/D128, 110-core chain harness:

| K/V | MAIN LoFi BF16 L2% | Full-compensated LoFi BF16 L2% |
|---|---:|---:|
| B8/B8 | 3.287 | 3.151 |
| B4/B8 | 12.397 | 12.367 |
| B8/B4 | 12.041 | 12.007 |
| B4/B4 | 16.961 | 16.938 |

Sources: `asym32k-{main_bf16,fast_bf16}-{b8_b8,b4_b8,b8_b4,b4_b4}-v1.json`.
Every output was checked finite, but numerical reference checks cover 128 sampled
Q rows per head and all K/V, not every Q row. These are experimental common-reader
results, not production dispatch. MAIN here means the private **LoFi** variant,
not unmodified-main HiFi2.

At 256K, B4/B4 becomes 26.264% with ordinary BF16 recurrence, 17.168% with full
compensation, and 16.853% with FP32 recurrence/cubic exp. That supports compensation
as protection against additional drift, not a cure for the roughly 17% format
floor. [MAIN](chain-h10-262144-lofi_main_b4-v1.json),
[compensated](chain-h10-262144-lofi_fast_b4-v1.json),
[FP32](chain-h10-262144-lofi_fp32_b4-v1.json).
The FP32 comparison also changes exp/subtraction, so it is not a pure state ablation.

Denominator-only compensation gave little benefit at 32K and a repeated-resident
control exposed numerator drift; retain that as a stress control. See
[the intermediate notebook](PROGRESS.md), not a claim that denominator-only is
universally equivalent to full compensation.

## Centering and smoothing: exact identities versus extra approximations

Let W be row-normalized attention weights. In exact arithmetic:

- **K:** replacing every key with K−c leaves softmax unchanged, because Qc is
  constant within each score row. Any token-constant c works; it need not be an
  exact mean. Different c per K chunk instead requires restoring chunk offsets.
- **Q:** Q=Q′+a requires scores Q′Kᵀ+aKᵀ. The second term varies across keys.
  A per-Q-block mean can share that correction across rows; the correction must
  retain the precision removed from Q. Existing 32K Q7/B8 CPU common-Q evidence
  is 6.412% without preprocessing, 5.193% with K centering, 1.490% with corrected
  Q/K centering. [Raw records](centered-models-v1.jsonl).
- **V:** V=V′+c gives O=WV′+c. After quantization V̂′=V′+E, merely restoring c
  leaves error WE. Subtracting the column mean of E removes its coherent component,
  not all attention-weighted error. This is an additional error correction,
  not a guarantee of exact quantization reversibility. An earlier 256K normal
  B8 model went 2.204% →39.845% after Q/K/V centering, then 1.955% with represented-V
  mean matching. That older quantizer/preprocessing configuration is a warning
  about mean bias, not a prediction for every RNE implementation.
  [Raw records](vmean-models-v1.jsonl).
- **Value rotation:** V′=VH, O=O′Hᵀ is exact for orthogonal H; a BF16 spill and
  quantization are not. Existing exact-K 32K V4 ablations show H16/BF16 improving
  sparse outliers 16.438%→9.783%, but worsening common V 0.030%→1.190%. No universal
  smoothing switch is justified. [Study and raw-source links](VALUE_SMOOTHING.md).

All identities should be checked against the **same already-BF16 inputs**.
Generating a new BF16 tensor after adding an offset changes the inputs through
rounding and is not a strict invariance test of the original tensor.

## Useful qualification controls and activation artifacts

Keep a small deterministic operator stress lane separate from model-quality
acceptance. Include normal seeds/lengths; isolated Q, K and V outliers; channel
outliers; scaled Q/K and near-tied logits; common K with/without centering;
corrected common Q; common/constant V; uniform attention; and repeated blocks
plus distinct long-context inputs. Include all-output small tests, trace replay,
quantizer ties/exponent boundaries, and chunk/order sensitivity at fixed Q/K
sizes. Unsupported exceptional/subnormal inputs remain an explicit contract,
not silently passing tests.

Report global and row/head-tail L2, PCC where defined, gain, absolute error,
reference RMS, and the BF16-output rounding-only floor. For coarse formats add
centered score error, attention disagreement, consumed-P mass, and represented-V
mean error. Raw max relative error near zero is useful diagnostically but should
not dominate acceptance by itself.

Before claiming model fidelity, obtain **authorized, provenance-tagged Q/K/V
captures** from representative text and/or diffusion workloads: model/checkpoint,
layer/head and GQA mapping, sequence positions, causal/padding masks, scale/bias,
RoPE and normalization placement, original dtype, prompt/seed or diffusion step,
and the exact reference implementation. Capture complete K/V sequences with
sampled Q rows initially, plus contiguous Q blocks for correction/streaming tests.
Include short/long contexts and early/middle/late layers, not only a convenient
head. Preserve cross-channel and cross-token correlations; scalar histograms or
fresh Gaussian surrogates are insufficient. Log per-group dynamic range, channel
means/outliers, score margins and attention concentration to identify which
synthetic stresses actually occur.

Replay all selected candidates on those same tensors before end-to-end tests.
Then measure the task's actual quality metric with paired inputs/seeds and an
appropriate tolerance/confidence estimate. Keep K8/V4 and K4/V8 separate until
that evidence exists; consider per-layer/head precision selection rather than a
single global coarse mode.

## Operator L2 is not Sage model-level fidelity

This report's metric is 100×‖O−Oref‖₂/‖Oref‖₂. It cannot be converted into
perplexity, benchmark accuracy, image/video quality, or a claim of “lossless”
attention. Our previous Sage-style INT4 simulation was not an executed Sage
kernel and quantized Q as well; it is not this Q7/asymmetric experiment.
[Prior comparison](SAGE_INT4_COMPARISON.md).

SageAttention2 v7 describes per-block Q centering/correction and E4M3 P/V, not
BFP4 V, and reports both operator and model evaluations. Its INT4 and INT8
results differ; a favorable INT8 model result is not evidence for BFP4/BFP4.
[Primary paper, §§3 and Tables 2/6](https://arxiv.org/html/2411.10958v7),
[pinned official dispatch](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/core.py).
The defensible present claim is **measured Sage-inspired low-precision operator
tradeoffs**, with model suitability still to be established.
