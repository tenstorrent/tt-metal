# Reversible value-channel smoothing with BFP4

Signed Hadamard rotation is useful for value outliers, but is not a universal
BFP4 accuracy improvement. H16 and H128 substantially reduce the isolated
value error of persistent channel outliers. The improvement survives with
BFP4 K, although QK error then limits the total. Normal inputs do not improve.
Uncentered common-mode V is a serious regression because rotation creates
coherent value-quantization mean error. The tested max-based channel scaling
does not improve aggregate L2.

## Experiment contract

- 120 CPU configurations, four threads, 21.7 seconds; no TT device.
- H1, Q128, D128, N4096/32768; seeds 1240/1241; original BF16 Q/K/V reference
  generated with `REPRO.make_inputs`.
- Q: per-value RNE7. K: host/native-group RNE BFP4 in all 96 main cases.
  Exact-K ablation: N32768/seed1240, four distributions, six transforms (24 cases).
- P: exact FP64 online exponentials, per-value RNE7, matched denominator,
  K chunks of 512. All QK/PV matmuls, recurrent corrections, and inverse transforms
  use FP64. This excludes hardware accumulator, subtraction, and exp errors.
- V: native shared-exponent groups of 16 along D, host RNE BFP4 conversion.
  Encoded values are checked to fit the LoFi right-operand width exactly.
  This is the unbiased target quantizer, not the older biased native packer.
- Fixed Hadamard signs use seed 20260915, independent of data seed. Forward
  butterflies run in FP32, with either FP32 directly feeding quantization or an
  explicit BF16 spill before quantization. Scaling by powers of two preserves
  these ordinary BF16 inputs exactly, so it has no separate spill variant.
- Final output is BF16 **after** inversion. JSON also records a BF16
  attention-output spill **before** inversion.
- `outliers` is the existing sparse 10x Gaussian perturbation on Q, K, and V;
  exact-K ablation separates its QK contribution from the V benefit.
  `common_v` adds 32 to V before BF16 input rounding.
  `channel_outlier_v` starts from normal inputs and multiplies every 16th V
  channel by 32, producing eight persistent high-amplitude value channels.

Tables are generated from [raw JSON](value-smoothing-v1.jsonl), which records
full precision, source hashes, and configuration. See the
[isolated driver](value_smoothing_models.py). Two seeds and one outlier layout
are evidence of a direction, not broad qualification.

## Main cases: BFP4 K and V

Each cell is mean L2% over both lengths and both seeds, followed by min–max.

| Distribution | No transform | Power-of-two scaling | H16, BF16 spill | H128, BF16 spill |
|---|---:|---:|---:|---:|
| Normal | 16.418 (15.971–16.757) | 18.558 (18.254–18.777) | 16.626 (16.510–16.757) | 16.597 (16.273–16.849) |
| Sparse Q/K/V outliers | 28.132 (19.001–48.078) | 34.103 (24.584–49.260) | 27.537 (16.713–47.842) | 27.327 (16.661–47.457) |
| V common mode +32 | 0.055 (0.029–0.082) | 0.055 (0.029–0.082) | 1.200 (1.190–1.208) | 8.483 (8.473–8.491) |
| Persistent V channel outliers | 17.855 (16.764–18.556) | 19.806 (17.642–21.128) | 14.946 (13.909–15.498) | 14.941 (14.030–15.587) |

Persistent channel-outlier improvement occurs on all four input cases, with
either rotation size and with BF16 preprocessing spills. Sparse-outlier results
remain highly variable because K quantization alone can dominate (the
exact-V/BFP4-K baseline reaches 46.682% on one case). V smoothing cannot fix QK.

The common-V baseline's tiny relative L2 is dominated by the large 32 offset;
it does **not** prove accurate recovery of its small varying component.

## Exact-K attribution: N32768, seed 1240

This retains Q7 and P7 but removes K quantization. Values are output L2%.

| V transform | Normal | Sparse outliers | Common V | Channel outliers |
|---|---:|---:|---:|---:|
| None | 11.041 | 16.438 | 0.030 | 12.676 |
| Power-of-two max scaling | 14.126 | 17.476 | 0.030 | 17.162 |
| Signed H16, FP32 prep | 11.490 | 9.797 | 1.562 | 7.080 |
| Signed H16, BF16 spill | 11.569 | 9.783 | 1.190 | 7.118 |
| Signed H128, FP32 prep | 11.498 | 11.104 | 8.449 | 7.111 |
| Signed H128, BF16 spill | 11.541 | 11.122 | 8.483 | 7.195 |

H16 reduces isolated sparse-outlier error from 16.438% to 9.797% and persistent
channel-outlier error from 12.676% to 7.080% with FP32 preprocessing. H128 offers
no consistent advantage over H16 here. Normal V representation error stays
near 11.7%, as expected: orthogonal rotation cannot systematically improve an
already approximately isotropic Gaussian distribution.

### Representation and coherent mean error

Selected exact-K/32K diagnostics:

| Input | Transform | V representation L2% | V mean-error RMS | Output L2% | After mean-error correction L2% |
|---|---|---:|---:|---:|---:|
| Persistent V channel outliers | None | 11.330 | 0.005 | 12.676 | 10.068 |
| Persistent V channel outliers | Signed H16, BF16 spill | 6.341 | 0.003 | 7.118 | 5.704 |
| Persistent V channel outliers | Signed H128, BF16 spill | 6.337 | 0.003 | 7.195 | 5.696 |
| V common mode +32 | None | 3.130 | 0.006 | 0.030 | 0.030 |
| V common mode +32 | Signed H16, BF16 spill | 3.980 | 0.398 | 1.190 | 0.030 |
| V common mode +32 | Signed H128, BF16 spill | 11.108 | 2.712 | 8.483 | 0.033 |

Mean-error correction subtracts the mean over token rows of
`inverse(quantized(transformed(V))) - V` from the reconstructed output.
It is diagnostic only, not implemented on device. With matched probability
normalization, a constant value error contributes the same constant output
error. Thus this correction is mathematically well-defined even when the
attention weights are approximate.

The common-V regression is almost entirely such a coherent error. It does
not average away as the context grows. Nearest-even rounding alone is not
sufficient to guarantee zero mean error on shifted, nonuniform distributions,
especially with shared-exponent saturation.

## Reversibility and rounding

Let H be normalized Walsh–Hadamard and D the fixed diagonal sign matrix.
The forward transform is T=DH, so V'=VT and O'=PV'. In exact arithmetic,
O=O'T^T=PV. H16 applies this independently to each contiguous 16-channel group;
H128 mixes all 128 channels.

For scaling, S is diagonal with
`s_c = 2^ceil(log2(max_token(abs(V_token,c))))`.
Then V'=VS^-1 and O=O'S. The scales are exact powers of two for ordinary-range
inputs. They avoid arbitrary-scale multiplier truncation, but do not make
subsequent BFP4 quantization reversible.

Self-tests verified the signed transform/inverse in FP64 and checked
FP32/BF16 preprocessing round trips. Across this suite the BF16 preprocessing
spill alone causes at most 0.175% V round-trip L2. It generally does not erase the
outlier benefit, although common-mode quantization thresholds remain sensitive.
A BF16 output spill before inverse changes total L2 by at most
0.024 percentage points in these 120 cases. That is measured here, not a
guarantee for other distributions.

## Engineering implications

1. H16 is the most attractive rotation candidate for targeted outlier V:
   four local butterfly stages instead of H128's seven, naturally aligned to
   the native shared-16 quantization groups. Both the V preparation and output
   inverse cost O(ND log16), independent of the quadratic attention matmuls.
   The CPU model makes no device-performance claim.
2. Do not enable signed V rotation indiscriminately. Centering V first and
   restoring its mean, or matching the reconstructed quantized mean, are
   natural next controls for the demonstrated common-mode failure. These
   combined schemes were **not** tested in this study.
3. Do not pursue this particular max-based scaling as an aggregate-L2 win.
   Per-channel extrema can push otherwise similar Gaussian channels across
   different power-of-two bins. On channel-outlier inputs, inverse scaling
   amplifies errors in dominant channels; balancing small-channel error need
   not minimize globally norm-weighted output L2. Small-channel-specific
   metrics could reveal a different tradeoff, but were not collected.
4. Scaling requires per-channel max reduction over all V tokens (per head),
   then application of scales and an output correction. Statistics generation,
   synchronization, storage, and O(ND) conversion costs are excluded from this
   CPU numerical model and must be counted in an implementation.
5. H16/128 are not routes to sub-percent accuracy with one-component BFP4 K.
   Their useful contribution is reducing structured V error at this coarser
   precision point; QK error and native BFP4's representation floor remain.

No production or frozen source was modified, and no device jobs were launched.

## Completed device centering follow-up: V8 and V4

The statement above describes the original CPU rotation/scaling study. The
parent subsequently ran the separate value-centering implementations on
Blackhole. These runs do **not** combine centering with Hadamard rotation.
All 36 V4/V8 records are now present and covered by the checkpoint validator:
H10, D128, Q256/K512, 110 cores, two K/V input slots, seed 1240, noncausal
square attention at N32768 and N262144, 128 explicit query rows per head against
original-input FP64 attention. Every full output is checked finite and replayed.
The attention mode is LoFi BF16 with denominator compensation only, not full
numerator compensation and not the accurate FP32 algorithm. K is B8 RNE5.

V8 uses RNE5 followed by native BFP8 packing. Its `matched_mean` correction
decodes the actual packed V8, applies the truncation to the right-operand bits
that LoFi consumes, and computes the device mean of **those effective values**.
The restored bias is the original device BF16 mean minus that effective mean.
Matching the decoded B8 mean without the consumption truncation is a different
algorithm. `original_mean` restores only the original device mean. Both produce
a BF16 attention output followed by a BF16 bias-add epilogue. Means, preparation
and the epilogue run on device and are included in combined timing.

### V8 results: total L2 is not centered residual L2

Each cell below is **total L2% / centered residual L2%**. Centering for the
metric subtracts the same FP64 mean of original V from both actual and reference;
there is no gain fitting. Constant-V residual L2 is undefined, shown as “—”.

| Context | Input | No centering | Original mean | Matched effective mean |
|---|---|---:|---:|---:|
| 32K | Normal | 3.3184 / 4.1552 | 11.4841 / 14.3799 | 3.1602 / 3.9571 |
| 256K | Normal | 4.5985 / 5.7845 | 40.6934 / 51.1885 | 4.4036 / 5.5393 |
| 32K | Common V +32 | 1.5073 / 6587.08 | 0.028655 / 125.228 | 0.028655 / 125.228 |
| 256K | Common V +32 | 25.6745 / 318472.74 | 0.010421 / 129.260 | 0.010421 / 129.260 |
| 32K | Constant V =1 | 1.4277 / — | ~0 / — | ~0 / — |
| 256K | Constant V =1 | 25.4669 / — | ~0 / — | ~0 / — |

Raw V8 evidence uses `valuecenter-b8-{32768,262144}-{none,original_mean,matched_mean}-{normal,common_v,constant_v}-v1.json`.
Representative endpoint records:
[32K normal baseline](valuecenter-b8-32768-none-normal-v1.json),
[32K normal matched](valuecenter-b8-32768-matched_mean-normal-v1.json),
[256K normal baseline](valuecenter-b8-262144-none-normal-v1.json),
[256K normal original mean](valuecenter-b8-262144-original_mean-normal-v1.json),
[256K normal matched](valuecenter-b8-262144-matched_mean-normal-v1.json),
[256K common-V matched](valuecenter-b8-262144-matched_mean-common_v-v1.json),
[256K constant-V baseline](valuecenter-b8-262144-none-constant_v-v1.json),
[256K constant-V original mean](valuecenter-b8-262144-original_mean-constant_v-v1.json),
[256K constant-V matched](valuecenter-b8-262144-matched_mean-constant_v-v1.json).

### What the constant and common-mode controls establish

For centered constant V, complete-output hashes match an all-ones BF16 tensor
exactly at both lengths, for both mean policies. The remaining reported L2 is
only FP64 reference summation noise (6.36e-14% and 3.65e-14%); the device constant
is exact. V=1 is itself exactly representable in B8 and B4. The uncentered
1.43%/25.47% output error therefore cannot be explained by a four-bit value
representation floor: it demonstrates error in the uncentered BF16 attention
path. Subtracting the constant makes the centered numerator zero and restores
the constant outside that recurrent path. This control does not establish
accurate varying-value attention.

For centered common V, complete-output hashes match an all-32 BF16 tensor for
both mean policies at both lengths. On the sampled reference rows, maximum
absolute error is 0.040146 at 32K and 0.015369 at 256K, below 0.0625, half the
smaller BF16 spacing adjacent to 32. Thus rounding those reference values to
BF16 also produces 32: the centered implementations reach the **BF16 output
rounding floor on the sampled rows**. This is not a BFP8/BFP4 representation
floor and does not mean the small signal survives. Centered residual L2 remains
125.23%/129.26%, absolute RMS error is 0.0091693/0.0033345, and output PCC is
undefined because the output is constant. A numerically accurate residual
evaluation requires an output representation that can retain that residual,
for example higher-precision output or separately carried mean and residual.

### Engineering conclusion

Do not enable original-mean-only V8 centering for general inputs: it severely
regresses normal data, especially at long context. Matching the mean of the
values actually consumed by LoFi removes this large coherent-error component
and yields a modest normal-input improvement, from 3.3184% to 3.1602% at 32K and
4.5985% to 4.4036% at 256K. It does not make the weighted quantization error,
BF16 recurrence, or output rounding exact. These are one-seed results, not a
universal guarantee.

The matched correction has a larger normal-input benefit with V4: total L2
falls from 12.0456% to 9.8149% at 32K and 12.5655% to 10.3229% at 256K. That is
consistent with more value-quantization error to correct, not proof of an
irreducible ~10% attention floor. The original CPU study's ~11.7% normal V4
reconstruction error describes its input quantizer, not a universal lower
bound on output error after attention averages values.

For normal V8, combined throughput is 206.74→189.42 TFLOP/s at 32K and
206.65→205.62 TFLOP/s at 256K when enabling matched centering. The additional
device work is therefore meaningful at 32K and largely amortized at 256K in
these measurements. These are combined, real-data-movement timings, not
resident FPU utilization. Centering is a targeted robustness/mean-error tool,
not a free or universal accuracy improvement, and it does not repair lost
common-mode residuals in a final BF16 output.
