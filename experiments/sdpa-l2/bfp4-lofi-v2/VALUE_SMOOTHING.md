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
