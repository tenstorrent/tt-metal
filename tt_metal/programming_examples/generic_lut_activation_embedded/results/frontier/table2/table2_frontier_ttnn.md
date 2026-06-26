# Table 2 - Frontier Pareto TTNN Comparison

Source: `results/frontier/<dtype>/data/csv/pareto_winners.csv`.
Rows marked `excluded` or `incomplete` are visible but are not counted as wins.

## bf16

Comparable rows: 52. Win on ULP and runtime: 40. Accuracy match but slower: 11. Faster but less accurate: 0. Incomplete TTNN refs: 7. Excluded: 1.

| activation | ours cfg | ours ULP | ours us | TTNN ULP | TTNN us | speedup | result |
|---|---:|---:|---:|---:|---:|---:|---|
| abs | poly:s2/d2 | 0.00 | 1.78 | 0.00 | 1.71 | 0.961 | accuracy_match_slow |
| acos | rational:s2/d10d3 | 0.06 | 6.47 | 0.00 | 5.74 | 0.887 | loss |
| acosh | rational:s1/d4d3 | 1.00 | 2.93 | 1.00 | 4.73 | 1.614 | win_both |
| asin | rational:s1/d8d8 | 0.00 | 3.03 | 0.00 | 5.70 | 1.881 | win_both |
| asinh | rational:s1/d6d5 | 1.00 | 2.53 | 1.00 | 4.12 | 1.628 | win_both |
| atan | poly:s1/d7 | 0.50 | 1.98 | 0.50 | 3.91 | 1.975 | win_both |
| atanh | rational:s1/d5d5 | 1.00 | 2.54 | 1.00 | 4.26 | 1.677 | win_both |
| cbrt | exponent_alu_pow:s16/d1 | 1.00 | 4.41 | 1.00 | 2.14 | 0.485 | accuracy_match_slow |
| celu | poly:s2/d13 | 0.00 | 2.74 | 0.00 | 3.61 | 1.318 | win_both |
| cos | trig:s1/d6d4 | 0.00 | 4.05 | 0.00 | 2.56 | 0.632 | accuracy_match_slow |
| cosh | poly:s1/d16 | 0.00 | 3.01 | 0.00 | 2.97 | 0.987 | accuracy_match_slow |
| digamma | rational:s1/d4d3 | 1.00 | 2.93 | 1.00 | 6.64 | 2.266 | win_both |
| elu | poly:s2/d11 | 0.00 | 2.74 | 0.00 | 3.60 | 1.314 | win_both |
| erf | rational:s1/d8d6 | 0.50 | 2.87 | 0.50 | 3.63 | 1.265 | win_both |
| erfc | poly:s4/d6 | 0.50 | 4.19 | 0.50 | 5.45 | 1.301 | win_both |
| erfinv | rational:s1/d4d4 | 1.00 | 2.38 | 1.00 | 5.96 | 2.504 | win_both |
| exp | exponent_alu_exp2:s1/d2 | 1.00 | 1.94 | 1.00 | 1.90 | 0.979 | accuracy_match_slow |
| exp2 | exponent_alu_exp2:s16/d2 | 0.50 | 1.94 | 0.50 | 2.52 | 1.299 | win_both |
| expm1 | poly:s1/d7 | 0.00 | 1.99 | 0.00 | 2.92 | 1.467 | win_both |
| gelu | poly:s16/d2 | 0.12 | 11.80 | 0.12 | 5.78 | 0.490 | accuracy_match_slow |
| hardmish | gated_affine_product:s3/d3 | 0.00 | 1.72 | 0.03 | 1.77 | 1.029 | win_both |
| hardshrink | threshold_identity:s3/d2 | 0.00 | 1.74 | 0.00 | 1.82 | 1.046 | win_both |
| hardsigmoid | poly:s3/d2 | 0.50 | 1.74 | 0.50 | 1.79 | 1.029 | win_both |
| hardswish | gated_affine_product:s3/d6 | 0.02 | 1.78 | 0.25 | 2.01 | 1.129 | win_both |
| hardtanh | poly:s3/d10 | 0.00 | 1.68 | 0.00 | 1.86 | 1.107 | win_both |
| i0 | poly:s1/d10 | 0.25 | 2.35 | 1.00 | 3.10 | 1.319 | win_both |
| i1 | rational:s1/d8d5 | 0.50 | 2.70 | 0.50 | 7.40 | 2.741 | win_both |
| identity | poly:s1/d13 | 0.00 | 0.13 | 0.00 | 0.16 | 1.231 | win_both |
| leaky_relu | poly:s2/d1 | 0.00 | 1.82 | -- | -- | -- | incomplete |
| lgamma | rational:s1/d6d4 | 1.00 | 3.42 | 1.00 | 13.48 | 3.942 | win_both |
| log | exponent_alu_log2:s16/d2 | 1.00 | 2.17 | 1.00 | 2.32 | 1.069 | win_both |
| log10 | exponent_alu_log2:s16/d3 | 0.50 | 2.26 | 0.50 | 2.48 | 1.097 | win_both |
| log1p | exponent_alu_log2:s16/d2 | 1.00 | 2.26 | 1.00 | 2.43 | 1.075 | win_both |
| log2 | exponent_alu_log2:s2/d3 | 1.00 | 2.26 | 1.00 | 2.47 | 1.093 | win_both |
| logit | rational:s1/d4d4 | 1.00 | 3.04 | 1.00 | 5.40 | 1.776 | win_both |
| logsigmoid | rational:s2/d5d5 | 0.00 | 5.46 | -- | -- | -- | incomplete |
| mish | poly:s2/d9 | 0.25 | 3.27 | 0.25 | 4.31 | 1.318 | win_both |
| multigammaln | poly:s1/d2 | 118.12 | 1.71 | 127.00 | 0.13 | 0.076 | excluded |
| polygamma | poly:s32/d10 | 16.00 | 25.98 | -- | -- | -- | incomplete |
| prelu | poly:s2/d13 | 0.00 | 1.78 | -- | -- | -- | incomplete |
| relu | poly:s2/d1 | 0.00 | 1.68 | 0.00 | 1.72 | 1.024 | win_both |
| relu6 | poly:s3/d6 | 0.00 | 1.71 | 0.00 | 1.84 | 1.076 | win_both |
| relu_max | poly:s3/d1 | 0.00 | 1.89 | -- | -- | -- | incomplete |
| relu_min | poly:s2/d1 | 0.00 | 1.78 | -- | -- | -- | incomplete |
| rsqrt | newton_root:s16/d2 | 0.00 | 2.01 | 0.00 | 2.47 | 1.229 | win_both |
| selu | poly:s2/d13 | 0.00 | 2.79 | 0.00 | 3.63 | 1.301 | win_both |
| sigmoid | exponent_alu_exp2:s8/d2 | 0.50 | 2.46 | 0.50 | 2.89 | 1.175 | win_both |
| sigmoid_accurate | exponent_alu_exp2:s1/d2 | 0.50 | 2.47 | 0.50 | 2.93 | 1.186 | win_both |
| silu | poly:s2/d12 | 0.25 | 3.85 | 0.25 | 3.11 | 0.808 | accuracy_match_slow |
| sin | trig:s1/d11 | 0.00 | 2.87 | 0.00 | 2.35 | 0.819 | accuracy_match_slow |
| sinh | rational:s1/d14d2 | 0.00 | 3.03 | 0.00 | 3.25 | 1.073 | win_both |
| softplus | poly:s2/d9 | 0.12 | 3.29 | 0.12 | 5.31 | 1.614 | win_both |
| softshrink | poly:s3/d2 | 0.00 | 2.00 | 0.00 | 1.91 | 0.955 | accuracy_match_slow |
| softsign | abs_denominator_rational:s2/d1d1 | 0.00 | 1.80 | 1.00 | 1.96 | 1.089 | win_both |
| sqrt | newton_root:s1/d14 | 0.00 | 2.16 | 0.00 | 2.28 | 1.056 | win_both |
| swish | poly:s2/d12 | 0.25 | 3.85 | 0.25 | 3.10 | 0.805 | accuracy_match_slow |
| tan | tan:s1/d7 | 0.12 | 3.31 | 0.12 | 2.94 | 0.888 | accuracy_match_slow |
| tanh | poly:s1/d6 | 0.50 | 2.10 | 0.50 | 2.29 | 1.090 | win_both |
| tanhshrink | rational:s1/d12d3 | 0.12 | 2.87 | 0.12 | 3.17 | 1.105 | win_both |
| threshold | poly:s2/d2 | 0.00 | 1.80 | -- | -- | -- | incomplete |

Excluded rows:
- multigammaln: frontier/native rows were generated before multigammaln.json was fixed to the true p=4 target; rerun p=4 sweep before claiming

## fp32

Comparable rows: 52. Win on ULP and runtime: 21. Accuracy match but slower: 4. Faster but less accurate: 15. Incomplete TTNN refs: 7. Excluded: 1.

| activation | ours cfg | ours ULP | ours us | TTNN ULP | TTNN us | speedup | result |
|---|---:|---:|---:|---:|---:|---:|---|
| abs | poly:s2/d13 | 0.00 | 2.89 | 0.00 | 2.94 | 1.017 | win_both |
| acos | rational:s1/d12d4 | 352.00 | 4.91 | 14.00 | 8.12 | 1.654 | faster_less_accurate |
| acosh | rational:s1/d6d4 | 5626.00 | 3.96 | 6791.00 | 5.29 | 1.336 | win_both |
| asin | rational:s1/d12d4 | 303.00 | 3.65 | 27.50 | 7.99 | 2.189 | faster_less_accurate |
| asinh | rational:s1/d6d6 | 9473.00 | 3.43 | 10107.00 | 4.69 | 1.367 | win_both |
| atan | poly:s4/d9 | 27.00 | 5.46 | 2.00 | 5.11 | 0.936 | loss |
| atanh | rational:s1/d11d4 | 19420.00 | 3.65 | 10029.00 | 4.79 | 1.312 | faster_less_accurate |
| cbrt | exponent_alu_pow:s4/d6 | 3.00 | 5.84 | 2.00 | 3.41 | 0.584 | loss |
| celu | poly:s2/d10 | 103.94 | 3.40 | 0.06 | 4.40 | 1.294 | faster_less_accurate |
| cos | trig:s1/d11d2 | 1.08 | 5.19 | 1.00 | 3.48 | 0.671 | loss |
| cosh | rational:s2/d14d1 | 4.00 | 7.48 | 1.00 | 4.10 | 0.548 | loss |
| digamma | rational:s2/d4d4 | 22.94 | 5.40 | 79.00 | 12.17 | 2.254 | win_both |
| elu | poly:s2/d10 | 103.94 | 3.39 | 0.06 | 4.38 | 1.292 | faster_less_accurate |
| erf | rational:s1/d15d15 | 3.50 | 4.68 | 3.50 | 5.49 | 1.173 | win_both |
| erfc | poly:s4/d7 | 9957.00 | 5.06 | 15979.50 | 5.99 | 1.184 | win_both |
| erfinv | rational:s1/d6d5 | 17413.00 | 3.29 | 20560.00 | 6.52 | 1.982 | win_both |
| exp | exp:s1/d5 | 2.00 | 4.41 | 1.00 | 4.02 | 0.912 | loss |
| exp2 | rational:s3/d5d4 | 13.00 | 7.64 | 0.50 | 3.59 | 0.470 | loss |
| expm1 | poly:s1/d7 | 6.00 | 3.12 | 1.00 | 4.50 | 1.442 | faster_less_accurate |
| gelu | poly:s32/d2 | 1204.88 | 20.51 | 2.00 | 6.61 | 0.322 | loss |
| hardmish | poly:s3/d4 | 0.00 | 3.06 | 0.03 | 3.05 | 0.997 | accuracy_match_slow |
| hardshrink | poly:s3/d2 | 0.00 | 3.02 | 0.00 | 2.98 | 0.987 | accuracy_match_slow |
| hardsigmoid | poly:s3/d1 | 0.50 | 3.05 | 0.50 | 3.05 | 1.000 | win_both |
| hardswish | poly:s3/d4 | 0.25 | 3.05 | 0.25 | 3.30 | 1.082 | win_both |
| hardtanh | poly:s3/d11 | 0.00 | 2.99 | 0.00 | 3.14 | 1.050 | win_both |
| i0 | poly:s1/d13 | 1183.00 | 3.35 | 2582.00 | 3.82 | 1.140 | win_both |
| i1 | rational:s1/d14d14 | 5.00 | 4.48 | 5.00 | 9.89 | 2.208 | win_both |
| identity | poly:s1/d16 | 0.00 | 1.83 | 0.00 | 1.90 | 1.038 | win_both |
| leaky_relu | poly:s2/d1 | 0.01 | 2.99 | -- | -- | -- | incomplete |
| lgamma | rational:s2/d5d3 | 1126.00 | 5.45 | 1685.67 | 21.33 | 3.914 | win_both |
| log | log:s2/d3d2 | 1.00 | 5.63 | 1.00 | 3.76 | 0.668 | accuracy_match_slow |
| log10 | exponent_alu_log2:s32/d6 | 21.50 | 3.29 | 1.00 | 3.89 | 1.182 | faster_less_accurate |
| log1p | exponent_alu_log2:s16/d6 | 25.00 | 3.31 | 1.00 | 3.88 | 1.172 | faster_less_accurate |
| log2 | exponent_alu_log2:s1/d6 | 36.00 | 3.28 | 1.00 | 3.87 | 1.180 | faster_less_accurate |
| logit | rational:s1/d6d6 | 1810.00 | 4.27 | 1.00 | 6.47 | 1.515 | faster_less_accurate |
| logsigmoid | rational:s2/d11d2 | 3.00 | 6.93 | -- | -- | -- | incomplete |
| mish | rational:s1/d13d12 | 5.00 | 6.39 | 2.00 | 5.94 | 0.930 | loss |
| multigammaln | poly:s1/d2 | 6168718.00 | 2.83 | 8301360.00 | 3.21 | 1.134 | excluded |
| polygamma | poly:s32/d13 | 1084562.00 | 26.46 | -- | -- | -- | incomplete |
| prelu | poly:s2/d13 | 0.00 | 2.94 | -- | -- | -- | incomplete |
| relu | poly:s2/d2 | 0.00 | 2.88 | 0.00 | 3.02 | 1.049 | win_both |
| relu6 | poly:s3/d6 | 0.00 | 3.00 | 0.00 | 3.06 | 1.020 | win_both |
| relu_max | poly:s3/d1 | 0.00 | 3.00 | -- | -- | -- | incomplete |
| relu_min | poly:s2/d1 | 0.00 | 2.93 | -- | -- | -- | incomplete |
| rsqrt | newton_root:s8/d1 | 47.00 | 3.00 | 1.00 | 3.33 | 1.110 | faster_less_accurate |
| selu | poly:s4/d9 | 14.62 | 4.26 | 11.00 | 4.40 | 1.033 | faster_less_accurate |
| sigmoid | rational:s2/d9d9 | 7.00 | 8.59 | 2.00 | 4.58 | 0.533 | loss |
| sigmoid_accurate | rational:s1/d8d7 | 8.00 | 4.74 | 2.00 | 4.58 | 0.966 | loss |
| silu | rational:s1/d10d6 | 4.00 | 4.89 | 2.00 | 4.62 | 0.945 | loss |
| sin | trig:s1/d9d1 | 1.00 | 4.00 | 1.00 | 3.41 | 0.853 | accuracy_match_slow |
| sinh | rational:s1/d13d12 | 5.00 | 4.25 | 1.00 | 4.48 | 1.054 | faster_less_accurate |
| softplus | rational:s1/d9d9 | 19.00 | 5.23 | 43.62 | 7.35 | 1.405 | win_both |
| softshrink | poly:s3/d6 | 0.00 | 3.05 | 0.00 | 3.25 | 1.066 | win_both |
| softsign | rational:s2/d1d1 | 2.00 | 3.14 | 3.00 | 3.19 | 1.016 | win_both |
| sqrt | newton_root:s3/d8 | 1.00 | 3.05 | 1.00 | 3.29 | 1.079 | win_both |
| swish | rational:s1/d8d8 | 4.00 | 4.90 | 2.00 | 4.61 | 0.941 | loss |
| tan | tan:s1/d9 | 17.00 | 4.00 | 1.00 | 4.64 | 1.160 | faster_less_accurate |
| tanh | rational:s1/d8d8 | 5.00 | 3.56 | 5.00 | 7.34 | 2.062 | win_both |
| tanhshrink | rational:s1/d12d4 | 227.75 | 3.66 | 0.50 | 6.71 | 1.833 | faster_less_accurate |
| threshold | poly:s2/d2 | 0.00 | 2.87 | -- | -- | -- | incomplete |

Excluded rows:
- multigammaln: frontier/native rows were generated before multigammaln.json was fixed to the true p=4 target; rerun p=4 sweep before claiming
