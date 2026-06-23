<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DFB 60-Activation Deployment Sweep (EXHAUSTIVE bf16 baseline)

The DFB analog of the tt-llk `quasar_sweep.sh` deployment validator. Each of the
60 deployed activations is the fitter's TRUE bf16 shipping pick (`best_ulp_*` in
`best.csv`), resolved to a coefficient CSV with the same 3-tier tolerant glob, run
through the generic `ttnn.experimental.quasar.unary_lut` DFB op on craq-sim Quasar.

Inputs are **EXHAUSTIVE**: every distinct representable bf16 value in the activation's
**FULL fit domain** `[lo, hi]` (asymptotic tail segments are NOT dropped). Output is
compared against the fitter `ground_truth` (the TRUE activation) with PCC, the bf16
sign-magnitude bit-distance ULP (max/mean/p99), and `ml_pass` (fraction within a
`1e-3` rel+abs tolerance band). All measured on craq-sim, never assumed.

`is_asymptotic` (asym column) is True iff the deployed CSV has >=1 segment with
`is_asymptotic=True` — i.e. the fit relies on asymptotic factoring the DFB kernel
does NOT yet implement. The diagnostic signal that distinguishes a real asymptotic
miss from a harmless near-root bit-distance artifact is **ULP_mean + ml_pass** (broad
error) vs **ULP_max only** (near-zero-crossing artifact).

**TALLY: 34/60 CLEAN exhaustively  |  26 fail  |  0 out-of-scope  (CLEAN = PCC_vs_true >= 0.99 AND bf16 ULP_p99 <= 1.0 AND bf16 ULP_mean <= 1.0; ml_pass MEASURED+REPORTED but does NOT gate. bf16 measured EXHAUSTIVELY on craq-sim Quasar over every representable bf16 in the full fit domain)**

## Results

| activation       | eval     | rr                 | segs | asym | PCC_true   | ml_pass | ULP_max | ULP_mean | ULP_p99 | status | classification     |
|------------------|----------|--------------------|------|------|------------|---------|---------|----------|---------|--------|--------------------|
| abs              | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    0.474 |     0.0 | clean  | -                  |
| acos             | RATIONAL | none               |    1 |   no |   0.999919 |  0.9620 |       1 |    0.024 |     1.0 | clean  | -                  |
| acosh            | RATIONAL | none               |    1 |   no |   0.999990 |  0.2789 |       1 |    0.497 |     1.0 | clean  | -                  |
| cbrt             | POLY     | none               |    1 |   no |   0.999537 |  0.2946 |       1 |    0.496 |     1.0 | clean  | -                  |
| celu             | POLY     | none               |    2 |   no |   1.000000 |  0.9904 |     128 |    0.967 |     1.0 | clean  | -                  |
| cos              | RATIONAL | trig               |    1 |   no |   0.999979 |  0.9209 |       1 |    0.082 |     1.0 | clean  | -                  |
| cosh             | POLY     | none               |    1 |   no |   0.999998 |  0.0268 |       1 |    0.958 |     1.0 | clean  | -                  |
| digamma          | RATIONAL | none               |    2 |   no |   0.999998 |  0.2927 |       1 |    0.485 |     1.0 | clean  | -                  |
| elu              | POLY     | none               |    2 |   no |   1.000000 |  0.9904 |     128 |    0.967 |     1.0 | clean  | -                  |
| erfc             | POLY     | none               |   32 |   no |   0.999915 |  0.5027 |       3 |    0.492 |     1.0 | clean  | -                  |
| exp2             | POLY     | exponent_alu_exp2  |    1 |   no |   0.999998 |  0.4973 |       1 |    0.495 |     1.0 | clean  | -                  |
| expm1            | POLY     | none               |    1 |   no |   0.999998 |  0.9930 |     127 |    0.582 |     1.0 | clean  | -                  |
| hardmish         | POLY     | none               |    3 |   no |   1.000000 |  0.9969 |     127 |    0.565 |     1.0 | clean  | -                  |
| hardsigmoid      | POLY     | none               |   16 |   no |   0.999963 |  0.9190 |   12928 |    0.478 |     1.0 | clean  | -                  |
| hardtanh         | POLY     | none               |    3 |   no |   1.000000 |  1.0000 |     127 |    0.488 |     0.0 | clean  | -                  |
| i0               | RATIONAL | none               |    1 |   no |   0.999998 |  0.0191 |       1 |    0.966 |     1.0 | clean  | -                  |
| identity         | POLY     | none               |    1 |   no |   1.000000 |  1.0000 |     127 |    0.474 |     0.0 | clean  | -                  |
| log              | POLY     | exponent_alu_log2  |    1 |   no |   0.999998 |  0.4484 |       1 |    0.491 |     1.0 | clean  | -                  |
| log10            | POLY     | exponent_alu_log2  |    1 |   no |   0.999999 |  0.6408 |       1 |    0.514 |     1.0 | clean  | -                  |
| logit            | RATIONAL | none               |    1 |   no |   0.999996 |  0.3345 |       1 |    0.494 |     1.0 | clean  | -                  |
| logsigmoid       | RATIONAL | none               |    2 |   no |   0.999999 |  0.0659 |       1 |    0.054 |     1.0 | clean  | -                  |
| multigammaln     | RATIONAL | none               |    1 |   no |   0.999997 |  0.2989 |       1 |    0.468 |     1.0 | clean  | -                  |
| polygamma        | POLY     | none               |   64 |   no |   0.999998 |  0.3665 |       1 |    0.487 |     1.0 | clean  | -                  |
| relu             | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    0.244 |     0.0 | clean  | -                  |
| relu6            | POLY     | none               |    3 |   no |   1.000000 |  1.0000 |     127 |    0.244 |     0.0 | clean  | -                  |
| relu_max         | POLY     | none               |    3 |   no |   1.000000 |  1.0000 |     127 |    0.244 |     0.0 | clean  | -                  |
| relu_min         | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    0.244 |     0.0 | clean  | -                  |
| sin              | POLY     | trig               |    1 |   no |   0.999999 |  0.9869 |     127 |    0.553 |     1.0 | clean  | -                  |
| sinh             | POLY     | none               |    1 |   no |   0.999998 |  0.9738 |     127 |    0.513 |     1.0 | clean  | -                  |
| softplus         | RATIONAL | none               |    2 |   no |   0.999999 |  0.0659 |       1 |    0.054 |     1.0 | clean  | -                  |
| softshrink       | POLY     | none               |    3 |   no |   1.000000 |  1.0000 |       0 |    0.000 |     0.0 | clean  | -                  |
| softsign         | RATIONAL | none               |    2 |   no |   0.999999 |  0.9815 |     127 |    0.648 |     1.0 | clean  | -                  |
| tan              | POLY     | tan                |    1 |   no |   0.999999 |  0.9896 |     127 |    0.515 |     1.0 | clean  | -                  |
| threshold        | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    0.244 |     0.0 | clean  | -                  |
| asin             | RATIONAL | none               |    1 |   no |   0.999999 |  0.9930 |     225 |    2.521 |    62.9 | fail   | OTHER:broad-error  |
| asinh            | RATIONAL | none               |    1 |   no |   0.999998 |  0.9762 |     179 |    1.034 |    12.6 | fail   | OTHER:broad-error  |
| atan             | POLY     | none               |    2 |   no |   0.999999 |  0.9907 |     128 |    1.481 |     1.0 | fail   | NEAR-ROOT-ARTIFACT |
| atanh            | RATIONAL | none               |    1 |   no |   0.999996 |  0.9934 |     268 |    3.190 |   105.9 | fail   | OTHER:broad-error  |
| erf              | RATIONAL | none               |    1 |   no |   0.999999 |  0.9775 |     143 |    1.049 |     1.0 | fail   | NEAR-ROOT-ARTIFACT |
| erfinv           | RATIONAL | none               |    1 |   no |   0.999999 |  0.9929 |     273 |    2.819 |   114.9 | fail   | OTHER:broad-error  |
| exp              | POLY     | exponent_alu_exp2  |    1 |   no |   0.977997 |  0.4656 |     578 |    6.668 |   219.0 | fail   | OTHER:broad-error  |
| gelu             | POLY     | none               |   16 |  yes |   0.988542 |  0.9837 |   31463 |  188.711 |   100.0 | fail   | NEEDS-ASYMPTOTIC   |
| hardshrink       | POLY     | none               |    2 |   no |   0.997569 |  0.0044 |   16105 | 15153.562 | 15914.2 | fail   | OTHER:broad-error  |
| hardswish        | POLY     | none               |    3 |   no |   1.000000 |  0.9923 |   13120 |    1.468 |    44.0 | fail   | OTHER:broad-error  |
| i1               | RATIONAL | none               |    1 |   no |   0.999997 |  0.9780 |     128 |    1.000 |    44.0 | fail   | OTHER:broad-error  |
| leaky_relu       | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    1.163 |    34.0 | fail   | OTHER:broad-error  |
| lgamma           | RATIONAL | none               |    2 |   no |   0.999998 |  0.3486 |   12035 |    6.160 |     1.0 | fail   | NEAR-ROOT-ARTIFACT |
| log1p            | RATIONAL | none               |    2 |   no |   0.999998 |  0.9843 |   23739 | 8750.933 | 23605.0 | fail   | OTHER:broad-error  |
| log2             | POLY     | exponent_alu_log2  |    1 |   no |   0.996350 |  0.0000 |   32135 | 4871.474 | 32129.0 | fail   | OTHER:broad-error  |
| mish             | RATIONAL | none               |    1 |   no |   0.999997 |  0.9859 |   24543 | 9288.490 | 24403.0 | fail   | OTHER:broad-error  |
| prelu            | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     128 |    0.856 |    35.6 | fail   | OTHER:broad-error  |
| rsqrt            | POLY     | none               |    1 |   no |        n/a |  0.0024 |     944 |  420.333 |   925.8 | fail   | OTHER:broad-error  |
| selu             | POLY     | none               |    4 |   no |   0.999998 |  0.9735 |     223 |    1.181 |     1.0 | fail   | NEAR-ROOT-ARTIFACT |
| sigmoid          | POLY     | exponent_alu_exp2  |    1 |   no |   0.323653 |  0.0001 |    1279 |  133.421 |   349.0 | fail   | OTHER:broad-error  |
| sigmoid_accurate | POLY     | exponent_alu_exp2  |    1 |   no |   0.323653 |  0.0001 |    1279 |  133.421 |   349.0 | fail   | OTHER:broad-error  |
| silu             | POLY     | none               |    8 |  yes |   0.989286 |  0.0130 |   32029 | 15757.361 | 31908.6 | fail   | NEEDS-ASYMPTOTIC   |
| sqrt             | POLY     | none               |    1 |   no |        n/a |  0.0052 |     927 |  414.584 |   918.0 | fail   | OTHER:broad-error  |
| swish            | RATIONAL | none               |    1 |   no |   0.999998 |  0.9863 |   24589 | 9352.380 | 24447.0 | fail   | OTHER:broad-error  |
| tanh             | RATIONAL | none               |    1 |   no |   0.999999 |  0.9835 |     128 |    1.469 |     1.0 | fail   | NEAR-ROOT-ARTIFACT |
| tanhshrink       | POLY     | none               |    4 |   no |   0.999999 |  0.9914 |    6142 |  368.605 |  5568.7 | fail   | OTHER:broad-error  |

## NEEDS-ASYMPTOTIC FACTORING (asym pick + broad real error: high ULP_mean / low ml_pass)
  - gelu: asym=yes PCC=0.988542 ml_pass=0.9837 ULP max/mean/p99=31463/188.711/100.0 -> NEEDS-ASYMPTOTIC
  - silu: asym=yes PCC=0.989286 ml_pass=0.0130 ULP max/mean/p99=32029/15757.361/31908.6 -> NEEDS-ASYMPTOTIC

## NEAR-ROOT ARTIFACT (high ULP_max but low ULP_mean + high ml_pass: harmless)
  - atan: asym=no PCC=0.999999 ml_pass=0.9907 ULP max/mean/p99=128/1.481/1.0 -> NEAR-ROOT-ARTIFACT
  - erf: asym=no PCC=0.999999 ml_pass=0.9775 ULP max/mean/p99=143/1.049/1.0 -> NEAR-ROOT-ARTIFACT
  - lgamma: asym=no PCC=0.999998 ml_pass=0.3486 ULP max/mean/p99=12035/6.160/1.0 -> NEAR-ROOT-ARTIFACT
  - selu: asym=no PCC=0.999998 ml_pass=0.9735 ULP max/mean/p99=223/1.181/1.0 -> NEAR-ROOT-ARTIFACT
  - tanh: asym=no PCC=0.999999 ml_pass=0.9835 ULP max/mean/p99=128/1.469/1.0 -> NEAR-ROOT-ARTIFACT

## OTHER non-clean
  - asin: asym=no PCC=0.999999 ml_pass=0.9930 ULP max/mean/p99=225/2.521/62.9 -> OTHER:broad-error
  - asinh: asym=no PCC=0.999998 ml_pass=0.9762 ULP max/mean/p99=179/1.034/12.6 -> OTHER:broad-error
  - atanh: asym=no PCC=0.999996 ml_pass=0.9934 ULP max/mean/p99=268/3.190/105.9 -> OTHER:broad-error
  - erfinv: asym=no PCC=0.999999 ml_pass=0.9929 ULP max/mean/p99=273/2.819/114.9 -> OTHER:broad-error
  - exp: asym=no PCC=0.977997 ml_pass=0.4656 ULP max/mean/p99=578/6.668/219.0 -> OTHER:broad-error
  - hardshrink: asym=no PCC=0.997569 ml_pass=0.0044 ULP max/mean/p99=16105/15153.562/15914.2 -> OTHER:broad-error
  - hardswish: asym=no PCC=1.000000 ml_pass=0.9923 ULP max/mean/p99=13120/1.468/44.0 -> OTHER:broad-error
  - i1: asym=no PCC=0.999997 ml_pass=0.9780 ULP max/mean/p99=128/1.000/44.0 -> OTHER:broad-error
  - leaky_relu: asym=no PCC=1.000000 ml_pass=1.0000 ULP max/mean/p99=127/1.163/34.0 -> OTHER:broad-error
  - log1p: asym=no PCC=0.999998 ml_pass=0.9843 ULP max/mean/p99=23739/8750.933/23605.0 -> OTHER:broad-error
  - log2: asym=no PCC=0.996350 ml_pass=0.0000 ULP max/mean/p99=32135/4871.474/32129.0 -> OTHER:broad-error
  - mish: asym=no PCC=0.999997 ml_pass=0.9859 ULP max/mean/p99=24543/9288.490/24403.0 -> OTHER:broad-error
  - prelu: asym=no PCC=1.000000 ml_pass=1.0000 ULP max/mean/p99=128/0.856/35.6 -> OTHER:broad-error
  - rsqrt: asym=no PCC=nan ml_pass=0.0024 ULP max/mean/p99=944/420.333/925.8 -> OTHER:broad-error
  - sigmoid: asym=no PCC=0.323653 ml_pass=0.0001 ULP max/mean/p99=1279/133.421/349.0 -> OTHER:broad-error
  - sigmoid_accurate: asym=no PCC=0.323653 ml_pass=0.0001 ULP max/mean/p99=1279/133.421/349.0 -> OTHER:broad-error
  - sqrt: asym=no PCC=nan ml_pass=0.0052 ULP max/mean/p99=927/414.584/918.0 -> OTHER:broad-error
  - swish: asym=no PCC=0.999998 ml_pass=0.9863 ULP max/mean/p99=24589/9352.380/24447.0 -> OTHER:broad-error
  - tanhshrink: asym=no PCC=0.999999 ml_pass=0.9914 ULP max/mean/p99=6142/368.605/5568.7 -> OTHER:broad-error

## Out-of-scope (deployed pick uses a feature the DFB kernel does not yet implement)
  (none)
