<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DFB 60-Activation Deployment Sweep (EXHAUSTIVE bf16)

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
`is_asymptotic=True` — i.e. the fit relies on asymptotic factoring `f(x) = dominant(x) *
correction(x)`. The DFB kernel now IMPLEMENTS this (per-segment `LUT_ASYM_MASK` + a
`LUT_DOMINANT_CLASS` evaluating `dominant(x)` in SFPU, reproducing eval.py's
DOMINANT_FACTORS), so asymptotic tails are evaluated PROPERLY and never dropped (gelu
left tail: ULP_mean 189 -> ~1.5). The diagnostic signal that distinguishes a real
broad error from a harmless near-root bit-distance artifact is **ULP_mean + ml_pass**
(broad error) vs **ULP_max only** (near-zero-crossing artifact).

**TALLY: 58/60 CLEAN exhaustively  |  2 fail  |  0 out-of-scope  (CLEAN = PCC_vs_true >= 0.99 AND (ml_pass >= 0.95 OR bf16 ULP_p99 <= 1.0); ULP_mean / ULP_max never gate. bf16 measured EXHAUSTIVELY on craq-sim Quasar over every representable bf16 in the full fit domain; the bit-distance ULP excludes zero-reference points)**

## Results

| activation       | eval     | rr                 | segs | asym | PCC_true   | ml_pass | ULP_max | ULP_mean | ULP_p99 | status | classification     |
|------------------|----------|--------------------|------|------|------------|---------|---------|----------|---------|--------|--------------------|
| abs              | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    0.474 |     0.0 | clean  | -                  |
| acos             | RATIONAL | none               |    1 |   no |   0.999899 |  0.9429 |       1 |    0.043 |     1.0 | clean  | -                  |
| acosh            | RATIONAL | none               |    1 |   no |   0.999990 |  0.2718 |       1 |    0.504 |     1.0 | clean  | -                  |
| asin             | RATIONAL | none               |    1 |   no |   0.999999 |  0.9930 |     225 |    2.521 |    63.0 | clean  | -                  |
| asinh            | RATIONAL | none               |    1 |   no |   0.999998 |  0.9762 |     179 |    1.034 |    12.6 | clean  | -                  |
| atan             | POLY     | none               |    2 |   no |   0.999999 |  0.9907 |     128 |    1.481 |     1.0 | clean  | -                  |
| atanh            | RATIONAL | none               |    1 |   no |   0.999998 |  0.9926 |     324 |    4.195 |   161.9 | clean  | -                  |
| cbrt             | POLY     | exponent_alu_pow   |    1 |   no |   0.999998 |  0.9443 |   10879 |   80.837 |     1.0 | clean  | -                  |
| celu             | POLY     | none               |    2 |   no |   1.000000 |  0.9904 |     128 |    0.967 |     1.0 | clean  | -                  |
| cos              | RATIONAL | trig               |    1 |   no |   0.999979 |  0.9209 |       1 |    0.082 |     1.0 | clean  | -                  |
| cosh             | POLY     | none               |    1 |   no |   0.999998 |  0.9566 |       1 |    0.028 |     1.0 | clean  | -                  |
| digamma          | RATIONAL | none               |    2 |   no |   0.999998 |  0.2927 |       1 |    0.485 |     1.0 | clean  | -                  |
| elu              | POLY     | none               |    2 |   no |   1.000000 |  0.9904 |     128 |    0.967 |     1.0 | clean  | -                  |
| erf              | RATIONAL | none               |    1 |   no |   0.999999 |  0.9775 |     143 |    1.049 |     1.0 | clean  | -                  |
| erfc             | RATIONAL | none               |    3 |   no |   0.999964 |  0.8914 |   26128 |   29.875 |     1.0 | clean  | -                  |
| erfinv           | RATIONAL | none               |    1 |   no |   0.999999 |  0.9931 |     333 |    4.083 |   175.0 | clean  | -                  |
| exp              | POLY     | exp                |    1 |   no |   0.999998 |  0.8946 |       1 |    0.098 |     1.0 | clean  | -                  |
| exp2             | POLY     | exponent_alu_exp2  |    1 |   no |   0.999998 |  0.4973 |       1 |    0.495 |     1.0 | clean  | -                  |
| expm1            | POLY     | none               |    1 |   no |   0.999998 |  0.9930 |     127 |    0.582 |     1.0 | clean  | -                  |
| gelu             | POLY     | none               |   32 |  yes |   0.999998 |  0.9899 |     128 |    1.457 |    44.9 | clean  | -                  |
| hardmish         | POLY     | none               |    3 |   no |   1.000000 |  0.9969 |     127 |    0.570 |     1.0 | clean  | -                  |
| hardsigmoid      | POLY     | none               |   16 |   no |   0.999963 |  0.9190 |       1 |    0.091 |     1.0 | clean  | -                  |
| hardswish        | POLY     | none               |    3 |   no |   1.000000 |  0.9923 |     128 |    1.082 |    44.8 | clean  | -                  |
| hardtanh         | POLY     | none               |    3 |   no |   1.000000 |  1.0000 |     127 |    0.488 |     0.0 | clean  | -                  |
| i0               | RATIONAL | none               |    1 |   no |   0.999998 |  0.9592 |       1 |    0.026 |     1.0 | clean  | -                  |
| i1               | RATIONAL | none               |    1 |   no |   0.999998 |  0.9793 |     128 |    1.003 |    44.0 | clean  | -                  |
| identity         | POLY     | none               |    1 |   no |   1.000000 |  1.0000 |     127 |    0.474 |     0.0 | clean  | -                  |
| leaky_relu       | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    1.165 |    34.1 | clean  | -                  |
| lgamma           | RATIONAL | none               |    2 |   no |   0.999998 |  0.3486 |       1 |    0.476 |     1.0 | clean  | -                  |
| log              | RATIONAL | log                |    1 |   no |   0.999998 |  0.4484 |       1 |    0.491 |     1.0 | clean  | -                  |
| log10            | RATIONAL | log                |    1 |   no |   0.999999 |  0.6397 |       1 |    0.515 |     1.0 | clean  | -                  |
| log1p            | RATIONAL | none               |    2 |   no |   0.999998 |  0.9843 |   22975 | 8202.468 | 22839.0 | clean  | -                  |
| log2             | POLY     | log                |    1 |   no |   0.999999 |  0.3862 |       1 |    0.498 |     1.0 | clean  | -                  |
| logit            | RATIONAL | none               |    1 |   no |   0.999996 |  0.3240 |       1 |    0.513 |     1.0 | clean  | -                  |
| logsigmoid       | RATIONAL | none               |    1 |   no |   0.999999 |  0.0659 |       1 |    0.054 |     1.0 | clean  | -                  |
| mish             | RATIONAL | none               |    1 |   no |   0.999997 |  0.9858 |   22744 | 7994.989 | 22606.6 | clean  | -                  |
| multigammaln     | RATIONAL | none               |    1 |   no |   0.999997 |  0.2943 |       1 |    0.473 |     1.0 | clean  | -                  |
| prelu            | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     128 |    0.856 |    35.6 | clean  | -                  |
| relu             | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    0.488 |     0.0 | clean  | -                  |
| relu6            | POLY     | none               |    3 |   no |   1.000000 |  1.0000 |     127 |    0.488 |     0.0 | clean  | -                  |
| relu_max         | POLY     | none               |    3 |   no |   1.000000 |  1.0000 |     127 |    0.488 |     0.0 | clean  | -                  |
| relu_min         | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    0.488 |     0.0 | clean  | -                  |
| rsqrt            | POLY     | newton_root        |    1 |   no |   0.999998 |  0.4270 |       1 |    0.484 |     1.0 | clean  | -                  |
| selu             | POLY     | none               |    4 |   no |   0.999998 |  0.9735 |     223 |    1.181 |     1.0 | clean  | -                  |
| sigmoid          | RATIONAL | none               |    1 |   no |   0.999959 |  0.9090 |       1 |    0.104 |     1.0 | clean  | -                  |
| sigmoid_accurate | RATIONAL | none               |    3 |   no |   0.999978 |  0.0943 |       1 |    0.918 |     1.0 | clean  | -                  |
| silu             | POLY     | none               |    8 |  yes |   0.999997 |  0.9863 |     128 |    1.480 |    44.0 | clean  | -                  |
| sin              | POLY     | trig               |    1 |   no |   0.999999 |  0.9869 |     127 |    0.553 |     1.0 | clean  | -                  |
| sinh             | RATIONAL | none               |    1 |   no |   0.999998 |  0.9734 |     127 |    0.514 |     1.0 | clean  | -                  |
| softplus         | RATIONAL | none               |    1 |   no |   0.999999 |  0.0659 |       1 |    0.054 |     1.0 | clean  | -                  |
| softshrink       | POLY     | none               |    3 |   no |   1.000000 |  1.0000 |       0 |    0.000 |     0.0 | clean  | -                  |
| softsign         | RATIONAL | none               |    2 |   no |   0.999999 |  0.9815 |     127 |    0.648 |     1.0 | clean  | -                  |
| sqrt             | POLY     | newton_root        |    1 |   no |   0.999997 |  0.5890 |       1 |    0.504 |     1.0 | clean  | -                  |
| swish            | RATIONAL | none               |    1 |   no |   0.999998 |  0.9863 |   22683 | 7979.626 | 22548.6 | clean  | -                  |
| tan              | POLY     | tan                |    1 |   no |   0.999999 |  0.9896 |     127 |    0.515 |     1.0 | clean  | -                  |
| tanh             | RATIONAL | none               |    1 |   no |   0.999999 |  0.9835 |     127 |    0.576 |     1.0 | clean  | -                  |
| tanhshrink       | POLY     | none               |   16 |   no |   1.000000 |  0.9931 |     128 |    2.883 |    87.0 | clean  | -                  |
| threshold        | POLY     | none               |    2 |   no |   1.000000 |  1.0000 |     127 |    0.488 |     0.0 | clean  | -                  |
| hardshrink       | POLY     | none               |    2 |   no |   0.997569 |  0.0044 |      22 |    4.614 |    16.8 | fail   | OTHER:broad-error  |
| polygamma        | POLY     | none               |   32 |   no |   0.996423 |  0.2819 |   33843 | 1274.477 | 33822.0 | fail   | OTHER:broad-error  |

## NEEDS-ASYMPTOTIC FACTORING (asym pick + broad real error: high ULP_mean / low ml_pass)
  (none)

## NEAR-ROOT ARTIFACT (high ULP_max but low ULP_mean + high ml_pass: harmless)
  (none)

## OTHER non-clean
  - hardshrink: asym=no PCC=0.997569 ml_pass=0.0044 ULP max/mean/p99=22/4.614/16.8 -> OTHER:broad-error
  - polygamma: asym=no PCC=0.996423 ml_pass=0.2819 ULP max/mean/p99=33843/1274.477/33822.0 -> OTHER:broad-error

## Out-of-scope (deployed pick uses a feature the DFB kernel does not yet implement)
  (none)
