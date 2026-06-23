<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DFB 60-Activation Deployment Sweep

The DFB analog of the tt-llk `quasar_sweep.sh` deployment validator. Each of the
60 deployed activations is the fitter's TRUE bf16 shipping pick (`best_ulp_*` in
`best.csv`), resolved to a coefficient CSV with the same 3-tier tolerant glob, run
through the generic `ttnn.experimental.quasar.unary_lut` DFB op on craq-sim Quasar,
and PCC-checked vs the fitter `ground_truth` (the TRUE activation). PCC is MEASURED
on craq-sim, never assumed.

**TALLY: 55/60 pass  |  3 fail  |  2 out-of-scope  (threshold PCC_vs_true >= 0.99, measured on craq-sim Quasar)**

## Results

| activation       | eval     | rr                 | segs | PCC_true   | status       |
|------------------|----------|--------------------|------|------------|--------------|
| abs              | POLY     | none               |    2 |   0.999893 | pass         |
| acos             | RATIONAL | none               |    1 |   0.999838 | pass         |
| acosh            | RATIONAL | none               |    1 |   0.999950 | pass         |
| asin             | RATIONAL | none               |    1 |   0.999861 | pass         |
| asinh            | RATIONAL | none               |    1 |   0.999996 | pass         |
| atan             | POLY     | none               |    2 |   0.999994 | pass         |
| atanh            | RATIONAL | none               |    1 |   0.999996 | pass         |
| cbrt             | POLY     | exponent_alu_pow   |    1 |   0.999998 | pass         |
| celu             | POLY     | none               |    2 |   0.999967 | pass         |
| cos              | RATIONAL | trig               |    1 |   0.999999 | pass         |
| elu              | POLY     | none               |    2 |   0.999967 | pass         |
| erf              | RATIONAL | none               |    1 |   0.999998 | pass         |
| erfc             | POLY     | none               |   32 |   0.999998 | pass         |
| erfinv           | RATIONAL | none               |    1 |   0.999741 | pass         |
| exp              | POLY     | exponent_alu_exp2  |    1 |   0.999997 | pass         |
| exp2             | POLY     | exponent_alu_exp2  |    1 |   0.999998 | pass         |
| expm1            | POLY     | none               |    1 |   0.999916 | pass         |
| gelu             | POLY     | none               |   16 |   0.992660 | pass         |
| hardmish         | POLY     | none               |    3 |   0.999961 | pass         |
| hardshrink       | POLY     | none               |    2 |   0.999935 | pass         |
| hardsigmoid      | POLY     | none               |   16 |   0.999999 | pass         |
| hardswish        | POLY     | none               |    3 |   0.999961 | pass         |
| hardtanh         | POLY     | none               |    3 |   1.000000 | pass         |
| i0               | RATIONAL | none               |    1 |   0.990254 | pass         |
| i1               | RATIONAL | none               |    1 |   0.992123 | pass         |
| identity         | POLY     | none               |    1 |   0.999973 | pass         |
| leaky_relu       | POLY     | none               |    2 |   0.999960 | pass         |
| lgamma           | RATIONAL | none               |    2 |   0.999956 | pass         |
| log              | POLY     | exponent_alu_log2  |    1 |   0.999994 | pass         |
| log10            | POLY     | exponent_alu_log2  |    1 |   0.999996 | pass         |
| log1p            | RATIONAL | none               |    2 |   0.993444 | pass         |
| log2             | POLY     | exponent_alu_log2  |    1 |   0.999995 | pass         |
| logit            | RATIONAL | none               |    1 |   0.997918 | pass         |
| logsigmoid       | RATIONAL | none               |    2 |   0.999957 | pass         |
| mish             | RATIONAL | none               |    1 |   0.999946 | pass         |
| multigammaln     | RATIONAL | none               |    1 |   0.999978 | pass         |
| prelu            | POLY     | none               |    2 |   0.999969 | pass         |
| relu             | POLY     | none               |    2 |   0.999960 | pass         |
| relu6            | POLY     | none               |    3 |   1.000000 | pass         |
| relu_max         | POLY     | none               |    3 |   1.000000 | pass         |
| relu_min         | POLY     | none               |    2 |   0.999960 | pass         |
| selu             | POLY     | none               |    4 |   0.999974 | pass         |
| sigmoid          | POLY     | exponent_alu_exp2  |    1 |   0.999998 | pass         |
| sigmoid_accurate | POLY     | exponent_alu_exp2  |    1 |   0.999998 | pass         |
| silu             | POLY     | none               |    8 |   0.999097 | pass         |
| sin              | POLY     | trig               |    1 |   0.999999 | pass         |
| sinh             | POLY     | none               |    1 |   0.991146 | pass         |
| softplus         | RATIONAL | none               |    2 |   0.999959 | pass         |
| softshrink       | POLY     | none               |    3 |   0.999971 | pass         |
| softsign         | RATIONAL | none               |    2 |   0.999999 | pass         |
| swish            | RATIONAL | none               |    1 |   0.999955 | pass         |
| tan              | POLY     | tan                |    1 |   0.999999 | pass         |
| tanh             | RATIONAL | none               |    1 |   0.999999 | pass         |
| tanhshrink       | POLY     | none               |    4 |   0.999965 | pass         |
| threshold        | POLY     | none               |    2 |   0.999960 | pass         |
| cosh             | POLY     | none               |    1 |   0.989081 | fail         |
| digamma          | RATIONAL | none               |    2 |  -0.026135 | fail         |
| polygamma        | POLY     | none               |   64 |   0.169024 | fail         |
| rsqrt            | POLYNOMI | newton_root        |    1 |     n/a    | out-of-scope |
| sqrt             | POLYNOMI | newton_root        |    1 |     n/a    | out-of-scope |

## Failures (DFB op ran, PCC_vs_true < 0.99)
  - cosh: PCC_vs_true 0.989081 < 0.99  (PCC_vs_approx=0.999998 -> DFB path faithful; fit-on-domain miss)
  - digamma: PCC_vs_true -0.026135 < 0.99  (PCC_vs_approx=1.000000 -> DFB path faithful; fit-on-domain miss)
  - polygamma: PCC_vs_true 0.169024 < 0.99  (PCC_vs_approx=1.000000 -> DFB path faithful; fit-on-domain miss)

## Out-of-scope (deployed pick uses a feature the DFB kernel does not yet implement)
  - rsqrt: RR method 'newton_root' not implemented in DFB kernel
  - sqrt: RR method 'newton_root' not implemented in DFB kernel
