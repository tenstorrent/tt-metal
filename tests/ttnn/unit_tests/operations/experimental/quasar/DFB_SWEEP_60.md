<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DFB 60-Activation Deployment Sweep

The DFB analog of the tt-llk `quasar_sweep.sh` deployment validator. Each of the
60 deployed activations is the fitter's TRUE bf16 shipping pick (`best_ulp_*` in
`best.csv`), resolved to a coefficient CSV with the same 3-tier tolerant glob, run
through the generic `ttnn.experimental.quasar.unary_lut` DFB op on craq-sim Quasar,
and PCC-checked vs the fitter `ground_truth` (the TRUE activation). PCC is MEASURED
on craq-sim, never assumed.

**TALLY: 60/60 pass  |  0 fail  |  0 out-of-scope  (threshold PCC_vs_true >= 0.99, measured on craq-sim Quasar)**

## Results

| activation       | eval     | rr                 | segs | PCC_true   | status       |
|------------------|----------|--------------------|------|------------|--------------|
| abs              | POLY     | none               |    2 |   1.000000 | pass         |
| acos             | RATIONAL | none               |    1 |   0.999972 | pass         |
| acosh            | RATIONAL | none               |    1 |   0.999955 | pass         |
| asin             | RATIONAL | none               |    1 |   0.999999 | pass         |
| asinh            | RATIONAL | none               |    1 |   0.999998 | pass         |
| atan             | POLY     | none               |    2 |   0.999999 | pass         |
| atanh            | RATIONAL | none               |    1 |   0.999996 | pass         |
| cbrt             | POLY     | exponent_alu_pow   |    1 |   0.999998 | pass         |
| celu             | POLY     | none               |    2 |   1.000000 | pass         |
| cos              | RATIONAL | trig               |    1 |   0.999999 | pass         |
| cosh             | POLY     | none               |    1 |   0.999998 | pass         |
| digamma          | RATIONAL | none               |    2 |   0.999992 | pass         |
| elu              | POLY     | none               |    2 |   1.000000 | pass         |
| erf              | RATIONAL | none               |    1 |   0.999998 | pass         |
| erfc             | POLY     | none               |   32 |   0.999997 | pass         |
| erfinv           | RATIONAL | none               |    1 |   0.999999 | pass         |
| exp              | POLY     | exponent_alu_exp2  |    1 |   0.999997 | pass         |
| exp2             | POLY     | exponent_alu_exp2  |    1 |   0.999998 | pass         |
| expm1            | POLY     | none               |    1 |   0.999998 | pass         |
| gelu             | POLY     | none               |   16 |   0.992005 | pass         |
| hardmish         | POLY     | none               |    3 |   1.000000 | pass         |
| hardshrink       | POLY     | none               |    2 |   0.999957 | pass         |
| hardsigmoid      | POLY     | none               |   16 |   0.999999 | pass         |
| hardswish        | POLY     | none               |    3 |   1.000000 | pass         |
| hardtanh         | POLY     | none               |    3 |   1.000000 | pass         |
| i0               | RATIONAL | none               |    1 |   0.999997 | pass         |
| i1               | RATIONAL | none               |    1 |   0.999997 | pass         |
| identity         | POLY     | none               |    1 |   1.000000 | pass         |
| leaky_relu       | POLY     | none               |    2 |   1.000000 | pass         |
| lgamma           | RATIONAL | none               |    2 |   0.999996 | pass         |
| log              | POLY     | exponent_alu_log2  |    1 |   0.999994 | pass         |
| log10            | POLY     | exponent_alu_log2  |    1 |   0.999996 | pass         |
| log1p            | RATIONAL | none               |    2 |   0.999994 | pass         |
| log2             | POLY     | exponent_alu_log2  |    1 |   0.999995 | pass         |
| logit            | RATIONAL | none               |    1 |   0.999999 | pass         |
| logsigmoid       | RATIONAL | none               |    2 |   0.999999 | pass         |
| mish             | RATIONAL | none               |    1 |   0.999994 | pass         |
| multigammaln     | RATIONAL | none               |    1 |   0.999995 | pass         |
| polygamma        | POLY     | none               |   64 |   1.000000 | pass         |
| prelu            | POLY     | none               |    2 |   1.000000 | pass         |
| relu             | POLY     | none               |    2 |   1.000000 | pass         |
| relu6            | POLY     | none               |    3 |   1.000000 | pass         |
| relu_max         | POLY     | none               |    3 |   1.000000 | pass         |
| relu_min         | POLY     | none               |    2 |   1.000000 | pass         |
| rsqrt            | POLY     | newton_root        |    1 |   0.999998 | pass         |
| selu             | POLY     | none               |    4 |   0.999998 | pass         |
| sigmoid          | POLY     | exponent_alu_exp2  |    1 |   0.999998 | pass         |
| sigmoid_accurate | POLY     | exponent_alu_exp2  |    1 |   0.999998 | pass         |
| silu             | POLY     | none               |    8 |   0.999035 | pass         |
| sin              | POLY     | trig               |    1 |   0.999999 | pass         |
| sinh             | POLY     | none               |    1 |   0.999998 | pass         |
| softplus         | RATIONAL | none               |    2 |   0.999999 | pass         |
| softshrink       | POLY     | none               |    3 |   1.000000 | pass         |
| softsign         | RATIONAL | none               |    2 |   0.999999 | pass         |
| sqrt             | POLY     | newton_root        |    1 |   0.999984 | pass         |
| swish            | RATIONAL | none               |    1 |   0.999997 | pass         |
| tan              | POLY     | tan                |    1 |   0.999999 | pass         |
| tanh             | RATIONAL | none               |    1 |   0.999999 | pass         |
| tanhshrink       | POLY     | none               |    4 |   0.999999 | pass         |
| threshold        | POLY     | none               |    2 |   1.000000 | pass         |

## Failures (DFB op ran, PCC_vs_true < 0.99)
  (none)

## Out-of-scope (deployed pick uses a feature the DFB kernel does not yet implement)
  (none)
