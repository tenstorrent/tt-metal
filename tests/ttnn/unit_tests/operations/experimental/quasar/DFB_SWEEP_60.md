<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DFB 60-Activation Deployment Sweep

The DFB analog of the tt-llk `quasar_sweep.sh` deployment validator. Each of the
60 deployed activations is the fitter's TRUE bf16 shipping pick (`best_ulp_*` in
`best.csv`), resolved to a coefficient CSV with the same 3-tier tolerant glob, run
through the generic `ttnn.experimental.quasar.unary_lut` DFB op on craq-sim Quasar,
and PCC-checked vs the fitter `ground_truth` (the TRUE activation). PCC is MEASURED
on craq-sim, never assumed.

**TALLY: 2/2 pass  |  0 fail  |  0 out-of-scope  (threshold PCC_vs_true >= 0.99, measured on craq-sim Quasar)**

## Results

| activation       | eval     | rr                 | segs | PCC_true   | status       |
|------------------|----------|--------------------|------|------------|--------------|
| rsqrt            | POLY     | newton_root        |    1 |   0.999998 | pass         |
| sqrt             | POLY     | newton_root        |    1 |   0.999984 | pass         |

## Failures (DFB op ran, PCC_vs_true < 0.99)
  (none)

## Out-of-scope (deployed pick uses a feature the DFB kernel does not yet implement)
  (none)
