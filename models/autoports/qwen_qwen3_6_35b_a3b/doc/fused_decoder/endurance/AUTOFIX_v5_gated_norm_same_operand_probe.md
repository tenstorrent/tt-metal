# Actual-operand gated norm local discriminator

Probe: `tests/v5_gated_norm_same_operand_probe.py`. No production/snapshot changes or device execution. CPU-only oracle validation passed for combined layer34 captures128/129/130/137/181. All capture/source identities were verified, canonical HF core/Z reproduces captured HF gated output bitwise, and canonical same-TT-input output reproduces the diagnostic's existing CPU tensor. A reporting-only final addition records each candidate against actual captured TT gated output; this does not change the adapters.

Three paths use the same captured FP32 recurrent output, with a mandatory BF16 core boundary, actual BF16 Z and actual layer34 BF16 gamma:

1. BF16-input native unweighted RMSNorm, BF16 output (current combined candidate's local path).
2. BF16 core widened toFP32, native RMSNorm FP32 output, explicit BF16 cast. This probes whether input/output format controls native intermediate precision.
3. Explicit FP32 square, sum/128, epsilon add, rsqrt, multiply, then BF16 cast. This isolates variance/inverse differences as well as output packing.

Every path uses identical subsequent accurate SFPU gamma multiplication with BF16 round/widen, followed by FP32 SiLU(Z) gating and final BF16 output. The probe records unweighted normalized, gamma-product and final gated tensors for all paths; nativeFP32 and explicit paths additionally retain unrounded normalizedFP32, and explicit retains squares/sum/variance/shifted/inverse. A standalone SiLU result is separately recorded as a local diagnostic; final output uses the same fused input SiLU activation as production, not the standalone result.

CPU reference uses the actual BF16 rounding boundaries and native torch FP32 mean/rsqrt. Code differences and PCCs are reported at every available stage, with full tensors in `.tensors.pt`. Exact equality is not silently required to pass a model quality gate or waived: this is an intermediate diagnostic only. There are no timing claims or host work inside a measured path. Device open is explicitly physical chip0; try/finally closes it and records failures. Existing artifacts cannot be overwritten.

Run once in the parent's serial device queue:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_gated_norm_same_operand_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_gated_norm_same_operand_device.json
```

The existing native RMSNorm API preserves FP32 input dtype at its output; unlike the previous BF16 case, its format-dependent intermediate selection can differ. This is a source-supported discriminator, not a promise of exact canonical arithmetic. Native/explicit variance reductions and rsqrt still require the actual result evidence.

## Parent-run measured result

`v5_gated_norm_same_operand_device.json` completed and closed the device cleanly. All five actual combined-candidate captures were tested. Native BF16 input has370–418 differing unweighted normalized BF16 values; native FP32 input followed by BF16 cast still has366–414. Thus widening the native input is not sufficient for these cases. Explicit FP32 square/sum/mean/epsilon/rsqrt/multiply produces bit-exact normalized **and gamma-weighted BF16** tensors at all five captures. Final gated output is bit-exact at129/130/137/181 and differs in one BF16 value at128; this residual is not waived or called exact.

These are local actual-operand measurements, not complete decoder accuracy or latency. The native source's FPU final normalization product is a plausible precision contributor, but this three-path comparison does not isolate all internal native arithmetic stages.

Artifact SHA256 `c8b8a36c07da80587b849392a030e87e87c91a12316fac197175df20fb56cc31`; provenance SHA256 `096b0350806a44e07036493e11ab0be10cdc9b44de33b8ebaa95d6e6bcad5668`; executed probe SHA256 `774e80898f714a355cfeecf6b38ba66ff08a7a857c7853c688f47911224dc4da`. The tensor sidecar SHA256 is `2e56f080a5c611464e697b2ed3a4771e952aa8ef42a8b14d42ce4751221f4432`.
Log `v5_gated_norm_same_operand_device.log` SHA256 `8ed2571a224e477e9df37b1db27dcaa016d35b7b595ef5735e80d45909d50f70`.
