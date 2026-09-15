# Improved SDPA: large common-mode inputs

September 9, 2026. Same built Blackhole P100A candidate as the preceding
distribution sweep; no kernel changes. Non-causal, source Q=K=256K, H=10,
D=128, Q/K chunks 128/512, HiFi2, seed 1236. Both paths output BF16.

Start from independent N(0,1) Q/K/V, add +8 or +32 to every element of ONE
tensor (Q, K, or V), then quantize inputs to BF16. Thus the common vector is
c*[1,...,1], identical across tokens and heads. The other two tensors remain
unchanged. Reference uses those same already-quantized inputs in FP64.
BF16 streaming uses no Q preprocessing; improved FP32 uses its c=1.0027
bit-ceiling Q preprocessing and compensated attention scale.

As before, execute 128 evenly spread query rows per head with sampled-device
mode. Each maximum covers 163,840 output elements, not the entire 256K-query
output. These are accuracy diagnostics, not full-prefill timings.

## Metrics and interpretation

L2 (%) = 100*||actual-reference||2/||reference||2.
Max relative (%) = 100*max_i |actual_i-reference_i|/|reference_i|, no epsilon
floor. Near-zero reference outputs can make this maximum enormous.

In exact arithmetic:

- Adding the same vector to every K adds a per-query constant to all logits;
  softmax cancels it. This is a useful test of score precision/cancellation.
- Adding the same vector to every V shifts the output by that vector.
  A large output constant can hide error in the small residual signal.
- Adding a common vector to Q generally changes the attention weights;
  unlike K, this is not a softmax invariance. Large Q offsets can make the
  distribution more peaked and change sensitivity to numerical error.

BF16 input rounding after the shift also changes the small input residuals.
We do not compare a shifted reference to an unshifted reference and call that
input-quantization difference an operator error. The reference self-check
tests K/V identities using the already-quantized shifted tensors.

For common V, we additionally report 100*||actual-reference||2/||reference-c||2.
It keeps the same error numerator and measures it against the residual signal,
without fitting or subtracting an estimated gain. The ideal BF16 output
rounding floor is essential context: at these offsets, the residual is much
smaller than the output BF16 spacing, so even a perfect kernel cannot retain
it in its BF16 output.

## Results

All entries are percentages. All 12 device runs completed, finite/trace
equality checks passed, and there were no exact-zero reference mismatches.

| Input offset | BF16 L2 | BF16 max relative | FP32 L2 | FP32 max relative |
|---|---:|---:|---:|---:|
| Q + 8 | 7.1893 | 11,533,017.6207 | 1.1448 | 2,133,231.1742 |
| Q + 32 | 15.6773 | 445,686.9982 | 2.7523 | 113,257.2653 |
| K + 8 | 5.1336 | 395,117.6237 | 0.5680 | 13,932.7181 |
| K + 32 | 16.5910 | 1,004,731.3294 | 1.1204 | 95,246.9549 |
| V + 8 | 0.9733 | 2.4204 | 0.0408 | 0.3024 |
| V + 32 | 0.9643 | 2.3671 | 0.0105 | 0.0513 |

For comparison, the previous matched no-offset Gaussian sweep gave BF16
L2 3.1620% and FP32 L2 0.4883%. The 0.5% FP32 target is not robust to these
Q/K common modes. No claim of distribution-independent accuracy is justified.

V common-mode residual diagnostics (percentages):

| Input offset | BF16 error / residual norm | FP32 error / residual norm | Ideal BF16 rounding error / residual norm | Ideal BF16 rounding ordinary L2 |
|---|---:|---:|---:|---:|
| V + 8 | 2,396.8853 | 100.5417 | 99.9999 | 0.0406 |
| V + 32 | 9,174.7551 | 100.0000 | 100.0000 | 0.0105 |

At V+32, FP32 exactly matches the ideal BF16 rounding L2 and loses essentially
all of the tiny residual signal, as a BF16 output must here. At V+8 it is
close to that floor. BF16 streaming's approximately 0.97% ordinary L2 hides
an error 24–92 times the residual norm, substantially beyond output rounding.
The compensated state update does not eliminate all common-mode sensitivity.

For common K, the exact cancellation identity makes this a particularly
useful follow-up target. The degradation is consistent with finite-precision
score formation/subtraction losing small logit differences in the presence
of a common offset. This sweep does not isolate individual kernel stages,
so it does not establish that as the sole cause. Common-Q errors additionally
involve genuinely changed/peaked attention and Q preprocessing sensitivity.

## Reproduction

Use the existing candidate build with the updated repro:

    export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
    export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
    python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
      --kv-lens 262144 --full --sampled-device --query-sampling spread \
      --heads 10 --seed 1236 --benchmark-iters 2 \
      --distribution common_k --common-mode 32 \
      --variants hifi2 --output bf16-common_k-32.jsonl

For FP32 use --variants fp32_hifi2 and add
--q-round-bits 6 --q-prescale 1.0027 --q-bitceil.
Repeat with common_q/common_k/common_v and common-mode 8/32.
Raw JSONL/log files are named by path, distribution, and offset.

No original-main comparison was run for these new distributions; these are
absolute error measurements of the retained improved implementations.
