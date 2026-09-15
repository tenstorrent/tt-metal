# Improved SDPA: distribution sweep and maximum relative error

September 9, 2026. Blackhole P100A, reservation 214149, yyzo-bh-26.
Uses the retained candidate in ../perf-v2/final-candidate.patch, based on main
2ba6fc2339d53300ae87c5202f335ef56492cfb3. No kernel changes in this sweep.

Follow-up: [large common modes on Q/K/V](../common-mode/REPORT.md) exposes
further failures, including FP32 L2 2.7523% for Q+32 and 1.1204% for K+32.

## Method

Non-causal B=1, H=10, source Q=K=262144, D=128, Q chunk=128, K chunk=512,
HiFi2, BF16 Q/K/V/output. Seed 1236 for every distribution and both paths.
Generate the full inputs, select 128 evenly spread query rows per head, and
execute those rows using --sampled-device. This compares 163,840 output
elements per case; these are not maxima over all 256K query rows, nor
full-prefill performance measurements.

BF16 uses the improved compute_streaming path, BF16 destination accumulation,
and no Q preprocessing. FP32 uses the improved non-streaming path with
six-fraction-bit bit-ceiling Q preprocessing and compensated scale c=1.0027.
Both are compared against an FP64 reference on the same original BF16 inputs.

- Relative L2 (%) = 100 * ||actual-reference||2 / ||reference||2.
- Maximum elementwise relative error (%) =
  100 * max_i |actual_i-reference_i| / |reference_i|, with no epsilon floor.
- Exact reference/output zeros contribute zero. A nonzero output at an exact
  reference zero is unbounded; the JSON uses null and an explicit mismatch
  count rather than nonstandard Infinity.
- Raw files also contain the reference/output at the worst element, reference
  RMS, maximum absolute error, per-row L2 summaries, and maximum relative
  error restricted to |reference| >= 1% of its global RMS. This last metric
  is supplementary, not a replacement for the requested unfiltered maximum.

## Input definitions

Inputs are independent standard Gaussian tensors before BF16 rounding,
except for the stated modification:

- normal: Q/K/V ~ N(0,1).
- scaled_qk: Q and K multiplied by two, giving each standard deviation two.
- outliers: independently add 10*N(0,1) at 0.1% of entries in each of Q/K/V.
- biased_v: add one to V, so V ~ N(1,1).
- uniform: Q=0, giving uniform attention; K/V remain Gaussian. This does not
  mean uniformly distributed random Q/K/V.
- constant_v: V=1, with Gaussian Q/K; exact mathematical output is one.

## Results

All numbers below are percentages. All 12 runs completed successfully; no
exact-zero reference mismatches occurred.

| Distribution | BF16 L2 | BF16 max relative | FP32 L2 | FP32 max relative |
|---|---:|---:|---:|---:|
| normal | 3.162 | 247,839.931 | 0.488 | 445,520.254 |
| scaled_qk | 5.491 | 629,346.930 | 0.866 | 102,612.761 |
| outliers | 3.775 | 288,847.180 | 0.630 | 346,396.616 |
| biased_v | 0.902 | 2.376 | 0.179 | 0.616 |
| uniform | 2.013 | 6,840.262 | 0.187 | 44.805 |
| constant_v | 0.542 | 1.563 | 3.621e-14 | 1.110e-13 |

The 0.5% FP32 L2 result is NOT distribution-independent: scaled Q/K and sparse
outliers fail it here. BF16 also varies, from about 0.54% to 5.49% L2. These
are single-seed synthetic diagnostics, not model-derived acceptance tests.
No main-kernel distribution sweep was run in this follow-up, so these data
alone do not quantify improvement over main on every distribution.

The huge maximum relative errors are real under the stated definition but
are dominated by near-zero reference elements. For normal inputs, the worst
reference element for both paths is 7.65087684258e-9. BF16 returns
-1.89542770386e-5; FP32 returns 3.40938568115e-5. The reference RMS is
0.00324617. Thus FP32 has lower aggregate error but a larger unfiltered
maximum in this case. A low L2 does not imply a small error at every element.
Constant-V FP32 errors around 1e-13% reflect FP64-reference roundoff about the
exact output one; they should be interpreted as effectively zero.

For context only, maxima after excluding |reference| < 1% of its RMS:

| Distribution | BF16 filtered max relative % | FP32 filtered max relative % |
|---|---:|---:|
| normal | 942.415 | 145.016 |
| scaled_qk | 2,174.495 | 443.854 |
| outliers | 3,267.561 | 400.298 |
| biased_v | 2.376 | 0.616 |
| uniform | 309.670 | 3.850 |
| constant_v | 1.563 | 1.110e-13 |

This cutoff is explicitly different from the requested raw maximum; it is
not used in the primary table. In particular, even this filter does not make
all elementwise relative errors small.

## Reproduction

Use the built remote candidate and the updated repro script:

    export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
    export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
    python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
      --kv-lens 262144 --full --sampled-device --query-sampling spread \
      --heads 10 --seed 1236 --benchmark-iters 2 --distribution normal \
      --variants hifi2 --output bf16-normal.jsonl

For FP32, replace hifi2 with fp32_hifi2 and add:

    --q-round-bits 6 --q-prescale 1.0027 --q-bitceil

Repeat for the distribution names above. The kernel sources and build are
unchanged from the preceding performance experiment. Only the Python repro
adds metrics/distributions. The reference self-check includes known 100%
relative-error and exact-zero cases; each device run checks finite outputs
and exact equality between ordinary execution and trace execution.
