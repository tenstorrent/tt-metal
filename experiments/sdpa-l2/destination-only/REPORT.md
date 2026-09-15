# SDPA: destination-only FP32 experiment

## Scope and method

Measured on the reserved Blackhole P100A (`yyzo-bh-08`, IRD 212869), main
`2ba6fc2339d53300ae87c5202f335ef56492cfb3`. No operator patches were applied.
This is not a Galaxy or NVIDIA measurement.

There are 58 hardware results in eight adjacent JSONL files: 29 matched pairs.
Each pair reuses identical BF16 Q/K/V and changes **only** `fp32_dest_acc_en`.
Both variants use HiFi2, `math_approx_mode=True`, `exp_approx_mode=True`, and
`packer_l1_acc=False`. Unlike the earlier report's `fp32` variant, the current
`fp32_hifi2` variant does not also change fidelity or approximation flags.
Pair metadata was audited for equality of inputs and all other settings.

Reference: bounded-memory FP64 online softmax on the same BF16 inputs, checked
against dense FP64 attention. Relative L2 is `100 * ||actual-reference|| / ||reference||`.
The BF16 output-rounding-only floor is approximately 0.165–0.168% for normal inputs.
Reported times include compilation and are not performance measurements.

## Normal inputs

B=1, H=1, Q=128, D=128, noncausal; Q chunk=128, K chunk=512; seed=1234.

| KV length | Destination BF16: L2 % | Destination FP32: L2 % | FP32 fitted gain |
|---:|---:|---:|---:|
| 4,096 | 2.5081 | 2.6352 | 0.98759 |
| 32,768 | 2.8364 | 2.4941 | 0.99827 |
| 131,072 | 5.0139 | 4.4647 | 1.03262 |
| 262,144 | 18.9419 | 8.8327 | 1.08002 |

FP32 improves long-context accuracy substantially but still misses 0.5% by a large
margin. At 256K, PCC remains 0.999398 despite 8.83% L2. Fitting
`actual = gain * reference + residual` gives approximately +8.00% gain error
and 3.74% residual relative L2. These orthogonal components explain the total;
the residual is a diagnostic, not a measured fixed implementation.

The result repeats with seed 1235: 256K L2 changes from 18.0281% to 8.7083%.
Full square causal attention with B=1, H=4, D=128 gives 18.2090% to 8.7205%
at 256K. Causal metrics cover the last 128 query rows per head, not the entire
output; the device computes full causal attention.

## What the flag actually changes

The factory's `can_use_streaming_compute()` returns `!fp32_dest_acc_en`.
Thus this is a controlled public-configuration comparison, but **not** a comparison
of the same streaming kernel with wider destination registers.

| Component | Destination off | Destination on |
|---|---|---|
| Compute path | streaming | standard |
| QK / exponential intermediate CB | BF16 | FP32 |
| Running denominator `sum_A/B` CBs | BF16 | FP32 |
| Running numerator `out_im_A/B` CBs | BF16 | BF16 |
| Max and max-correction CBs | BF16 | BF16 |
| Q/K/V and final output | BF16 | BF16 |
| Destination tile capacity used for subblocking | 8 | 4 |

See `sdpa_program_factory.cpp`, lines 78, 477, 780–784, and 881–888, under
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/`.
The internal CB promotions are expected; the remaining BF16 state and changed
algorithm matter when interpreting the results.

## Controls locate remaining problems

### Uniform attention and constant values

Set Q=0 and V=1. Exact attention output is 1, independent of K and sequence length.
This removes QK error and any variation in the true softmax probabilities.

| KV length | BF16 destination L2 % | FP32 destination L2 % | FP32 output (every element) |
|---:|---:|---:|---:|
| 512 (one chunk) | 0.78125 | 3.125 | 0.96875 |
| 4,096 | 1.953125 | 3.515625 | 0.96484375 |
| 32,768 | 0.390625 | 0.78125 | 1.0078125 |
| 131,072 | 0.390625 | 2.734375 | 0.97265625 |
| 262,144 | 0.78125 | 51.5625 | 0.484375 |

The final value is exactly half the single-chunk value. Together with the CB
formats, this is consistent with BF16 numerator updates losing increments or
saturating while the promoted denominator continues growing. The BF16/BF16
case can conceal correlated numerator/denominator errors through cancellation.
This is a source-supported inference; internal CB values were not instrumented.

Two separate controls reinforce that promotion is not uniformly beneficial:

- Q=0, random V at 256K: L2 improves from 97.0286% to 5.6164%.
- Random Q/K, V=1 at 256K: L2 worsens from 15.1294% to 20.4842%.

### A different reciprocal can contribute short-context bias

On Blackhole, streaming normalization explicitly uses `recip_tile_init<false>()`
and `recip_tile<false>` (`compute_streaming.hpp`, around line 754).
The standard path instead calls `recip_tile_first_column` with default legacy
compatibility (`compute_common.hpp`, around line 267).
`ckernel_sfpu_sdpa.h`, line 38, selects `_reciprocal_compat_<APPROX ? 2 : 3>`.
With the unchanged approximate-math flag, that means two Newton steps.

The implementation in `ckernel_sfpu_rsqrt_compat.h`, around lines 62–84, starts
with 1.442695 on a normalized mantissa m in [0.5, 1). In exact arithmetic its
two-step relative reciprocal error is `-(1 - 1.442695*m)^4`: a low bias up to
approximately 3.84%. Three steps reduce this analytical bound to about 0.148%.
These are mathematical bounds for that routine, not isolated hardware results.
The measured 3.125% single-chunk constant-output deficit is consistent with this
mechanism, but does not establish it as the sole cause.

Normal random inputs also retain error with only one chunk: 2.4112% off versus
2.7746% on. Therefore long-context recurrence is not the only issue.

### More recurrence steps still increase error

At 256K, destination-FP32 normal-input L2 is 53.3626%, 20.3729%, and 8.8327%
for K chunks 128, 256, and 512 respectively (Q chunk stays 128). Corresponding
destination-off values are 133.6583%, 67.8522%, and 18.9419%.
This strongly implicates repeated state updates, though chunk size also changes
the matmul/reduction schedule.

FP32 destination is not end-to-end FP32: the standard compute code retains
BF16 output state and approximate exponentiation. Its main exponentiation calls
explicitly use the approximate template. FPU operations passing FP32 CB data
through SrcA/B can also encounter TF32 precision; the repository accuracy guide
warns that full-width copies require the appropriate unpack-to-destination mode.
The contribution of each remaining mechanism has not been independently measured.

## Next targeted experiments

1. Change only the standard-path reciprocal implementation/iteration count and
   rerun the single-chunk Q=0, V=1 control, then normal inputs.
2. Promote the running numerator as well as the denominator, checking arithmetic
   and unpack paths; rerun the 256K invariant and chunk sweep.
3. Separately isolate exponential approximation and TF32 ingress after those
   controls pass. Do not attribute all remaining error to matmul fidelity.

These are follow-up opportunities, not fixes applied in this experiment.

## Reproduce

From a built Blackhole checkout with its Python environment activated:

```sh
python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --variants hifi2 fp32_hifi2 --label destination-only --output normal.jsonl

python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --variants hifi2 fp32_hifi2 --distribution uniform_constant_v \
  --kv-lens 512 4096 32768 131072 262144 \
  --label destination-only --output uniform-constant-v.jsonl
```

Raw files: `normal.jsonl`, `seed1235.jsonl`, `causal-tail.jsonl`, `uniform.jsonl`,
`constant-v.jsonl`, `chunks.jsonl`, `uniform-constant-v.jsonl`, `single-chunk.jsonl`.
Each row records the complete tested shape, seed, flags, L2, PCC, gain, and row-error
statistics. The script can also enforce a chosen L2 limit with `--max-l2-pct`.
