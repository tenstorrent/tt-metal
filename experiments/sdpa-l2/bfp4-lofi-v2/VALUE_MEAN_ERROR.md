# Uncentered V quantization with device mean-error correction

`value_mean_error_fullchip.py` is an isolated wrapper around the qualified
`value_centered_fullchip.py` / `value_centered_b8_fullchip.py` builders, always
invoked with `center_mode=none`. It changes no attention kernel or circular
buffer. Q256/K512/D128, two K/V slots, per-head forwarding chains, and original
BF16 Q/K/V references are preserved. The 16 parent-run long-context cases
below pass their recorded device/replay checks; this is a scoped experiment,
not a universal accuracy qualification or production implementation.

Supported points are K8/V4 and K8/V8, BF16 MAIN or BF16 FAST with full or
denominator-only compensation. FP32 is intentionally rejected: the reused
builders explicitly guard against it. The inherited exponential path is
unchanged; this wrapper does not select a new native/LUT-exp implementation.

## Mathematical scope

Let `W` be exact row-normalized attention weights, `Vq` the values actually
consumed by LoFi, and `mu` the mean over the KV sequence:

```
Vq = consume(quantize(V))
delta = mu(V) - mu(Vq)
Y = W Vq + delta
Y - W V = (W - uniform_weights) (Vq - V)
```

Thus the correction removes the unweighted/DC component of quantization
error. It is not an exact correction for attention-weighted quantization
error, QK error, softmax error, or BF16 recurrence drift. Constant-V failure
caused by recurrence cannot generally be repaired: if quantization represents
that constant exactly, `delta=0` despite a potentially large operator error.

Existing matched centering instead computes
`Vc=consume(quantize(V-mu(V)))`, then `W Vc + mu(V)-mu(Vc)`.
The new scheme gives the quantizer ORIGINAL V and avoids changing its input
distribution. These are equal only under appropriate translation-equivariant
quantization; shared-exponent quantization is not generally equivariant.
The existing fused center preprocessor already retains the subtraction in
FP32 until quantization, so do not claim this wrapper removes a full-sized
BF16 precentering spill that the existing implementation does not make.

## Actual device precision and cost

Both means are computed on device through `center_mean.build`. Default
`bf16_fpu` uses HiFi4/FP32 DST but returns BF16 means; its reciprocal is BF16
truncated. `fp32_sfpu` includes full-size BF16-to-FP32 input conversion and
FP32 SFPU mean, but the returned bias still narrows to BF16. The delta is
BF16 RNE of the two actual BF16 means. A large common offset can therefore
erase a small delta even in `fp32_sfpu` mode. `CORRECTION_CHECK` reports the
actual delta error against FP64 means of original and consumed V, without
silently replacing the device bias with a host result.

V4 is decoded to BF16 and is exact in LoFi's five significant right-operand
bits. V8 is decoded and passed through the qualified trunc5 primitive; this
is truncation toward zero, NOT RNE5. Taking the mean of decoded V8 without
this step would correct the wrong values.

The attention core writes BF16, then a broadcast `ttnn.add` writes a separate
BF16 final output. This adds an intermediate output rounding compared with a
fused high-precision epilogue. The standalone epilogue is checked against
BF16 RNE of the actual core output and actual delta.

Combined trace timing includes Q/K/V quantization, both device means, packed-V
decode, V8-only effective truncation, small delta subtraction/slice, attention,
and the full-output add. No additional attention input slots or chunk changes
are hidden in the comparison. All allocation/callable owners remain alive
through trace replay. Host FP64 diagnostics are outside device timing and are
never uploaded as biases.

## Measured 32K / 256K full-FAST results

These are **device results**, not codec models: H10, D128, 110 active cores,
Q256/K512, two K/V slots, full numerator and denominator compensation,
LoFi, seed 1240, BF16-FPU device means, no V precentering. Accuracy uses all
heads/all KV and 128 explicitly sampled Q rows per head, not all query rows.
Each median uses three timed replays after two warmups, one invocation per
replay. Mandatory correctness replays are separate.

`normal` is BF16 normal Q/K/V. **`uniform` means Q=0**, with the same normal
K/V as the corresponding normal case; it produces exactly uniform attention
weights. It is not a uniform random input distribution. The exact output is
the original V mean, so its centered reference residual is zero; the records
correctly leave centered relative L2 undefined and report absolute error.

The two linked L2 columns identify all 16 underlying records. TFLOP/s is
combined preprocessing + attention + optional correction/epilogue throughput.

| N | K/V | Input | L2 %, no correction | L2 %, mean-error | PCC, no correction → mean-error | Combined TFLOP/s, no correction → mean-error |
|---:|---|---|---:|---:|---:|---:|
| 32768 | B8/B4 | normal | [12.0074](mean-error-32768-b8_b4-normal-none-v1.json) | [9.8072](mean-error-32768-b8_b4-normal-mean_error-v1.json) | 0.99281594 → 0.99524210 | 191.09 → 178.77 |
| 32768 | B8/B4 | uniform | [11.6704](mean-error-32768-b8_b4-uniform-none-v1.json) | [1.7160](mean-error-32768-b8_b4-uniform-mean_error-v1.json) | 0.99318283 → 0.99996795 | 191.14 → 178.87 |
| 32768 | B8/B8 | normal | [3.1511](mean-error-32768-b8_b8-normal-none-v1.json) | [3.0385](mean-error-32768-b8_b8-normal-mean_error-v1.json) | 0.99953343 → 0.99956734 | 190.11 → 175.73 |
| 32768 | B8/B8 | uniform | [1.7302](mean-error-32768-b8_b8-uniform-none-v1.json) | [1.0420](mean-error-32768-b8_b8-uniform-mean_error-v1.json) | 0.99987468 → 0.99997043 | 190.22 → 175.91 |
| 262144 | B8/B4 | normal | [12.2926](mean-error-262144-b8_b4-normal-none-v1.json) | [10.0637](mean-error-262144-b8_b4-normal-mean_error-v1.json) | 0.99275296 → 0.99528085 | 195.18 → 193.82 |
| 262144 | B8/B4 | uniform | [12.3808](mean-error-262144-b8_b4-uniform-none-v1.json) | [4.2089](mean-error-262144-b8_b4-uniform-mean_error-v1.json) | 0.99301308 → 0.99995112 | 202.49 → 200.66 |
| 262144 | B8/B8 | normal | [3.7350](mean-error-262144-b8_b8-normal-none-v1.json) | [3.6200](mean-error-262144-b8_b8-normal-mean_error-v1.json) | 0.99953761 → 0.99957178 | 190.39 → 189.58 |
| 262144 | B8/B8 | uniform | [2.4726](mean-error-262144-b8_b8-uniform-none-v1.json) | [1.9723](mean-error-262144-b8_b8-uniform-mean_error-v1.json) | 0.99986542 → 0.99996815 | 201.57 → 199.95 |

The correction consistently helps these eight paired cases, most strongly
for uniform attention with V4. Normal V4 remains near 10% and normal V8 near
3–3.6%; this does not establish a generally accurate low-precision operator.
Do not extend these findings to common offsets, outliers, captured activations
or other seeds without corresponding measurements.

### Is the remaining error caused by estimating the correction?

The recorded check compares the **actual BF16 device delta** against
`FP64 mean(original V) - FP64 mean(actual LoFi-consumed Vq)`. Thus it includes
both mean reductions, their BF16 narrowing, and the BF16 delta subtraction.
Normal/uniform share V exactly at each N/format, so their correction diagnostics
are identical even though their attention outputs differ.

| N | V | Ideal delta RMS | Device delta error RMS vs FP64 | Equivalent L2 percentage points, normal / uniform |
|---:|---|---:|---:|---:|
| 32768 | B4 | 6.39962e-4 | 1.34783e-5 | 0.1473 / 0.2461 |
| 32768 | B8 | 7.77331e-5 | 1.32148e-5 | 0.1444 / 0.2413 |
| 262144 | B4 | 2.32027e-4 | 4.81802e-6 | 0.1487 / 0.2452 |
| 262144 | B8 | 2.81190e-5 | 4.49607e-6 | 0.1388 / 0.2288 |

The final column is `100 * delta_error_RMS / original_reference_RMS`, not
an additive decomposition of L2. The bias error is about 2.1% of the ideal
V4 correction and 16–17% of the much smaller V8 correction. It is only
1.5% of the remaining normal V4 output-error RMS and 3.8–4.8% for normal V8.
For uniform attention it is 14.3%/23.2% of remaining error RMS at 32K
(V4/V8), falling to 5.8%/11.6% at 256K. A better mean is relevant, especially
for V8, but cannot by itself explain most of the residual error.

Both N values are powers of two: the mean's BF16 reciprocal `1/N` is exactly
representable here. Its documented reciprocal-truncation limitation therefore
does **not** explain these particular delta errors; accumulation and BF16
mean/delta roundings remain relevant.

Uniform attention is the clearest diagnostic. In ideal arithmetic the DC
correction exactly removes V quantization error when Q=0. The measured
corrected absolute output-error RMS is nevertheless:

| N | V4 | V8 |
|---:|---:|---:|
| 32768 | 9.39973e-5 | 5.70742e-5 |
| 262144 | 8.27165e-5 | 3.87606e-5 |

Absolute error decreases less rapidly than the reference mean RMS as N grows,
so relative L2 rises. This points to attention reduction/recurrence/final
normalization and output rounding beyond the measured bias-estimation error.
It does not individually identify which of those dominates. Corrected uniform
gain remains about 0.9963–0.9968, so final reciprocal precision is a reasonable
separate control, not a proven explanation. The BF16 epilogue matches its RNE
oracle exactly in all corrected records, but that confirms implementation,
not absence of epilogue rounding error. The aggregate norms do not provide
the error-vector correlations needed for an exact causal decomposition.

### Cost versus sequence length

| N | V | Paired combined-time increase, normal / uniform | Extra measured prep + epilogue, normal / uniform |
|---:|---|---:|---:|
| 32768 | B4 | 6.889% / 6.858% | 1.988 / 1.979 ms |
| 32768 | B8 | 8.185% / 8.137% | 2.463 / 2.423 ms |
| 262144 | B4 | 0.704% / 0.911% | 15.724 / 15.585 ms |
| 262144 | B8 | 0.424% / 0.807% | 19.469 / 19.208 ms |

Here extra stages mean `corrected preprocessing - control preprocessing +
corrected epilogue`, measured as separate stage traces. They scale roughly
linearly with N, whereas useful square-attention work scales quadratically.
The epilogue alone grows from about 0.67 ms to 5.20 ms. V8 costs more because
its actual-consumed-value mean also requires the full-sized trunc5 pass.

Do not treat the sub-percent 256K paired differences as clean causal overhead
estimates: the normal attention-only medians differ despite an unchanged
attention kernel, e.g. V8 1808.272 ms control versus 1800.790 ms corrected.
Separate stage costs suggest roughly 0.9–1.1% of the control's 256K combined
time, while measured paired combined deltas range 0.42–0.91%. Three timed
replays and a single seed are insufficient to resolve such small differences
robustly. Report measured TFLOP/s as above, without clock renormalization;
the later 1306 MHz active telemetry snapshot does not establish every run's
clock, and configured max1350 is not proof of sustained1350.

All 16 records report finite outputs, zero input-preprocessing mismatches,
two successful bitwise combined correctness replays, and unchanged original
CPU/device inputs. All eight corrections have exact delta-subtract and
epilogue-oracle checks. Their 50 recorded source hashes match current local
files at this review. This is not a complete source-closure claim: the deferred
[manifest plan](SOURCE_MANIFEST_PATCH_PLAN.md) documents known missing shared
dependencies, and historical records are not retrospectively repinned.

## Parent-run commands

Local standard-library control-flow tests (no Torch, TTNN or device):

```bash
python3 experiments/sdpa-l2/bfp4-lofi-v2/test_value_mean_error.py
```

The five tests check two unconditional combined replays before timing,
bitwise-output-hash validation, CPU/device input mutation rejection, and
trace release after a replay failure. They do not qualify hardware or numerics.
The driver now performs those two correctness replays even with `--iters 0`,
and hashes CPU/device original BF16 inputs before/after replay and after
timing. Records include explicit replay counts, per-replay output hashes,
and immutability booleans; correctness overhead is excluded from timings.

Run from the configured tt-metal worktree; no device jobs were launched by
the implementing agent:

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/value_mean_error_fullchip.py \
  --label mean-error-b4-normal-smoke-v1 --kv-formats b8_b4 \
  --destination fast_bf16 --length 1024 --heads 2 --cores 4 \
  --sample-rows 1024 --check-preprocess --iters 0

python experiments/sdpa-l2/bfp4-lofi-v2/value_mean_error_fullchip.py \
  --label mean-error-b8-commonv-smoke-v1 --kv-formats b8_b8 \
  --destination fast_bf16 --denom-only --distribution common_v --common-mode 32 \
  --length 1024 --heads 2 --cores 4 --sample-rows 1024 --check-preprocess --iters 0
```

For exact unchanged controls use `--correction-mode none` and a fresh label.
Repeat both formats for normal/common_v/constant_v, full and denominator-only,
before timing. `--mean-mode fp32_sfpu` diagnoses reduction sensitivity but does
not remove the BF16-bias limit. After qualification, use N32768/H10/C110,
`--iters 10 --warmup 5 --trace-repeats 1`; accuracy still samples the explicitly
listed query rows. Include centered-output/residual error, not only global
L2, for common-V inputs. Constant V reports absolute residual errors, not a
percentage against a zero reference residual.

## Independently checked resident evidence

The six `grid7-exact-resident-{main,full,denom}-k{2,512}-v1.jsonl` records each
contain normal and constant-V results. All 36 Q/K/V preprocessing checks have
zero mismatches. These are **native-exp controls** (`grid7_exp=false`) with
`q_repeats=2`; they do not retrospectively qualify old grid7=true records.

| Repeated KV chunks | Variant | Normal L2 % | Constant-V L2 % |
|---|---|---:|---:|
| 2 | MAIN / full / denominator-only, identical output hashes | 3.109215 | 0.681410 |
| 512 | MAIN | 20.308683 | 19.378738 |
| 512 | Full compensation, identical to its 2-chunk output | 3.109215 | 0.681410 |
| 512 | Denominator-only | 31.968307 | 32.660806 |

These inputs repeat the same KV512 block, so ideal attention is invariant to
the repeat count. This evidence resolves the prior unverified-preprocessing
caveat for the native control and strongly isolates recurrence drift. It is
not a distinct-token long-context accuracy result and is not a result from
the new correction wrapper.
