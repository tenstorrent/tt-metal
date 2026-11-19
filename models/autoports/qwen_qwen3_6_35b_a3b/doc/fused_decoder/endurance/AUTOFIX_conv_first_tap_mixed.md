# First convolution tap: mixed-input SFPU control

The snapshot and probe were prepared without device execution or production edits by this investigation agent. Root-owned isolated device results are recorded below; full-decoder selection remains pending.

## Exact change and native contract

The [baseline snapshot](conv_first_tap_baseline.py.snapshot), SHA256 `61ac211a397a4ab24737581b931ea318a4ed83d2e32d8eff67a4f26acda0a178`, freezes the current decoder. The [candidate snapshot](conv_first_tap_mixed.py.snapshot), SHA256 `a795e9b9dd94f5cb76b3973ab70167aa516a05e526fa4bf50c8f4694db2f5161`, changes only the first tap operation:

```python
accum = ttnn.multiply(
    history[:, :s, :], self.taps_fp32[0],
    dtype=ttnn.float32, fast_and_approximate_mode=False,
)
```

The original first operation explicitly casts the BF16 history slice to FP32 before multiplying. The candidate preserves the same history slice, FP32 tap, FP32 output, all later accumulations, convolution rounding/SiLU chain, and three-row carry update. Explicit `dtype` is required because the default follows the first input. Explicit `fast_and_approximate_mode=False` selects SFPU for mixed BF16/FP32 MUL (`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:51–65`). The program factory enables FP32 destination when either input or the output is FP32 (`binary_ng_program_factory.cpp:1134–1145`) and requests FP32 unpack for both input circular buffers (`:1371–1381`). This is operand widening during unpack without an input activation; the earlier BF16 Z/SiLU activation packing issue does not apply. Subnormal/signed-zero behavior remains a device diagnostic question, not a source-only guarantee.

The **other three casts cannot be removed while retaining existing addcmul**: `ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp:348–353` requires `input_b.dtype() == input_a.dtype() && input_c.dtype() == input_a.dtype()` before its fallback. The Python addcmul binding exposes `value`, `memory_config`, and `output_tensor`, but no output `dtype` override (`ternary_nanobind.cpp:170–227`). An FP32 accumulator and tap with a BF16 history slice fails input validation regardless of the output tensor. Replacing each addcmul with separate mixed multiply and add saves no dispatches and can change fused arithmetic, so it is excluded from this control.

## Product discriminator

[conv_first_tap_mixed_probe.py](../../../tests/conv_first_tap_mixed_probe.py) loads only the layer 0 checkpoint convolution weights. It selects eight representative distinct first-tap values across their ordered range and enumerates **every one of the 65,280 finite BF16 input codes** for each. It excludes the 256 infinity/NaN input codes explicitly; finite inputs that produce nonfinite products are retained and reported. Every FP32 output code, including signed zero, is compared between baseline and candidate; independent CPU-reference comparisons are reported separately with finite/nonfinite counts. No tolerance hides tiny differences.

Additional deterministic synthetic BF16 histories use actual `[1,S+3,8192]` dimensions at S1, S33 and S128 with all 8,192 real first-tap weights. These histories are synthetic shape controls, not recorded model activations. Timings include the history slice, original cast when applicable, and multiply, with 5 warmups, 30 samples and 20 traced replays per sample. They exclude the other three taps and are explicitly not full-convolution or decoder timings. Shape cases with code differences are not timed. The report retains all exhaustive differences even if a shape case matches. Input hashes, checkpoint tap hash, both snapshot hashes, source hash and standard device provenance are recorded. Existing output is protected by default.

CPU-only setup passed all eleven cases. The new test was formatted with Black 23.10.1 and isort 5.13.2. CPU setup verifies enumeration, shapes and local comparison plumbing, not installed device arithmetic.

Root-owned device command, through the normal exclusive-device wrapper:

```bash
timeout 600 env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe
```

Default output: `endurance/conv_first_tap_mixed_probe.json` plus provenance. The standalone probe opens only physical chip 0. `--oracle-only --output /tmp/conv_first_tap_mixed_cpu.json` performs setup without importing TTNN or opening a device.

## Complete decoder benchmark

Use both immutable snapshots to isolate the first cast from any subsequent production changes. Each invocation runs the standard complete decoder harness in a fresh process, including all four convolution taps, activation boundaries, carry updates and the rest of the decoder. The benchmark records the measured snapshot hash and does not select a production candidate.

```bash
env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/conv_first_tap_baseline.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/conv_first_tap_baseline_s256.json

env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/conv_first_tap_mixed.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/conv_first_tap_mixed_s256.json
```

A faster short result still requires the existing seeded endurance, continuation, determinism and final fresh-reference gates. Exhaustive synthetic differences alone must be interpreted by their actual codes and range; this diagnostic does not waive model-level validation or automatically classify a subnormal-only difference as a model PCC failure.

## Root-owned isolated device result

[conv_first_tap_mixed_probe.json](conv_first_tap_mixed_probe.json), with [provenance](conv_first_tap_mixed_probe.json.provenance.json), reports **`device_exact=true`**. Baseline and candidate agree on every FP32 code for all eight exhaustive finite-BF16/real-tap cases and all three full-width shape cases. First and last replay outputs also agree exactly. This is equality to the existing device path, not a claim of equality to CPU for every extreme input: both device paths retain identical CPU-reference differences (including 3, 102 and 367 code differences at S1/S33/S128, maximum finite absolute differences approximately 1.09e-38/1.15e-38/1.17e-38). Those differences are visible in the report and are not new candidate errors.

Complete first-tap **slice + cast when present + multiply** trace medians, 30 samples with 20 replays each:

| Shape | Original (ms) | Mixed (ms) | Saving (ms) |
|---|---:|---:|---:|
|S1|0.033395|0.020887|0.012507|
|S33|0.043859|0.026383|0.017476|
|S128|0.060905|0.039492|0.021414|

The isolated adapter is faster at every tested shape. These measurements exclude the remaining convolution taps and decoder, so they support advancing to the complete decoder comparison without establishing the final speedup. The full-path snapshot benchmark subsequently completed; see below. Final acceptance remains pending the unchanged seeded endurance/continuation/determinism gates; production adoption and final closure are not claimed here.

## Full-decoder result: measured improvement, acceptance pending

The root-owned [matched baseline](conv_first_tap_baseline_s256.json) and [mixed-first-tap candidate](conv_first_tap_mixed_s256.json) use layer0, S256/context512, 10 warmed prefill samples and 60 warmed decode samples. Both report deterministic replay, a passing runtime-fallback audit and timing with allocation tracking disabled.

| Complete decoder | Prefill (ms) | Decode (ms) |
|---|---:|---:|
|Baseline|154.292234|1.458949642|
|Mixed first tap|153.988650|1.451479970|

The measured decode improvement is **0.007470 ms (0.512%)**; prefill is **0.303585 ms (0.197%)** lower in this matched run. Prefill PCC0.999962395060449, prefill contribution PCC0.9999594863488576 and eager/traced decode PCC0.9999779064754709 are identical across the controls. This establishes a measured full-path benefit while retaining this harness's numerical results, rather than extrapolating the isolated product saving.

Advance the first-tap change as a candidate for the existing endurance gates; **do not mark it finally accepted from this short benchmark**. Root is separately benchmarking a combined candidate with scalar fusions and rank5 recurrence after its recurrence primitive check passed. That combined candidate requires its own measured-source provenance and validation; none of its eventual results are attributed to this isolated snapshot here.
