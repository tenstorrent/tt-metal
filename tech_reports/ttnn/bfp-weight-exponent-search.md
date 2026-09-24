# BFP weight conversion: optional exponent search

This prototype can reduce weight conversion error for BFP4_B and BFP8_B.
The option is off by default.
It does not require calibration data or an inference pass.
It does not change device kernels, tensor shapes, or the number of bytes in a packed tensor.
Lower weight error does not guarantee better model accuracy.

## How it works

A BFP group contains 16 values that share one exponent.
The usual conversion selects the largest input exponent, called Emax.
The new option also tests Emax−1.

For each group, the CPU does these steps:

1. Convert the values with Emax.
2. Convert the values with Emax−1.
3. Calculate the sum of squared weight errors for each result.
4. Select the result with the smaller error.

If the errors are equal, the CPU selects Emax.
Emax−1 gives half the spacing between representable values.
It also reduces the largest representable magnitude.
The conversion limits larger magnitudes to this maximum.

The code uses the existing rounding rule: round to nearest, with ties to even.
It keeps the usual exponent for groups with infinity, NaN, or very small values.
It does not change the source weights.

The search occurs after the code divides the tensor between devices and adds padding.
Thus, it uses the same groups of 16 values that the device will read.
This method tests two exponents only.
It does not use GPTQ or the previous three-exponent experiment.

## Use the option with or without a cache

Build the Python package and native library from this branch together.
Follow the [build instructions](../../INSTALLING.md).
An older installed wheel does not support the new argument.

```python
import torch
import ttnn

weights = torch.randn(1024, 1024, dtype=torch.bfloat16)

# Convert without a cache.
optimized = ttnn.from_torch(
    weights,
    dtype=ttnn.bfloat4_b,  # bfloat8_b is also supported.
    layout=ttnn.TILE_LAYOUT,
    optimize_bfp=True,
)

# Convert and save the result for later loads.
cached = ttnn.as_tensor(
    weights,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    cache_file_name="/tmp/my-model-cache/layer0.wq",
    optimize_bfp=True,
)
```

A cache is a file that stores the converted tensor for later use.
The search does not require a cache.
`ttnn.as_tensor(..., optimize_bfp=True)` also works without `cache_file_name`.

The global setting has a narrower scope:

```bash
export TTNN_CONFIG_OVERRIDES='{"enable_bfp_weight_optimization":true}'
```

Set this variable before model construction and cache checks.
If the variable already contains other settings, add this key to the existing JSON object.
You can also set `ttnn.CONFIG.enable_bfp_weight_optimization = True` in Python.
Keep the setting constant until the model has loaded.

| Call | When the search runs |
| --- | --- |
| `from_torch(..., optimize_bfp=True)` | On each conversion, without a cache requirement. |
| `as_tensor(..., optimize_bfp=True)` | On conversion. If the correct cache file exists, the call reads that file. |
| `as_tensor(..., optimize_bfp=False)` | Never. This argument overrides the global setting. |
| `as_tensor(...)` with no explicit argument | For BFP conversion when the global setting is true and a cache filename is set. |
| `from_torch(...)` with no explicit argument | Never. The global setting does not apply. |

The global setting uses a cache filename as an indication that the tensor could contain weights.
The API cannot identify weights from their values or shape.
Applying the setting to every call could change activation conversion during inference.
The cache condition is an initial integration choice, not a requirement of exponent search.
Some cached tensors contain data other than weights.
Some weight loaders do not use this cache API.
Use the explicit argument when the caller knows that a tensor contains weights.

The option requires BFP4_B or BFP8_B and tile layout.
Do not combine it with `enable_bfloat_opt=True`.
That option permits a separate device conversion with different rounding.

The C++ functions `pack_as_bfp4_tiles` and `pack_as_bfp8_tiles` accept a final `optimize_bfp` argument after `tile`.
`tt::tt_metal::to_dtype` accepts the argument after the target dtype.
The default is false.
The search requires B-format (`is_exp_a=false`).
It does not apply to BFP2 or device conversion operations.

## Conversion time and memory

The normal weight conversion already runs on the CPU.
It uses a C++ packer that can process tiles in parallel.
The new search uses this same packer and parallel execution.
The disabled option does not select a special TT hardware conversion for cached weights.

The first prototype expanded every optimized input to FP32 before it built the tiled host tensor.
This was an implementation choice.
The search itself does not require a complete FP32 copy.
The revised code keeps BF16 inputs in BF16 when padding is zero and `col_tilize` is false.
The packer converts individual values as it processes each group.
This removes the full BF16-to-FP32 input conversion in that case.
It also reduces the temporary tiled buffer from four bytes to two bytes per value, before alignment overhead.

FP32 input, column tilization, and nonzero padding still use the FP32 path.
This preserves the conversion results from the first prototype.
The enabled path still builds a temporary tiled tensor before it packs the BFP data.
It also calculates two errors per group.
We have not measured the total time or peak memory on the target machine.
We cannot yet say whether the first model load will be considerably slower.

| Stage | Effect of the option |
| --- | --- |
| First conversion | Adds exponent search and temporary host storage. |
| Load from an existing optimized cache | Reads the converted tensor. No exponent search. |
| Model inference after loading | Uses the same dtype, packed size, and kernels. |

Further improvements are possible.
The packer could reuse the rounded values calculated during the search.
It could also combine tensor layout conversion and BFP packing to remove the full temporary tiled tensor.
These changes are not included yet.
They need separate checks for padding, device partitioning, and identical output bytes.

Measure first conversion, later cache loads, and inference separately.
Use the same weights, machine, thread count, and precision settings for each comparison.

## Cache files and model loaders

Optimized tensor filenames end in `_bfp_emax_minus1_v1.tensorbin`.
Ordinary conversion retains the existing filenames.
The source checkpoint must be available when the optimized cache does not exist.
The cache directory must be writable.
Do not rename ordinary cache files to use the optimized suffix.

Some model loaders use a completion marker to skip reading source weights.
These loaders supply uninitialized tensors when they expect every converted weight to exist in the cache.
The shared cache helper now uses a separate completion marker for optimized files.
Thus, an ordinary cache cannot cause a loader to save uninitialized values as optimized weights.

The automatic setting covers cached calls in `models/tt_transformers`, also called TTTv1.
The initial targets are Llama-3.1-8B and the Blackhole Qwen3-32B implementation.

`models/common/modules/lazy_weight.py`, used by TTTv2, has a different cache and conversion path.
This includes the Wormhole Qwen3-32B implementation checked for this prototype.
The global setting does not enable search there.
That loader needs an explicit conversion argument and separate cache filenames.

Other custom loaders need the same checks.
If a loader can skip source weights, its cache checks must account for the conversion option.
If a loader chooses the option separately for each weight, its cache checks must also record those choices.
Before an accuracy test, verify the effective setting and the optimized cache filenames.

## Tests

Run these tests in a built tt-metal environment:

```bash
build/test/tt_metal/unit_tests_legacy --gtest_filter='HostOnlyTest.Bfp*'
pytest -q tests/ttnn/unit_tests/base_functionality/test_bfp_weight_optimization.py
```

The C++ tests check packed bytes, decoded values, group errors, signs, layout, and special values.
They also compare direct BF16 input with the same values expanded to FP32.
The Python tests check padding, small tiles, cache selection, configuration, column tilization, and copies to devices.
A BF16 comparison test also checks nonzero padding and column tilization.
Device tests require TT hardware.

The initial version passed 24 local C++ cases and the repository checks.
Those cases matched the ordinary bytes from main and the earlier NumPy search results.
The local test program replaced hardware support functions with test substitutes.
Python cache checks also used test substitutes for native conversion and file storage.
These checks do not establish that the full TTNN library builds or runs correctly on a device.
The BF16 revision passed 16 local comparisons of direct BF16 input and the same values expanded to FP32.
The inputs included all 65,536 BF16 bit patterns and 1,048,576 random finite values.
Both BFP dtypes, both input layouts, and both option settings produced identical bytes.
The host dtype selection also passed 64 local checks.
These checks used source code compiled with test substitutes for platform functions.

A full Linux build, integration tests, and model measurements remain required.

## Accuracy tests after review

Compare the option enabled and disabled on the same commit.
Keep the checkpoint, hardware, precision, prompts, task versions, and generation settings fixed.
Save the sample counts and individual results.
A passing CI threshold alone does not show an improvement.

### First: tt-metal model tests

Use [Models Tier 2 E2E](../../.github/workflows/models-t2-e2e-tests.yaml).
This branch adds the `optimize-bfp-weights` input, with false as its default.
It applies to jobs that run on one host.
Scheduled runs keep it disabled.

Start with `llama3.1-8b` and `qwen3-32b` on `bh_quietbox_2 (BH QB2)`.
Use a writable cache directory for the first enabled run.
If necessary, set `mlperf-read-only=false` to permit writes to the mounted cache.
Prefer a cache directory on the runner when available.
Allow enough host memory and time for the first conversion.

The test selection is in [models_e2e_tests.yaml](../../tests/pipeline_reorg/models_e2e_tests.yaml).
`performance-ci-token-matching` checks up to 500 positions from one text continuation.
It supplies the reference text tokens as input at successive positions.

- `top1_token_accuracy`: the model's highest-scoring token equals the reference model's highest-scoring token.
- `top5_token_accuracy`: the model's highest-scoring token is among the reference model's five highest-scoring tokens.

The second metric does not check whether the reference top token appears in the quantized model's top five.
`ci-eval-32` checks repeated outputs and performance.
These tests can detect regressions, but they are too small to establish a general accuracy gain.

### Next: full task evaluations through Shield

Use [Shield On dispatch](https://github.com/tenstorrent/tt-shield/blob/main/.github/workflows/on-dispatch.yml)
with the [inference-server evaluation workflow](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/workflows_user_guide.md).
Set `workflow=evals` and `run-full-evals=true`.
Specify exact commits for tt-metal, inference-server, and vLLM.
The `benchmarks` workflow measures speed with random prompts.
It does not measure task accuracy.

The inspected task configuration includes IFEval, GPQA, and LongBench for Llama-3.1-8B-Instruct.
It includes AIME24, MATH500, and GPQA Diamond for Qwen3-32B.
Full AIME24 has only 30 questions.
Include larger tasks and report statistical uncertainty.
Some configured Qwen GPU scores are unverified estimates.
Use our measured baseline to assess the effect of this change.

Shield needs one more change before it can enable this option.
Its workflow must write the TTNN setting into the model server's Docker `.env` file.
Setting it only in the evaluation client has no effect on the server.
Selecting this branch alone also leaves the option off.
Verify the selected model loader and the server setting before each run.
This PR does not change Shield.

Alternatively, start the model server yourself with the setting enabled.
Then run the inference-server evaluation client against that server.

### Detailed comparison with the reference model

Keep the existing experiment evaluator for perplexity, KL divergence, and the additional top-five agreement metrics.
The standard workflows above do not supply all these metrics.
Use identical test prefixes and reuse the reference logits.
Use data that was not used to choose the method.
Report the number of tokens and documents, with uncertainty estimates from paired document samples.

Inspected commits: tt-metal `b99aa035f39`, inference-server `935360671a0a4a894a2c35ee483d665978f2d2c7`,
and Shield `8d3c3010aa9c66e004690c874634d0abba87a4cd`.
Check workflow inputs and model loaders again before you start CI.
Model CI remains deferred until review is complete.
