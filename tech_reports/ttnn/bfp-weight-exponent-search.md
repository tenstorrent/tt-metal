# Opt-in BFP weight exponent search (prototype)

This branch adds `optimize_bfp=True` to host conversion to `ttnn.bfloat4_b` and
`ttnn.bfloat8_b`. Existing calls default to ordinary packing. There is no
calibration, inference pass, model rewrite, GPTQ, or change to device kernels.
The dtype, packed size, tensor shape, and runtime arithmetic stay the same.

For every physical group of 16 values sharing an exponent, the packer:

1. Finds the usual maximum exponent, Emax.
2. Rounds with Emax and with Emax−1, using the existing round-to-nearest-even
   mantissa conversion. The lower exponent has half the step size and saturates
   outliers to its largest representable magnitude.
3. Compares the two sums of squared weight errors in double precision. It uses
   Emax−1 only when its error is strictly smaller; ties keep Emax.

Groups containing NaN/Inf, and groups close to exponent underflow, keep the
ordinary encoding. Source weights are untouched. This is the two-exponent
method from the experiments, **not** the three-exponent search or GPTQ+search.
It guarantees no increase in the local weight error for eligible groups, not an
increase in end-to-end model accuracy. Activation errors can add or cancel.

## Use from Python

Build/install this branch's TTNN and native library together using the normal
[tt-metal build instructions](../../INSTALLING.md). A Python-only checkout
change against an older wheel is insufficient.

```python
import torch
import ttnn

weights = torch.randn(1024, 1024, dtype=torch.bfloat16)
optimized = ttnn.from_torch(
    weights,
    dtype=ttnn.bfloat4_b,   # or ttnn.bfloat8_b
    layout=ttnn.TILE_LAYOUT,
    optimize_bfp=True,
)

cached = ttnn.as_tensor(
    weights,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    cache_file_name="/tmp/my-model-cache/layer0.wq",
    optimize_bfp=True,
)
```

`from_torch` always requires explicit opt-in. `as_tensor` accepts True, False,
or None (the default). None follows the configuration below **only for cached
BFP tensors**. Explicit False overrides the configuration. The flag is
incompatible with `enable_bfloat_opt=True`, which is a different, lossy device
conversion path. Unsupported dtypes are rejected for explicit True.

Packing searches the groups after mesh partitioning, padding, tilization and
optional `col_tilize`, including fused model weights in their final layout.
The prototype constructs an FP32 host tensor before packing when enabled, so
cold conversion uses additional temporary host memory. Warm cache loads avoid
that work. It does not reduce inference memory or imply a throughput speedup.

The C++ host packers `pack_as_bfp4_tiles` and `pack_as_bfp8_tiles` accept a final
`bool optimize_bfp=false` argument after `tile`; `tt::tt_metal::to_dtype` accepts
it after the target dtype. Only B-format is supported, not `is_exp_a=true`,
BFP2, device `typecast`, or device tilization.

## Existing model scripts

Set this **before model construction and cache-completeness checks**:

```bash
export TTNN_CONFIG_OVERRIDES='{"enable_bfp_weight_optimization":true}'
```

Or set `ttnn.CONFIG.enable_bfp_weight_optimization = True` in the process.
Merge this key into existing overrides if you already use that environment
variable. Keep the setting fixed for the entire model load.

This enables search for both BFP4 and BFP8 through cached `ttnn.as_tensor` calls,
without changing the model's precision configuration. It also affects any
other static BFP tensors cached through that API. It deliberately leaves
ordinary activation `from_torch` calls unchanged.

Optimized tensor filenames end in `_bfp_emax_minus1_v1.tensorbin`. Disabled
calls retain their original names. The shared model cache helper adds the same
algorithm discriminator to its `.weights_complete` marker, so an ordinary
warm-cache marker cannot authorize placeholder source weights for an optimized
cache miss. Original checkpoint weights and writable cache storage must be
available on the first enabled run. Do not rename existing tensorbins into the
new namespace. Use separate cache roots for controlled comparisons.

Coverage is a property of the loader, not just the model name:

- `models/tt_transformers` (TTTv1), including Llama-3.1-8B and the Blackhole
  Qwen3-32B CI path, uses cached `as_tensor` and is the initial target.
- `models/common/modules/lazy_weight.py` (TTTv2) uses its own cache and direct
  `from_torch` calls. The config switch does **not** enable it. This includes the
  current Wormhole Qwen3-32B E2E path. Its loader would need explicit opt-in and
  its own cache discriminator before using this experiment there.
- Other custom loaders, prepacked tensorbins, uncached `from_torch`, C++ model
  loaders and device conversions are not automatically enabled. Audit their
  call sites and cache identity first. A model-specific warm-cache shortcut
  outside `models/common/weight_cache.py` must not supply placeholders for new
  optimized files. Per-weight explicit overrides also belong in that loader's
  build-variant identity.

Check the effective config at startup and the new tensorbin filenames before
attributing a model result to this feature. Absence of those files can mean a
loader was never covered.

## Validation and review

Added tests:

```bash
build/test/tt_metal/unit_tests_legacy --gtest_filter='HostOnlyTest.Bfp*'
pytest -q tests/ttnn/unit_tests/base_functionality/test_bfp_weight_optimization.py
```

The C++ tests check real pack/unpack values, local error, signs, exponent bytes,
layout order and special-value fallback. Python tests cover BF16/FP32 inputs,
padding, small tiles, TensorSpec, cache separation/reuse, configuration scope,
marker transitions, invalid combinations, column tilization, device copies
and mesh sharding. Device/mesh tests require appropriate hardware fixtures.

Before review, the actual native packer body was compiled locally with platform
services stubbed (HAL alignment, executor and logging). Twenty-four cases
matched main's default packed bytes and the earlier NumPy exponent-search
oracle; every checked physical group had non-increasing squared error. This
is **not** a full TTNN integration build. The Python cache policy and marker
transitions also passed local checks with native conversion/serialization mocked. Native-library linking, Python
bindings and device tests still need the Linux/TT validation above. No model
accuracy result is claimed for this branch yet.

## CI plan — launch after review

Run **both modes on this same commit**, same checkpoint revision, hardware,
precision profile, prompts, task versions and sampling settings. Record exact
sample counts and per-example results, not only a pass/fail threshold.

### 1. tt-metal: conversion and model regression checks

This branch adds a default-off `optimize-bfp-weights` input to
[Models Tier 2 E2E](../../.github/workflows/models-t2-e2e-tests.yaml).
It forwards the config to single-host model jobs. Scheduled jobs stay off.
After review, dispatch baseline and optimized runs for:

- Model `llama3.1-8b`, SKU `bh_quietbox_2 (BH QB2)`.
- Model `qwen3-32b`, same SKU.

Use `mlperf-read-only=false` where a new cache must be written; enabled runs
must not attempt to populate a read-only shared cache. The optimized suffix
preserves ordinary files, but this grants the job write access to the mounted
cache. Prefer runner-local writable caches when available. Cold-build memory,
cache population and timeout headroom should be checked before a broad sweep.
Do not select all models/SKUs initially: coverage varies by loader, and the
switch is wired only to single-host jobs in this prototype.

The matrix is in
[models_e2e_tests.yaml](../../tests/pipeline_reorg/models_e2e_tests.yaml).
`performance-ci-token-matching` measures teacher-forced agreement with stored
reference predictions, over up to 500 steps of one continuation. Its `top1`
is predicted-token agreement; its `top5` is **quantized top1 in reference top5**,
not reference top1 in quantized top5 or top5-set overlap. Compare the numerical
`top1_token_accuracy` and `top5_token_accuracy` artifacts. `ci-eval-32` also
checks repeatability/performance; its name does not make it a broad task
accuracy or perplexity benchmark. These checks are useful regression gates
but too narrow to establish a small general accuracy gain.

### 2. Shield + inference-server: full task evaluations

[tt-shield On dispatch](https://github.com/tenstorrent/tt-shield/blob/main/.github/workflows/on-dispatch.yml)
orchestrates
[tt-inference-server evals](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/workflows_user_guide.md).
Choose `workflow=evals`, `run-full-evals=true`, an exact `tt-metal-git-ref`, and
pin the inference-server/vLLM refs as well. `benchmarks` measures random-prompt
latency/throughput; it is not the task-accuracy suite. `release` includes both.

Current
[eval configuration](https://github.com/tenstorrent/tt-inference-server/blob/main/reference_config/evals/eval_config.py)
includes Llama-3.1-8B-Instruct `meta_ifeval`, `meta_gpqa_cot` and LongBench tasks;
Qwen3-32B uses `r1_aime24`, `r1_math500` and `r1_gpqa_diamond`.
Full AIME is still only 30 questions: a one-question change is 3.33 percentage
points. Include the larger tasks, keep generations paired/seeded, and report
uncertainty. Some configured Qwen GPU reference scores are explicitly marked
unvalidated estimates; compare our paired baseline/optimized outputs rather
than treating those numbers as a controlled reference.

**One additional Shield wiring change is needed before dispatching enabled
runs.** Selecting this branch alone leaves the feature off. Its reusable
inference-server workflow writes a Docker `.env` containing selected fields;
it currently has no input for this new TTNN setting. Add a default-off input
and write `TTNN_CONFIG_OVERRIDES={"enable_bfp_weight_optimization":true}` into
that server `.env` when selected (merge any other TTNN keys). Validate the
server's effective config and optimized cache files. Setting the variable only
in the evaluation client is insufficient. Select the covered `tt-transformers`
implementation and verify the resolved QB2 device/model configuration before
launching. No Shield repository or workflow was changed by this prototype.

An alternative is to start a server built from this branch with the setting in
its environment, then run the inference-server `evals` client against that
server. This avoids needing a Shield change for an initial manual experiment.

### 3. Sensitive paired metrics from the earlier experiment

Neither of these standard workflows supplies our held-out perplexity,
full-distribution KL divergence, reference-top1-in-quantized-top5 recall, or
reference/quantized top5 overlap. Retain the existing experiment evaluator for
those measurements. Use identical held-out prefixes and at least the previous
large evaluation sample, reused reference logits, and both flag settings at the
same precision. Report token/document counts and paired document-bootstrap
intervals. Exponent search has no calibration split to leak into evaluation,
but a dataset used to choose methods should not serve as the final test set.

Measure cold weight load/conversion separately from warm model startup and
steady-state inference. A lower local weight error, a higher reference
agreement, and lower perplexity need not move together.

CI inspection pins: tt-metal base `b99aa035f39`, inference-server
`935360671a0a4a894a2c35ee483d665978f2d2c7`, Shield
`8d3c3010aa9c66e004690c874634d0abba87a4cd`. Workflow definitions may change;
recheck inputs and loader coverage at dispatch time. No CI was launched while
preparing this prototype.
