# AutoFix report: BFP8 activation/residual cache-update boundary

## Starting evidence

- Fresh inspection report: `AUTODEBUG.md` in this directory.
- Original failure: `doc/datatype_sweep/smokes/group_a.log`, where the
  `residual_activation_bfp8` policy reaches bounded-sliding prefill's
  `paged_update_cache` with packed K/V and the runtime rejects anything other
  than FLOAT32/BFLOAT16.
- The TTNN paged update and paged fused update validators both require a
  FLOAT32/BFLOAT16 update input while independently allowing low-precision
  cache storage. The update kernel owns the final repack into the cache dtype.

## Hypothesis experiment

- Hypothesis: BFP8 residual storage is valid, but K/V must be widened only at
  the paged-update boundary; otherwise non-aligned sliding prefill fails first
  and both decode update branches remain latent failures.
- Experiment: trace the selected multichip caller through bounded prefill,
  full/sliding decode, and the C++ validators; extract the conversion into a
  device-independent helper; test it with dtype-bearing fake tensors and
  source-order assertions.
- Result: verified. The failing prefill helper claimed BF16 token updates but
  did not cast them. Both decode branches likewise submitted native K/V.
- Fix:
  - `OptimizedDecoder._prepare_cache_update_input` preserves BF16 inputs and
    converts packed inputs to the fixed BF16 update format.
  - Bounded-prefill tail K/V are converted once before token slicing/sharding.
  - Optimized and multichip decode K/V are converted before the branch shared
    by paged fused update and paged update.
  - BFP8 residuals, ordinary bulk prefill cache-dtype fills, and the configured
    KV-cache allocation remain unchanged.
  - Runtime policy evidence reports the derived `cache_update_input_dtype`.
- Verification:

  ```text
  env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig pytest -q \
    models/autoports/google_gemma_4_31b/tests/test_cache_update_dtype_contract.py \
    models/autoports/google_gemma_4_31b/tests/test_precision_config.py \
    models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py
  ```

  Result: 46 passed, with three unrelated import/deprecation warnings.

## Second hypothesis experiment: decode QKV head-split input

- New evidence: after the first repair, the serialized hardware rerun
  `doc/datatype_sweep/smokes/group_a_autofix.log` passed bounded-prefill cache
  updates and then failed during decode prewarm in
  `nlp_create_qkv_heads_decode` with `Unsupported data format`.
- Hypothesis: decode QKV matmul inherits the BFP8 residual input dtype, while
  the head-split kernel accepts only FLOAT32/BFLOAT16 tile input.
- Experiment: compare the logged stack with the QKV matmul default-output
  rule and the head-split C++ validator; force the producer output dtype at the
  earliest shared packed/split projection boundary; assert source ordering in
  both optimized and multichip decode paths.
- Result: verified. Both QKV loops omitted `dtype`, so matmul defaulted to the
  BFP8 activation. The split validator independently requires BF16/FP32.
- Fix: both decode QKV matmuls now request their fixed BF16 split-input format
  directly. This avoids an extra typecast/copy, leaves the retained residual
  BFP8, and makes the later decode cache-input normalization an identity.
  Runtime policy evidence reports `qkv_split_input_dtype`.
- Verification: the same focused command above now reports 47 passed, with
  the same three unrelated warnings.

## Third hypothesis experiment: BFP8 packed-MLP L1 geometry

- New evidence: `doc/datatype_sweep/smokes/group_f.log` fails the
  `mlp_bfp8_lofi` candidate during decode prewarm at the packed gate/up
  `ttnn.linear`. The program reports 1,937,280 bytes of static circular buffers
  against Blackhole's 1,572,864-byte L1.
- Hypothesis: the passing BFP4 program's 14-core, K-block-width-12 geometry is
  not reusable for the BFP8 packed weight because the DRAM-sharded matmul
  triple-buffers its weight block.
- Experiment: instantiate the production TP4 geometry, reproduce the factory's
  weight-CB formula, enumerate exact divisors of the 12 K tiles/core, and make
  only the packed gate/up width depend on weight dtype.
- Result: verified. Packed gate/up has 336 N tiles, or 42 per DRAM reader. At
  BFP8's aligned 1,088-byte tile size, width 12 requires
  `42 * 12 * 3 * 1088 = 1,645,056` bytes for the weight CB alone, already over
  physical L1. Adding the BF16 input/intermediate, BFP8 output, and 111,360-byte
  unreserved base exactly reproduces 1,937,280 bytes. Width 6 is the largest
  smaller divisor of 12 and predicts 1,090,176 bytes, leaving 482,688 bytes.
- Fix: `_TPOptimizedSharedMLP.packed_gate_up_in0_block_w` caps BFP8 packed
  gate/up at width 6 dynamically. BFP4 remains width 12; separate gate/up and
  down programs retain their policy widths. Runtime evidence reports the
  effective geometry.
- Verification: the focused suite including `test_mlp_dtype_geometry.py`
  reports 49 passed with the same three unrelated warnings.

## Fourth hypothesis experiment: precision-policy consumption audit

- New evidence: `OptimizedDecoder.from_state_dict` loaded every decode
  attention tensor from the legacy shared `attention_weight_dtype`, despite
  the policy exposing separate resolved QKV and output dtypes. It also built
  and consumed one shared attention compute config instead of the separate
  resolved fidelities.
- Hypothesis: the single-device optimized loader violates the shared precision
  policy, while the measured TP4 path may or may not share the bug.
- Experiment: trace all six weight groups, five compute fidelities, layer
  exceptions, activation/residual, phase-specific CCL, KV cache, logits, and
  sampling fields through both loaders and the same-weight runner using AST and
  source-only tests. Do not import TTNN while the hardware runner is active.
- Result:
  - verified for the single-device optimized loader;
  - refuted for the measured TP4 attention path, which already loads decode
    QKV/output with the separate resolved dtypes and builds separate compute
    configs while intentionally retaining `attention_prefill` for both
    prefill projections;
  - one additional non-measured API discrepancy was found: dynamically created
    non-default-batch eager samplers hard-coded FP32 gather values.
- Fix:
  - single-device decode QKV/split tensors load from
    `resolved_attention_qkv_weight_dtype`, decode output loads from
    `resolved_attention_o_weight_dtype`, and their matmuls consume distinct
    resolved compute configurations;
  - eager samplers use `model.config.sampling_dtype`;
  - TP4 runtime summaries add actual constructed tensor dtypes for prefill and
    decode attention, prefill and decode MLP, and LM-head shards rather than
    relying only on policy names.
- Verification:

  ```text
  python -m py_compile \
    models/autoports/google_gemma_4_31b/tt/optimized_decoder.py \
    models/autoports/google_gemma_4_31b/tt/model.py \
    models/autoports/google_gemma_4_31b/tt/generator.py \
    models/autoports/google_gemma_4_31b/tests/test_precision_loader_source_contract.py
  python models/autoports/google_gemma_4_31b/tests/test_precision_loader_source_contract.py
  ```

  Result: seven source-only tests passed without importing TTNN.

- Rerun impact: no existing Stage 08 TP4 accuracy/performance row is invalidated
  by this loader bug. Any single-device optimized evidence for the following
  policies would require rerun: `attention_bfp8_hifi2`,
  `attention_qkv_bfp4_lofi`, `attention_qkv_bfp4_hifi2`,
  `attention_output_bfp4_lofi`, `attention_output_bfp4_hifi2`, and
  `canonical_accuracy_bfp8_hifi2_bf16commcache`. No such row is used by this
  TP4 sweep.

## Static regression repair: eager-sampler fixture contract

The sampling-policy fix correctly requires `model.config.sampling_dtype`, but
the existing isolated eager-sampler test used a dummy model without `config`.
The fixture now supplies a non-default BF16 sampling dtype and asserts the
constructed eager sampler receives that exact value. This both repairs the
fixture contract and proves the production path is not silently fixed to FP32.
Focused non-device verification:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig \
  pytest -q \
  models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py::test_eager_sampler_is_keyed_to_actual_batch_without_changing_canonical_trace_sampler
```

Result: `1 passed, 3 warnings in 1.86s`. The warnings are pre-existing SWIG
deprecation warnings and the repository's Pydantic-v2 migration warning.

## Stage-review P2: readable Pareto reporting

- Starting evidence: `stage_review.md` verified that all required artists were
  technically present, but the 1% accuracy outlier compressed the other 18
  points and unconditional annotations collided throughout the decision
  cluster.
- Hypothesis: a threshold-centered main panel plus an all-policy overview can
  preserve every exact point and the global Pareto frontier without allowing
  the outlier to destroy decision-region readability.
- Experiment: generate the current 19-row charts under `/tmp`, inspect both
  2,540 x 1,455 PNGs, and add source-only tests for global nondominance,
  decision-row labels, all-point overview coverage, explicit exclusions, and
  PNG creation.
- Verdict: verified. The main panels expose the 89--93% top-1 cluster and the
  100% top-5 stack; the overview panels retain the 1% outlier and every other
  point. The frontier is computed once over all rows, the selected point is a
  red star, and both panels retain the vertical dotted gate. Deterministic
  leader-line lanes label the selection, closest passing alternatives,
  frontier alternatives, and key failures without overlap. Passing rows are
  blue circles and rejected rows are amber X markers.
- Verification:

  ```text
  env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig \
    pytest -q models/autoports/google_gemma_4_31b/tests/test_datatype_sweep_artifacts.py
  ```

  Result: `2 passed in 0.93s`. The final Stage 08 PNGs were intentionally not
  regenerated because review-required candidates were still running.

## Stage-review packaging concern: ignored generator

The ignored untracked
`tests/build_datatype_sweep_artifacts.py` was renamed with `apply_patch` to
`tests/generate_datatype_sweep_artifacts.py`. README and work-log reproduction
commands now use the new module. `git check-ignore -v` produces no match for
the new path, and `git status --short` exposes it as a normal untracked file for
the Stage 08 checkpoint. The old path remains only in `stage_review.md` as the
reviewer's historical finding.

## Final status

The exposed source-boundary violations, BFP8 packed-MLP resource overflow, and
precision-policy API discrepancies are repaired with static regression
evidence. No TT hardware was used in this AutoFix task because the Stage 08
owner serializes device use. Hardware reruns remain owned by the Stage 08
runner; the loader audit itself invalidates no measured TP4 numeric row.
