# AutoDebug: North-Mini Batch-32 MoE Active-Expert Path

## Scope

Focus path: `models/autoports/coherelabs_north_mini_code_1_0`.

This was a source-only investigation. I did not rerun the AutoDebug launcher, did not edit implementation/tests/docs, and did not attempt hardware reproduction. Hardware-dependent performance conclusions below are therefore framed as code-supported hypotheses unless backed by existing saved artifacts.

## Headline Findings

### 1. Batch-32 optimized MoE is explicitly routed to the dense all-expert path

This is the direct source-level explanation for the reported 3.33 ms device-resident dense path.

`OptimizationConfig` defaults `dense_expert_batch_threshold = 32` and `_sparse_moe()` chooses `_dense_expert_moe_chunk()` whenever `total_tokens >= cfg.dense_expert_batch_threshold`:

- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:72-77`
- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:887-897`

For decode, `seq_len == 1`, so batch 32 gives `total_tokens = 32` and always enters this branch. The dense path repeats the token input across all 128 experts, runs gate/up/down batched dense matmuls, applies routed weights, and sums:

- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:807-885`

The saved candidate `final_correct_moe_decode_b32.json` records `mean_ms = 3.323370839934796`, `batch = 32`, `layer = 1`, and policy fields `dense_expert_batch_threshold = 32`, `dense_expert_chunk_size = 1024`, `dense_expert_cores = 100`:

- `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/candidates/final_correct_moe_decode_b32.json:1-112`

This branch is not a correctness bug by itself; it is logically equivalent to the functional all-expert baseline. But it contradicts the optimize checklist requirement for a batch-32 active-expert runtime path with no dense all-expert path.

### 2. The existing `ttnn.sparse_matmul` formulation is legal, but it is not a compact active-route formulation

The North-Mini sparse calls are legal under the current TTNN sparse-matmul contract:

- gate/up: `A=[1,T,1,H]`, `B=[1,E,H,I]`, `sparsity=[1,T,1,E]`, default `is_input_b_sparse=True`
- down: `A=[T,E,1,I]`, `B=[1,E,I,H]`, `is_input_a_sparse=True`, `is_input_b_sparse=False`

Relevant model call sites:

- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:723-805`

Relevant TTNN contract:

- `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1035-1151`
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:68-169`

The performance problem is visible in the lowering:

- Sparse output shape retains the full batch/expert surface, not a compact `T * top_k` route list.
- Sparse output tensors are zero-filled before compute.
- With `nnz=None`, the factory sets `num_batch_compute = sparsity.logical_volume()`.
- The sender/receivers/compute kernels scan and synchronize over the full sparsity volume, skipping invalid entries dynamically.

Code evidence:

- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:189-210`
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:238-252`
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:264-329`
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp:176-224`
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp:246-256`

For North-Mini batch-32 decode, `T=32`, `E=128`, `top_k=8`. The active routes are nominally `256`, but the `nnz=None` sparse calls process a sparsity volume of `4096` slots for each sparse matmul and still produce full expert-shaped gate/up/down surfaces. The model then multiplies and reduces over the full expert axis:

- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:795-805`

This explains why the recorded sparse-style batch-32 attempts are much slower than the dense fallback:

- `moe_chunk32_fused_silu_hybrid_b32.json`: `mean_ms = 21.29631155075913`, `moe_chunk_size = 32`
- `moe_chunk32_dram_b32.json`: `mean_ms = 21.896113300681463`, `moe_chunk_size = 32`

These artifacts are at:

- `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/candidates/moe_chunk32_fused_silu_hybrid_b32.json:1-67`
- `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/candidates/moe_chunk32_dram_b32.json:1-67`

### 3. A legal faster sparse attempt probably needs separate binary sparsity and routing weights

The current sparse path uses the sigmoid routing weights both as the sparse mask and as the final routing multiplier:

- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:732-735`
- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:795-798`

That is correct for weighting, and `nnz=None` is safe. But it prevents a guaranteed static `nnz = token_count * top_k` unless every selected routing weight remains nonzero after device math/layout conversion.

TTNN documents that explicit `nnz` must exactly equal device-side `count_nonzero(sparsity)`; a mismatch can deadlock:

- `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1051-1052`
- `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1104-1111`

The safer formulation to test is:

1. Build a separate binary BF16 row-major sparsity mask from `top_indices` with ones at selected experts.
2. Pass exact `nnz = token_count * self.top_k` to all three sparse matmuls.
3. Keep the sigmoid `routing` tensor separate for the final post-down weighting.

This should reduce receiver/compute looping from full sparsity volume to exact live routes, while preserving routing weighting. It still will not remove the full-shaped sparse output zero-fill/reduce cost, so this is a focused performance hypothesis, not a guaranteed replacement for the 3.33 ms dense fallback.

## Other Findings And Test Gaps

### Routing correctness looks intact

I did not find a clear expert-id or routing-weight correctness bug. The functional reference evaluates only selected experts and weights by `sigmoid(top_values)`:

- `models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py:278-297`

The functional and optimized dense paths scatter top-k weights to expert positions, multiply expert outputs by that routing tensor, and sum over experts:

- `models/autoports/coherelabs_north_mini_code_1_0/tt/functional_decoder.py:555-583`
- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:823-885`

The optimized sparse path also preserves expert IDs through the scatter mask and applies routing once after down projection:

- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:732-735`
- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:795-805`

### Batch-32 sparse path is not protected by the current tests

`test_optimized_path_audit()` only checks that `_sparse_moe_chunk` contains `sparse_matmul`; it does not assert that batch-32 decode reaches that path:

- `models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py:132-167`

The batch-32 decode correctness test includes `(layer_idx, batch) = (1, 32)`, but because `total_tokens == 32`, it exercises `_dense_expert_moe_chunk()`, not `_sparse_moe_chunk()`:

- `models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py:197-247`
- `models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py:891-897`

The optimized active-expert sparse prefill test uses `sequence = 33`, which also crosses the dense threshold:

- `models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py:337-381`

So the current tests can pass while batch-32 MoE never uses the sparse active-expert runtime path required by the checklist.

### Optimized PCC evidence exists, but path labels need source interpretation

`doc/optimized_decoder/README.md` and `work_log.md` do preserve summarized optimized PCC evidence. The README reports `16 passed in 93.38s` and PCC summaries including traced sliding-MoE batch 1/32 and active-expert layer 1/4 prefill:

- `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/README.md:41-68`

The work log also records the open checklist item directly: MoE still lacks an active-expert path at every measured batch with no dense all-expert runtime fallback:

- `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/work_log.md:174-182`
- `models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/work_log.md:211-213`

What I did not find is a raw optimized pytest/JUnit transcript preserving the per-test `_assert_pcc()` output. `_assert_pcc()` prints PCC at runtime:

- `models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py:317-320`

So the README/work-log PCC summaries are useful evidence, but source-level path tracing is still needed. In particular, labels such as "active-expert layer 1 prefill" should not be read as proof that the current sparse path ran for `sequence=33`, because the current `_sparse_moe()` threshold sends `total_tokens >= 32` to the dense expert branch.

## Focused Implementation Hypotheses For A Later Fix

These are intentionally not implemented in this investigation.

1. Remove or bypass the batch-32 dense-expert threshold for the optimized MoE target path, but avoid falling back to the default `moe_chunk_size=4` batch-32 behavior without re-measuring. A single 32-token sparse chunk matches the recorded sparse attempts and avoids eight repeated sparse-op launches.
2. Use a binary sparsity mask plus exact `nnz = token_count * top_k`, while preserving the existing sigmoid routing tensor for final weighting.
3. Consider packing sparse gate/up weights along the output dimension so gate and up use one sparse matmul plus a split, analogous to the dense `packed_dense_gate_up` option. This is legal by shape, but must be PCC/perf validated.
4. Add a test/audit guard that batch-32 sparse-layer decode does not call `_dense_expert_moe_chunk()` when the active-expert policy is selected.
5. Persist optimized PCC logs and update optimized docs/work log when a candidate is validated.

## Claims Rechecked

I demoted the following attractive but unsupported explanations:

- "The dense all-expert path is a math bug." False; it is a correctness-equivalent baseline/fallback, just disallowed by the optimization contract.
- "`ttnn.sparse_matmul` is illegal for the North-Mini shapes." False under current validation and documented modes.
- "`nnz=None` is a correctness bug." False; it is the safe dynamic mode. The issue is performance/topology.
- "Routing weights are double-applied or expert IDs are lost." Not supported by the source; the expert ID and weight pairing is preserved through scatter and post-down weighting.
