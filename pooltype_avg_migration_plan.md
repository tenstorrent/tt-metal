# Plan: Pass PoolType::AVG from host for mean/avg operations

## Background

Currently `PoolType::AVG` exists in `llk_defs.h` but is never passed from the host. All mean/average operations are implemented as `PoolType::SUM` with a pre-computed `1/N` scaler. At the LLK level, AVG and SUM are **functionally identical** (both use GAPOOL), so this is a safe semantic/labeling change.

## Guiding principle

Change `PoolType::SUM` → `PoolType::AVG` wherever the **scaler tile carries 1/N** (meaning the reduce instruction itself produces a mean). Keep `PoolType::SUM` where the scaler is 1.0 and 1/N is applied as a separate post-reduce step (e.g. `row_wise_mean` in `numeric.h`).

---

## Part A: Host dispatch infrastructure (6 files)

### 1. Add AVG to ReduceOpMath enum
**File:** `ttnn/cpp/ttnn/operations/reduction/generic/device/common.hpp`
- `enum class ReduceOpMath { SUM, MAX, MIN };` → `{ SUM, AVG, MAX, MIN };`

### 2. Update `reduce_op_utils::get_defines()`
**File:** `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
- Add proper handling: AVG → `"PoolType::AVG"`, SUM → `"PoolType::SUM"`, MAX → `"PoolType::MAX"`
- **Important:** Extend `REDUCE_ROW_SUM_VIA_MM` condition (line 34-36) to also fire for `AVG + W`, so mean-along-W keeps the matmul optimization.

### 3. Use ReduceOpMath::AVG in generic_reductions.cpp
**File:** `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp`
- Line 296: `ReduceOpMath::SUM` → `ReduceOpMath::AVG` in the `ReduceType::Mean` branch
- Scaler stays `scalar / reduced_volume`

### 4. Use ReduceOpMath::AVG in moreh_mean factories
**File:** `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/moreh_mean_h_program_factory.cpp` (line 110)
**File:** `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/moreh_mean_w_program_factory.cpp` (line 109)

### 5. Use PoolType::AVG in pool_utils defines
**File:** `ttnn/cpp/ttnn/operations/pool/pool_utils.cpp`
- Line 65: `"PoolType::SUM"` → `"PoolType::AVG"` for `AVG_POOL2D`

---

## Part B: Fused op compute kernels — scaler carries 1/N (change to AVG)

These compute kernels hardcode `PoolType::SUM` but their program factories put 1/N into the scaler tile, so the reduce IS semantically computing an average.

### Layernorm (scaler = 1/block_w, 1/num_blocks)

| Compute kernel | Change |
|---|---|
| `normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp` | `PoolType::SUM` → `AVG` (~6 occurrences) |
| `normalization/layernorm/device/kernels/compute/layernorm_large_tensor.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/layernorm/device/kernels/compute/layernorm_sharded_pre_allgather.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/layernorm/device/kernels/compute/layernorm_sharded_post_allgather.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather.cpp` | `PoolType::SUM` → `AVG` |

### Layernorm pre-allgather distributed (scaler = 1/W)

| Compute kernel | Change |
|---|---|
| `normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp` | `PoolType::SUM` → `AVG` |

### RMSNorm distributed (scaler = 1/N)

| Compute kernel | Change |
|---|---|
| `normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp` | `PoolType::SUM` → `AVG` |

### GroupNorm (scaler = 1/sqrt(N))

| Compute kernel | Change |
|---|---|
| `normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/groupnorm/device/kernels/compute/groupnorm.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/groupnorm/device/kernels/compute/welford_groupnorm_sharded_v2.cpp` | `PoolType::SUM` → `AVG` |
| `normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp` | `PoolType::SUM` → `AVG` |

### Moreh ops (scaler = 1/N or 1/sqrt(N))

| Program factory / compute | Change |
|---|---|
| `moreh/moreh_layer_norm/device/moreh_layer_norm_program_factory.cpp` | `ReduceOpMath::SUM` → `AVG` |
| `moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp` | `ReduceOpMath::SUM` → `AVG` |
| `moreh/moreh_layer_norm_backward/device/moreh_layer_norm_backward_input_grad_program_factory.cpp` | `ReduceOpMath::SUM` → `AVG` |

### Experimental ops

| Compute kernel | Change |
|---|---|
| `experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp` | `PoolType::SUM` → `AVG` |
| `experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather.cpp` | `PoolType::SUM` → `AVG` |
| `experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp` | `PoolType::SUM` → `AVG` |
| `experimental/transformer/dit_layernorm_pre_all_gather/device/kernels/compute/layernorm_pre_allgather_welford.cpp` | `PoolType::SUM` → `AVG` |
| `experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp` | `PoolType::SUM` → `AVG` |

---

## Part C: NO CHANGE — true SUM operations

These use `PoolType::SUM` with scaler=1.0 (or the 1/N is applied separately). The reduce is genuinely a SUM:

- `layernorm.cpp` (non-sharded) — uses `row_wise_mean` which has scaler=1.0 + separate `scale_dest(1/N)`
- All softmax ops — reduce for sum of exponentials
- `moreh_softmax*` — reduce for normalization denominators
- `moreh_sum_h/w_program_factory.cpp` — true summation
- `moreh_dot` — true dot product reduction
- `moreh_norm` — true norm computation
- `moreh_clip_grad_norm` — true norm computation
- `ssm_1d_sum_reduce` — true summation
- `deepseek_grouped_gate` — true summation
- `moe.cpp`, `sampling.cpp` — true summation
- `sdpa_flash_decode.cpp` — true summation for attention

---

## Verification

Since AVG and SUM are identical at the LLK level (both use GAPOOL with scaler), all existing tests should pass unchanged. Key test suites:
- Reduce op tests (mean/sum/max)
- Pool op tests (avg_pool2d)
- moreh_mean tests
- Layernorm tests (sharded + non-sharded)
- Groupnorm tests
- RMSnorm tests
