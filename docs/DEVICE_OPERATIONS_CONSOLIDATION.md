# Device Operations Consolidation Plan

> **Overview**: Migrate all TTML metal device operations and eliminate ttnn_fixed workarounds to achieve the "All device operations in TTNN" success criterion from Phase 1 of the restructuring plan.

This document details the work required to achieve the "All device operations in TTNN" success criterion from the TTNN/TTML restructuring plan. The goal is to consolidate all device operations into `ttnn/cpp/ttnn/operations/` and eliminate the need for `ttml/ttnn_fixed/` workarounds.

## Current State Analysis

### TTML Device Operations to Migrate

Location: `tt-train/sources/ttml/metal/ops/`

| Operation | Target Location | Notes |
|-----------|-----------------|-------|
| `cross_entropy_fw/` | `ttnn/operations/loss/cross_entropy/` | New device op for training |
| `cross_entropy_bw/` | `ttnn/operations/loss/cross_entropy/` | Backward pass |
| `layernorm_fw/` | `ttnn/operations/normalization/layernorm/` | Compare with existing ttnn::layernorm |
| `layernorm_bw/` | `ttnn/operations/normalization/layernorm/` | Add backward to existing |
| `rmsnorm_fw/` | `ttnn/operations/normalization/rmsnorm/` | Compare with existing |
| `rmsnorm_bw/` | `ttnn/operations/normalization/rmsnorm/` | Add backward to existing |
| `sdpa_fw/` | `ttnn/operations/transformer/sdpa/` | Training variant with dropout and intermediates |
| `silu_bw/` | `ttnn/operations/eltwise/unary_backward/` | Fused kernel vs existing composite |
| `swiglu_fw/` | `ttnn/operations/eltwise/` | Custom fused operation |
| `softmax/` | Merge with `ttnn/operations/normalization/softmax/` | Stable softmax variant |
| `profiler_no_op/` | `ttnn/operations/debug/` | Profiling utility |

### TTML Optimizers to Migrate

Location: `tt-train/sources/ttml/metal/optimizers/`

| Operation | Target Location | Notes |
|-----------|-----------------|-------|
| `sgd_fused/` | `ttnn/operations/moreh/moreh_sgd/` | May merge with existing `moreh_sgd` |

### ttnn_fixed Workarounds to Eliminate

Location: `tt-train/sources/ttml/ttnn_fixed/`

| Workaround | Analysis Required | Action |
|------------|-------------------|--------|
| `sum_over_dim`, `sum_over_batch` | Wrapper around `ttnn::moreh_sum` | Inline usage, remove wrapper |
| `log_softmax(tensor, dim)` | Stable log-softmax implementation | Add stable mode to `ttnn::log_softmax` |
| `softmax(tensor, dim)` | Calls `ttnn::softmax` with `stable=true` | Document existing capability, inline |
| `divide(a, b)` | Uses `reciprocal` + `multiply` | Check if `ttnn::divide` works correctly |
| `mean_moreh`, `mean_ttnn` | Wrappers with `ComputeKernelConfig` | Inline with proper defaults |
| `sum_moreh`, `sum_ttnn` | Wrappers with `ComputeKernelConfig` | Inline with proper defaults |
| `sample` | Gumbel sampling implementation | Add as new `ttnn::sample` or `ttnn::gumbel_sample` |
| `to_l1_interleaved`, `to_dram_interleaved` | Simple memory config wrappers | Inline usage |
| `matmul` (with transpose flags) | Wrapper with default core grid | Verify `ttnn::matmul` handles all cases |
| `matmul_backward` | Complex backward logic | Add `ttnn::matmul_backward` operation |

### Distributed Workarounds

Location: `tt-train/sources/ttml/ttnn_fixed/distributed/`

| Workaround | Notes |
|------------|-------|
| `all_gather` | Wrapper that manages semaphores from autograd context |
| `all_reduce` | Wrapper that manages semaphores from autograd context |
| `reduce_scatter` | Wrapper that manages semaphores from autograd context |

These require autograd context integration and may remain as higher-level wrappers until Phase 2.

---

## Migration Work Items

### 1. Cross Entropy Loss (Priority: High)

**Source:** `tt-train/sources/ttml/metal/ops/cross_entropy_fw/` and `cross_entropy_bw/`

**Target:** `ttnn/cpp/ttnn/operations/loss/cross_entropy/`

TTNN currently has `mse_loss` and `l1_loss` but no cross entropy. This is a critical training operation.

**Files to create:**
- `device/cross_entropy_device_operation.hpp`
- `device/cross_entropy_device_operation.cpp`
- `device/cross_entropy_program_factory.cpp`
- `device/kernels/compute/cross_entropy_fw_kernel.cpp`
- `device/kernels/compute/cross_entropy_bw_kernel.cpp`
- `device/kernels/dataflow/reader_*.cpp`
- `device/kernels/dataflow/writer_*.cpp`
- `cross_entropy.hpp`
- `cross_entropy.cpp`
- `cross_entropy_nanobind.hpp`
- `cross_entropy_nanobind.cpp`

**API:**

```cpp
namespace ttnn::operations::loss {
struct CrossEntropyLossOperation {
    static Tensor invoke(
        const Tensor& input,   // logits (N, 1, H, W)
        const Tensor& target,  // ground truth (N, H)
        LossReductionMode mode = LossReductionMode::MEAN,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct CrossEntropyLossBackwardOperation {
    static Tensor invoke(
        const Tensor& grad_output,
        const Tensor& input,
        const Tensor& target,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};
}
```

---

### 2. SiLU Backward - Fused Kernel (Priority: Medium)

**Source:** `tt-train/sources/ttml/metal/ops/silu_bw/`

**Analysis:**
- TTNN already has `ttnn::silu_bw` in `operations/eltwise/unary_backward/`
- Current TTNN implementation is composite: `sigmoid(x) * (1 + x * (1 - sigmoid(x)))`
- TTML has a fused device kernel that may be more efficient

**Decision needed:** Benchmark both implementations. If TTML's fused kernel is faster:

**Target:** `ttnn/cpp/ttnn/operations/eltwise/unary_backward/silu_backward/`

Add as alternative implementation selectable via program config or replace existing.

---

### 3. SwiGLU Forward (Priority: Medium)

**Source:** `tt-train/sources/ttml/metal/ops/swiglu_fw/`

**Analysis:**
- TTNN has `ttnn::swiglu` as a composite operation in `operations/eltwise/unary/`
- TTML has a fused kernel with two variants: `swiglu_fw_kernel.cpp` and `swiglu_fw_kernel_m_fits_l1.cpp`

**Target:** `ttnn/cpp/ttnn/operations/eltwise/swiglu/` (as device operation)

The TTML version takes separate weight tensors (w1, w2, w3), which is different from the composite version. May need to add as a separate operation or extend existing.

---

### 4. SDPA with Training Support (Priority: High)

**Source:** `tt-train/sources/ttml/metal/ops/sdpa_fw/`

**Analysis:**
- TTNN has comprehensive SDPA in `operations/transformer/sdpa/`
- TTML version adds: `dropout_probability` and `return_intermediates` parameters
- These are needed for training backward pass

**Target:** Extend existing `ttnn/cpp/ttnn/operations/transformer/sdpa/`

**Changes required:**
- Add `dropout_probability` parameter to `ExecuteScaledDotProductAttention`
- Add `return_intermediates` parameter to return attention weights for backward
- May need new kernel variants

---

### 5. LayerNorm Backward (Priority: High)

**Source:** `tt-train/sources/ttml/metal/ops/layernorm_bw/`

**Analysis:**
- TTNN has forward layernorm in `operations/normalization/layernorm/`
- TTNN has `moreh_layer_norm_backward` in `operations/moreh/`
- Need to evaluate if TTML version should replace or complement moreh version

**Target:** `ttnn/cpp/ttnn/operations/normalization/layernorm/` or consolidate with moreh

---

### 6. RMSNorm Forward and Backward (Priority: High)

**Source:** `tt-train/sources/ttml/metal/ops/rmsnorm_fw/` and `rmsnorm_bw/`

**Analysis:**
- TTNN has `rmsnorm` in `operations/normalization/rmsnorm/`
- Need to add backward pass

**Target:** `ttnn/cpp/ttnn/operations/normalization/rmsnorm/`

---

### 7. Softmax Stable Mode (Priority: Low)

**Source:** `tt-train/sources/ttml/metal/ops/softmax/`

**Analysis:**
- TTNN already has `ttnn::softmax` with `stable` parameter
- TTML's `ttnn_fixed::softmax` just calls `ttnn::softmax(t, dim, nullopt, config, true)`
- This workaround can be inlined

**Action:** Document that `stable=true` should be used, inline workaround usage.

---

### 8. SGD Fused Optimizer (Priority: Medium)

**Source:** `tt-train/sources/ttml/metal/optimizers/sgd_fused/`

**Analysis:**
- TTNN has `moreh_sgd` in `operations/moreh/moreh_sgd/`
- Compare implementations for feature parity

**Target:** Consolidate into `ttnn/cpp/ttnn/operations/moreh/moreh_sgd/` if different, or verify moreh_sgd meets training needs.

---

### 9. Profiler No-Op (Priority: Low)

**Source:** `tt-train/sources/ttml/metal/ops/profiler_no_op/`

**Target:** `ttnn/cpp/ttnn/operations/debug/profiler_no_op/`

Simple utility operation for profiling.

---

### 10. Matmul Backward (Priority: High)

**Source:** `tt-train/sources/ttml/ttnn_fixed/matmuls.cpp`

**Analysis:**
- This is a composed backward implemented using forward matmul operations
- Not a device kernel, but important for training

**Target:** `ttnn/cpp/ttnn/operations/matmul/` or `ttnn/cpp/ttnn/operations/matmul_backward/`

**API:**

```cpp
namespace ttnn::operations::matmul {
struct MatmulBackward {
    static std::tuple<Tensor, Tensor> invoke(
        const Tensor& a,
        const Tensor& b,
        const Tensor& grad_output,
        bool transpose_a = false,
        bool transpose_b = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};
}
```

---

### 11. Sample/Gumbel Sampling (Priority: Low)

**Source:** `tt-train/sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp` (`sample` function)

**Target:** `ttnn/cpp/ttnn/operations/sampling/` or `ttnn/cpp/ttnn/operations/reduction/sampling/`

Implement Gumbel softmax sampling for training.

---

## Migration Pattern

Each device operation migration follows this pattern:

1. **Create directory structure:**

```
ttnn/cpp/ttnn/operations/<category>/<op_name>/
  CMakeLists.txt
  <op_name>.hpp
  <op_name>.cpp
  <op_name>_nanobind.hpp
  <op_name>_nanobind.cpp
  device/
    <op_name>_device_operation.hpp
    <op_name>_device_operation.cpp
    <op_name>_program_factory.hpp
    <op_name>_program_factory.cpp
    kernels/
      compute/<op_name>_kernel.cpp
      dataflow/reader_<op_name>.cpp
      dataflow/writer_<op_name>.cpp
```

2. **Adapt namespaces:**
   - From: `ttml::metal::ops::<op_name>::device`
   - To: `ttnn::operations::<category>::<op_name>`

3. **Register operation:**

```cpp
namespace ttnn {
constexpr auto <op_name> = ttnn::register_operation<
    "ttnn::<op_name>",
    ttnn::operations::<category>::<OpName>>();
}
```

4. **Add Python bindings** in `<op_name>_nanobind.cpp`

5. **Update CMakeLists.txt** to include new sources

6. **Add tests** in `tests/ttnn/unit_tests/operations/<category>/`

---

## Workaround Elimination Strategy

For each `ttnn_fixed` workaround:

1. **Analyze** why the workaround exists
2. **Fix** the underlying TTNN operation if needed
3. **Inline** the workaround at call sites
4. **Remove** the workaround function
5. **Update** TTML code to use TTNN directly

---

## Success Criteria Checklist

- [ ] Cross entropy loss (forward + backward) in TTNN
- [ ] LayerNorm backward in TTNN
- [ ] RMSNorm backward in TTNN
- [ ] SDPA with dropout and intermediate returns
- [ ] Fused SiLU backward (if faster than composite)
- [ ] Fused SwiGLU (if different from composite)
- [ ] SGD optimizer consolidated
- [ ] Matmul backward operation
- [ ] Gumbel sampling operation
- [ ] Profiler no-op utility
- [ ] All `ttnn_fixed/trivial_ttnn_ops.hpp` functions eliminated
- [ ] All `ttnn_fixed/matmuls.hpp` functions eliminated
- [ ] Distributed operations documented (may remain as wrappers until Phase 2)

---

## Estimated Effort

| Category | Operations | Complexity | Est. Days |
|----------|------------|------------|-----------|
| Loss functions | cross_entropy_fw, cross_entropy_bw | High | 5 |
| Normalization | layernorm_bw, rmsnorm_bw | Medium | 4 |
| Transformer | sdpa_fw training extensions | High | 5 |
| Eltwise | silu_bw fused, swiglu_fw | Medium | 3 |
| Matmul | matmul_backward | Medium | 2 |
| Optimizers | sgd_fused consolidation | Low | 1 |
| Utilities | profiler_no_op, sample | Low | 1 |
| Workaround elimination | inline and remove | Low | 2 |
| Testing | unit tests for all | Medium | 5 |

**Total estimated: ~28 days**

---

## Related Documents

- [TTNN and TTML Architecture Restructuring Plan](TTNN_TTML_RESTRUCTURE_PLAN.md)
