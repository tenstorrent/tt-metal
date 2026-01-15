# Eliminate ttnn_fixed Workarounds

> **Overview**: Improve TTNN APIs to eliminate the need for `ttml::ttnn_fixed::*` workarounds by adding missing operations, improving existing APIs, and providing training-friendly defaults.

## TODOs

| ID | Task | Status |
|----|------|--------|
| add-log-softmax | Add `ttnn::log_softmax` to `ttnn/cpp/ttnn/operations/normalization/` | Pending |
| add-matmul-backward | Add `ttnn::matmul_backward` to `ttnn/cpp/ttnn/operations/matmul/` | Pending |
| verify-divide-precision | Verify `ttnn::divide` precision vs `reciprocal*multiply` workaround | Pending |
| add-compute-presets | Add training compute kernel config presets | Pending |
| consider-gumbel-sample | Consider adding `ttnn::gumbel_sample` for sampling | Pending |
| migrate-ttml-callsites | Update TTML call sites to use native TTNN APIs | Pending |
| delete-ttnn-fixed | Delete `ttnn_fixed` directory after migration | Pending |

---

## Current State Analysis

The `ttml::ttnn_fixed` namespace ([`tt-train/sources/ttml/ttnn_fixed/`](../tt-train/sources/ttml/ttnn_fixed/)) contains workarounds for TTNN API gaps. These must be absorbed into TTNN proper.

### Workarounds Inventory

**File: [`trivial_ttnn_ops.hpp`](../tt-train/sources/ttml/ttnn_fixed/trivial_ttnn_ops.hpp)**

| Workaround | Issue | Action Required |
|------------|-------|-----------------|
| `log_softmax(t, dim)` | Missing in TTNN | Add `ttnn::log_softmax` |
| `softmax(t, dim)` | Needs `stable=true` + compute config | Document defaults, possibly add preset |
| `divide(a, b)` | Uses `reciprocal * multiply` workaround | Verify `ttnn::divide` precision, document |
| `sum_over_dim`, `sum_over_batch` | Convenience wrappers | Optional - document pattern |
| `mean_moreh`, `mean_ttnn`, `sum_moreh`, `sum_ttnn` | Wrappers with precise compute config | Add training config presets |
| `sample(t, temp, seed, mask)` | Gumbel sampling | Add `ttnn::gumbel_sample` |
| `to_l1_interleaved`, `to_dram_interleaved` | Convenience wrappers | Document pattern (not needed) |

**File: [`matmuls.hpp`](../tt-train/sources/ttml/ttnn_fixed/matmuls.hpp)**

| Workaround | Issue | Action Required |
|------------|-------|-----------------|
| `matmul(a, b, transpose_a, transpose_b)` | Fixed core_grid + compute config | Use existing API params |
| `matmul_backward(a, b, grad, ta, tb)` | Missing backward op | Add `ttnn::matmul_backward` |

**File: [`distributed/ttnn_ops.hpp`](../tt-train/sources/ttml/ttnn_fixed/distributed/ttnn_ops.hpp)**

| Workaround | Issue | Action Required |
|------------|-------|-----------------|
| `all_gather(t, dim)` | Wraps experimental async API with semaphore management | Simplify CCL API or move autograd integration |
| `all_reduce(t)` | Same | Same |
| `reduce_scatter(t, dim)` | Same | Same |

**File: [`distributed/tt_metal.hpp`](../tt-train/sources/ttml/ttnn_fixed/distributed/tt_metal.hpp)**

| Workaround | Issue | Action Required |
|------------|-------|-----------------|
| `enable_fabric(num_devices)` | Sets mesh graph descriptor env var | Keep as TTML utility or document setup |

---

## Phase 1: Add Missing TTNN Operations

### 1.1 Add `ttnn::log_softmax`

Currently implemented in TTML as:

```cpp
// Stable log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
tt::tt_metal::Tensor log_softmax(const tt::tt_metal::Tensor& t, int dim) {
    auto t_max = ttnn::max(t, dim, true);
    auto t_sub_max = ttnn::subtract(t, t_max);
    auto t_sub_max_exp = ttnn::exp(t_sub_max);
    auto t_sum_over_dim = sum_over_dim(t_sub_max_exp, dim);
    auto log_t_sum_over_dim = ttnn::log(t_sum_over_dim, true);
    return ttnn::subtract(t_sub_max, log_t_sum_over_dim);
}
```

**Action**: Add `ttnn::log_softmax` to [`ttnn/cpp/ttnn/operations/normalization/`](../ttnn/cpp/ttnn/operations/normalization/)
- Create `log_softmax/` directory following `softmax/` pattern
- Implement numerically stable version (using the pattern above or fused kernel)
- Register as `ttnn::log_softmax`

### 1.2 Add `ttnn::matmul_backward`

Currently implemented in [`matmuls.cpp`](../tt-train/sources/ttml/ttnn_fixed/matmuls.cpp) with full transpose handling.

**Action**: Add to [`ttnn/cpp/ttnn/operations/matmul/`](../ttnn/cpp/ttnn/operations/matmul/)
- Add `MatmulBackwardOperation` struct
- Handle all transpose combinations (transpose_a, transpose_b)
- Return `std::pair<Tensor, Tensor>` for gradients

### 1.3 Add `ttnn::gumbel_sample` (Optional)

Currently implements Gumbel-max trick for sampling:

```cpp
// Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
rand = ttnn::neg(ttnn::log(ttnn::neg(ttnn::log(rand))));
out = ttnn::mul_sfpu(out, 1.0F / temperature);
out = ttnn::add(out, rand);
return ttnn::argmax(ttnn::untilize(out), 3, true, std::nullopt, true);
```

**Action**: Consider adding `ttnn::gumbel_sample` to [`ttnn/cpp/ttnn/operations/reduction/sampling/`](../ttnn/cpp/ttnn/operations/reduction/sampling/)

---

## Phase 2: Improve Existing TTNN APIs

### 2.1 Verify `ttnn::divide` Precision

TTML uses `reciprocal * multiply` instead of `ttnn::divide`:

```cpp
tt::tt_metal::Tensor divide(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b) {
    auto inv_b = ttnn::reciprocal(b);
    return ttnn::multiply(a, inv_b);
}
```

**Action**:
- Test `ttnn::divide` with training workloads
- If precision issues exist, fix `ttnn::divide` implementation
- Document recommended approach for high-precision divide

### 2.2 Add Training Compute Kernel Config Presets

TTML uses [`core/compute_kernel_config.hpp`](../tt-train/sources/ttml/core/compute_kernel_config.hpp):

```cpp
static ttnn::WormholeComputeKernelConfig precise();  // High precision
static ttnn::WormholeComputeKernelConfig softmax();  // Softmax-specific
static ttnn::WormholeComputeKernelConfig matmul();   // Matmul-specific
static ttnn::WormholeComputeKernelConfig fast();     // Performance-oriented
```

**Action**: Add to [`ttnn/cpp/ttnn/operations/core/compute_kernel/`](../ttnn/cpp/ttnn/operations/core/compute_kernel/)
- Add `ttnn::ComputeKernelConfigPresets` or similar
- Provide `training()`, `inference()`, `precise()`, `fast()` presets

---

## Phase 3: Simplify Distributed CCL API

### Current Problem

TTML distributed wrappers manage CCL resources manually:

```cpp
tt::tt_metal::Tensor all_gather(const tt::tt_metal::Tensor& tensor, int dim) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    return ttnn::experimental::all_gather_async(
        tensor, dim,
        ccl_resources.get_all_gather_semaphore(),
        /* ... many params ... */);
}
```

### Options

**Option A**: Promote experimental APIs to stable with simpler signatures
- Move `ttnn::experimental::all_gather_async` -> `ttnn::all_gather`
- Auto-manage semaphores internally

**Option B**: Keep CCL resource management in autograd layer
- When autograd moves to TTNN (Phase 2 of main plan), move CCL helpers too
- These become `ttnn::autograd::all_gather`, etc.

**Action**: Defer to Phase 2 of main restructuring plan - CCL wrappers will move with autograd infrastructure.

---

## Phase 4: Remove ttnn_fixed

After completing Phases 1-3:

1. Update all TTML call sites to use native TTNN APIs
2. Delete [`tt-train/sources/ttml/ttnn_fixed/`](../tt-train/sources/ttml/ttnn_fixed/)
3. Delete [`tt-train/tests/ttnn_fixed/`](../tt-train/tests/ttnn_fixed/)

---

## File Changes Summary

### New Files to Create

```
ttnn/cpp/ttnn/operations/normalization/log_softmax/
  log_softmax.hpp
  log_softmax.cpp
  log_softmax_nanobind.hpp
  log_softmax_nanobind.cpp

ttnn/cpp/ttnn/operations/matmul/
  matmul_backward.hpp  (or add to matmul.hpp)
  matmul_backward.cpp
```

### Files to Modify

```
ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt  (add log_softmax)
ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.cpp
ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp
```

### Files to Delete (after migration)

```
tt-train/sources/ttml/ttnn_fixed/  (entire directory)
tt-train/tests/ttnn_fixed/  (entire directory)
```

---

## Dependencies

- Phase 1 and 2 can proceed independently
- Phase 3 depends on autograd migration (main plan Phase 2)
- Phase 4 depends on all prior phases

## Testing Strategy

1. Run existing TTML tests with modified call sites
2. Add unit tests for new operations (`log_softmax`, `matmul_backward`)
3. Benchmark training performance before/after
