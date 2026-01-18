# Kernel Analysis Final Report: NOC Write Barrier Patterns

## Executive Summary

Analysis of ~250 data movement kernels in the tt-metal repository for inefficient or unsafe NOC/fabric write patterns.

### Issue Categories

| Category | Description | Fix |
|----------|-------------|-----|
| **heavy_barrier** | CB wait/pop is INSIDE loop. Synchronization IS required before `cb_pop_front`, but `noc_async_write_barrier()` is heavier than needed. | Replace with `noc_async_writes_flushed()` |
| **barrier_location** | CB wait/pop is OUTSIDE the loop but barrier is INSIDE. Barrier can be moved outside. | Move barrier outside loop, use `noc_async_writes_flushed()` |
| **missing_flush** (UNSAFE) | Missing synchronization before `cb_pop_front` after writes from CB. | Add `noc_async_writes_flushed()` before pop |
| **CORRECT** | Valid patterns - cross-core signaling, ring algorithms, mcast sync. | No fix needed |

### Overall Statistics

| Category | Count |
|----------|------:|
| heavy_barrier | ~95 |
| barrier_location | ~15 |
| missing_flush (UNSAFE) | 1 |
| CORRECT (no fix needed) | ~15 |
| UNCLEAR (needs manual review) | ~5 |

---

## Key Understanding

### When is `noc_async_write_barrier()` required?

1. **At the end of a kernel** - ensures all writes complete before kernel terminates
2. **Before signaling another core about data** - when writing to location A, then signaling core B about data at A

### When should `noc_async_writes_flushed()` be used instead?

**Every other case**, including:
- Before `cb_pop_front` to ensure CB memory isn't reused before write completes
- Inside loops where multiple writes occur sequentially

### The Optimal Pattern

```cpp
for (uint32_t i = 0; i < iterations; i++) {
    cb_wait_front(cb, batch_size);
    for (uint32_t j = 0; j < batch_size; j++) {
        noc_async_write_tile(...);
    }
    noc_async_writes_flushed();  // Lightweight - ensures writes are issued
    cb_pop_front(cb, batch_size);
}
noc_async_write_barrier();  // Full barrier only at kernel end
```

---

## UNSAFE Issues (High Priority)

### Issue: root_receive_writer_kernel.cpp
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root_receive_writer_kernel.cpp`
**Lines**: 23-39
**Problem**: First two `cb_pop_front` calls happen BEFORE any flush/barrier.
**Fix**: Add `noc_async_writes_flushed()` before each `cb_pop_front`.

---

## High Priority Inefficient Patterns

### Multiple barriers per loop iteration:

| File | Barriers/Iteration | Location |
|------|-------------------|----------|
| writer_running_statistics.cpp | 3 | batch_norm |
| writer_moreh_adamw.cpp | 3-4 | moreh_adamw |
| writer_moreh_adam.cpp | 3-4 | moreh_adam |
| writer_moreh_sgd.cpp | 2 | moreh_sgd |
| writer_rotate_half_interleaved_start_id.cpp | 2 | rotate_half |

### Deeply nested barriers:

| File | Nesting Level | Location |
|------|---------------|----------|
| writer_interleaved_no_bcast.cpp | 6 levels | binary_ng |
| writer_unary_interleaved_start_id_wh.cpp | 3 levels | eltwise/unary |
| welford_writer_unary_gn_rm_gb.cpp | 2 levels | groupnorm |

---

## Summary by Directory

### CCL/Fabric Operations
| Category | Count |
|----------|------:|
| heavy_barrier | 4 |
| missing_flush (UNSAFE) | 1 |
| CORRECT | 2 |

### Data Movement Operations
| Category | Count |
|----------|------:|
| heavy_barrier | 24 |
| barrier_location | 8 |
| CORRECT | 3 |

### Moreh/Reduction Operations
| Category | Count |
|----------|------:|
| heavy_barrier | 54 |
| barrier_location | 2 |
| CORRECT | 1 |

### Transformer/Matmul Operations
| Category | Count |
|----------|------:|
| heavy_barrier | 12 |
| barrier_location | 1 |
| CORRECT | 4 |

### Other Operations (normalization, eltwise, pool, etc.)
| Category | Count |
|----------|------:|
| heavy_barrier | 11 |
| barrier_location | 2 |
| CORRECT | 3 |

---

## Good Reference Implementations

These files demonstrate optimal patterns:

1. **KV Cache**: `kv_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`
   - Uses `noc_async_writes_flushed()` in loop, `noc_async_write_barrier()` at end

2. **Paged Cache**: `experimental/paged_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`
   - Same optimal pattern

3. **Deepseek**: `experimental/reduction/deepseek_grouped_gate/device/kernels/dataflow/writer_deepseek_grouped_gate.cpp`
   - Batches writes, flush in loop, barrier at end

4. **RMS Pre-Allgather**: `fused_distributed_rmsnorm/device/kernels/dataflow/rms_pre_allgather_writer.cpp`
   - Flush in loop, barrier at end

5. **Slice Writers**: `slice/writer_multicore_slice_4d.cpp`, `writer_multicore_slice_nd.cpp`
   - Flush in loop, barrier at end

---

## Recommendations

### Immediate Actions

1. **Fix UNSAFE issue**: `root_receive_writer_kernel.cpp` - add missing flushes

2. **Fix high-impact files**: Focus on files with multiple barriers per iteration:
   - `writer_running_statistics.cpp` (3 barriers)
   - `writer_moreh_adamw.cpp` (3-4 barriers)
   - `writer_moreh_adam.cpp` (3-4 barriers)

3. **Simple find-and-replace** for most files: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` (except at kernel end or before cross-core signaling)

### Verification Checklist

Before changing any barrier to flush, verify:
- [ ] No cross-core signaling follows the barrier
- [ ] Not at kernel end (keep barrier at end)
- [ ] Not followed by reading from same location by another core
- [ ] The pattern is write → flush → cb_pop_front (not write → cb_pop_front → flush)

---

## Files Generated

| File | Description |
|------|-------------|
| `ccl_fabric_analysis.md` | CCL/fabric kernel analysis (4 heavy_barrier, 1 unsafe, 2 correct) |
| `data_movement_analysis.md` | Data movement analysis (24 heavy_barrier, 8 barrier_location, 3 correct) |
| `moreh_reduction_analysis.md` | Moreh/reduction analysis (54 heavy_barrier, 2 barrier_location, 1 correct) |
| `transformer_matmul_analysis.md` | Transformer/matmul analysis (12 heavy_barrier, 1 barrier_location, 4 correct) |
| `other_ops_analysis.md` | Other operations analysis (11 heavy_barrier, 2 barrier_location, 3 correct) |
| `final_report.md` | This consolidated report |

All files located at: `/Users/snijjar/work/tt-metal/kernel_analysis_results/`
