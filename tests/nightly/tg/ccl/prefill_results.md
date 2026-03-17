# Fused MM+RS Prefill Results for Llama 70B Galaxy (8K)

## Summary

Fused `minimal_matmul_strided_reduce_scatter_async` for FF1/FF3 operations in Llama 70B prefill at 8K sequence length.

**Best Result: ~10% improvement per layer, ~23.4ms total prefill improvement for 80 layers**

---

## Baseline (Separate MM + RS)

| Op | MM (µs) | RS (µs) | Total (µs) |
|----|---------|---------|------------|
| FF1 | 751 | 712 | 1,463 |
| FF3 | 752 | 694 | 1,446 |
| **Combined** | | | **2,909** |

---

## Fused Op Results

| Version | Config | FF1 (µs) | FF3 (µs) | Total (µs) | vs Baseline | 80-Layer Savings |
|---------|--------|----------|----------|------------|-------------|------------------|
| v3 | N=32, buf=2 | 1,331 | 1,338 | 2,669 | -8.3% | ~19.2 ms |
| v6 | N=32, buf=4 | 1,303 | 1,327 | 2,630 | -9.6% | ~22.3 ms |
| **v7** | **N=34, buf=4** | **1,300** | **1,316** | **2,616** | **-10.1%** | **~23.4 ms** |
| v8 | N=34, buf=8 | 1,302 | 1,315 | 2,617 | -10.0% | ~23.4 ms |

---

## Best Configuration (v7)

```python
fused_config = ttnn.MinimalMatmulConfig(
    M_block_size=5,
    K_block_size=8,
    N_block_size=34,
    subblock_h=1,
    subblock_w=16,
    compute_with_storage_grid_size=ttnn.CoreCoord(7, 6),
)

# Fused op parameters
num_links=4
num_workers_per_link=2
chunk_width_in_mm_blocks=1
num_buffers_per_channel=4
topology=ttnn.Topology.Ring
cluster_axis=1
rs_core_grid_offset=ttnn.CoreCoord(0, 6)
```

---

## Key Findings

1. **N_block_size=34 works** despite 3584 not being evenly divisible by 34 (kernel handles internally)
2. **Larger N_block improves RS overlap** - more output tiles per chunk = better hiding of RS latency
3. **num_buffers_per_channel=4** provides good balance between performance and L1 usage
4. **Fused op more beneficial for higher ISLs** - larger M dimension = more MM work = more RS overlap opportunity

---

## L1 Constraints

- N=36 with buf=4 causes L1 overflow (~4KB over limit)
- N=36 with buf=2 also causes L1 overflow
- N=34 with buf=4 fits within L1 ✅
- N=32 with buf=4 fits within L1 ✅

---

## Grid Configuration

- **MM cores**: 7x6 = 42 cores (rows 0-5, cols 0-6)
- **RS cores**: row 6 (offset at y=6)
- **Row 7 unavailable**: conflicts with dispatch cores

---

## Tuning Summary

Tried configurations:
- [x] `num_buffers_per_channel=8` with N=34 → No improvement over buf=4 (saturated)
- [x] `N_block_size=35` with buf=4 → Same as N=34, no improvement
- [x] `7x5 grid` with N=36 → L1 overflow (per-core L1 limit, not total)
- [x] `chunk_width_in_mm_blocks=2` → Slightly worse (~10µs)
- [x] `num_workers_per_link=1` → Error (invalid config)

**Conclusion**: N=34 with 7x6 grid, buf=4, chunk=1, workers=2 is the optimal config.

## Next Steps

- [ ] Test at 16K, 32K, 128K sequence lengths to verify scaling
- [ ] Consider kernel-level L1 optimization to enable N=36

---

## Files Modified

- `/home/tvardhineni/teja_RS+MM/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py`
  - Added `USE_FUSED_MM_RS` env var toggle
  - Fused op integration for FF1 and FF3 in `forward_prefill()`
