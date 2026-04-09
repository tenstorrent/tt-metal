# MatmulOp API Gaps

After migrating all 40 call sites (T1-T14, B1-B16), the following API observations
and minor gaps were identified.

## Gap 1: Bias-via-ones pattern requires multiple MatmulOp instances (B14)

**Call sites affected**: B14 (moe_gpt), B13 (moe_compute)

**Issue**: The MOE GPT kernel implements bias addition by calling
`matmul_block(cb_c2c_ones_tile, cb_r2c_w, 0, last_k_index, 0, false, 4, 1, 1)` --
the same matmul_block call but with a DIFFERENT in0 CB (ones tile instead of the
data input). Since MatmulOp binds in0_cb_id at construction time, the caller must
create a separate MatmulOp instance for the bias calls.

**Severity**: Low. Creating a second MatmulOp is trivial (it's a plain struct copy)
and the two instances share the same MATH engine configuration. No init() call is
needed for the bias instance since ct_dim/rt_dim/kt_dim match.

**Possible improvement**: Add an overload of matmul() that takes explicit CB IDs:
```cpp
FORCE_INLINE void matmul(uint32_t in0_cb, uint32_t in1_cb,
                         uint32_t in0_tile_index, uint32_t in1_tile_index,
                         uint32_t dst_tile_index) const;
```
This would allow a single MatmulOp to service both data and bias matmul calls.
However, this blurs the config-based abstraction and might be better left as-is.

**Current workaround**: Two MatmulOp instances (works correctly, no perf impact).

## Gap 2: end_to_output uses sequential pack, not pack_tile_block or pack_tile<true>

**Call sites affected**: B1, B2, B3, B5, B7, B11, B14, B16, T4

**Issue**: Many production kernels use either `pack_tile_block()` (a helper that
packs a contiguous block of DST tiles) or `pack_tile<true>(dst_idx, cb, out_idx)`
(out-of-order packing to specific output positions). MatmulOp's `end_to_output()`
uses the simple `for (i) pack_tile(i, cb)` pattern.

**Severity**: Not a gap -- this is a deliberate design decision (see Design Spec
section 6). Call sites needing custom pack patterns use Mode 1 or Mode 2 with
manual commit/wait/pack/release. The simple sequential pack in `end_to_output()`
correctly serves the subset of call sites that use it.

**No action needed**.

## Gap 3: No PACKER_L1_ACC support in MatmulOp

**Call sites affected**: B1, B2, B3, B9, B15, B16

**Issue**: Several kernels use PACKER_L1_ACC as an alternative to software
spill/reload. The L1_ACC configuration (`llk_pack_reconfig_l1_acc`,
`pack_reconfig_data_format`) is interleaved with the matmul subblock loop.
MatmulOp's `run()` (Mode 3) does software spill/reload only.

**Severity**: Medium for Mode 3 -- kernels using L1_ACC cannot use `run()` without
modification. Low for Mode 1/2 -- callers manage L1_ACC alongside MatmulOp methods.

**Possible improvement**: Add a `bool use_l1_acc` config field to enable L1_ACC
in Mode 3's `run()` method, replacing the spill/reload path with L1_ACC calls.

**Current workaround**: Use Mode 2 for L1_ACC kernels (all current call sites
already use Mode 2 when L1_ACC is enabled).

## Gap 4: Dynamic CB switching in moreh kernels (T6, T8, T9)

**Call sites affected**: T6 (moreh_matmul), T8 (moreh_mean_w), T9 (moreh_sum_w)

**Issue**: These kernels dynamically change the input CB based on whether masking
was applied (e.g., `cb_input = cb_masked_input`). Since MatmulOp binds CB IDs at
construction, a new MatmulOp must be constructed when the CB changes.

**Severity**: Low. Constructing a new MatmulOpConfig + TileMatmulOp is a
zero-overhead operation (struct copy, no heap allocation, no hardware calls).
The init_short() call is needed regardless.

**No action needed** -- constructing per-iteration MatmulOp instances is idiomatic.

## Summary

| Gap | Severity | Action |
|-----|----------|--------|
| Bias-via-ones (multiple MatmulOp) | Low | No change needed; two instances work |
| Sequential pack only | By design | No change needed; use Mode 1/2 |
| No PACKER_L1_ACC in run() | Medium | Future: add `use_l1_acc` config field |
| Dynamic CB switching | Low | No change needed; reconstruct per-iteration |

**Conclusion**: The MatmulOp API is sufficient for all 40 call sites. No blocking
gaps were found. The three areas noted above are either deliberate design choices
or low-severity issues with clean workarounds. The medium-severity L1_ACC gap only
affects Mode 3 and can be addressed in a future iteration if Mode 3 adoption grows.
