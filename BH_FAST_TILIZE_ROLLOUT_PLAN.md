# BH Fast-Tilize Rollout Plan: LLK → tt-metal → ttnn

## Goal

Replace the standard tilize path with BH fast-tilize for all supported
format/dimension combinations on Blackhole. The change should be transparent:
any code that calls `fast_tilize_init` / `fast_tilize_block` / `fast_tilize_uninit`
(or the `compute_kernel_lib::tilize` helper with `can_use_fast_tilize` == true)
should automatically get the fast path on BH instead of the current fallback to
standard tilize.

## Current State

### What exists today

| Layer | File | BH fast-tilize? |
|-------|------|-----------------|
| LLK impl | `tt_llk_blackhole/llk_lib/llk_unpack_tilize.h` | Yes |
| LLK impl | `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h` | Yes — self-contained init/uninit (remap handled internally) |
| LLK impl | `tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h` | Yes — BFP-aware pack with per-tile MOP |
| LLK API | `tt_metal/hw/ckernels/blackhole/metal/llk_api/` | **No** — needs `llk_*_fast_tilize_*` wrappers |
| Metal API | `tt_metal/hw/inc/api/compute/tilize.h` | **No** — `#ifdef ARCH_BLACKHOLE` falls back to standard |
| Helper lib | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl` | **No changes needed** — auto-detects fast path |

### What the BH LLK fast-tilize supports

| Input | Output | dest\_acc | Status |
|-------|--------|-----------|--------|
| Float16\_b | Float16\_b | No + Yes | 86 tests (18 shapes), silicon + ttsim |
| Float16\_b | Bfp8\_b | No + Yes | 86 tests (6 shapes), silicon + ttsim |
| Float16\_b | Bfp4\_b | No + Yes | 86 tests (6 shapes), silicon + ttsim |
| Float32 | Float16\_b | No + Yes | 86 tests (5 shapes), silicon + ttsim |
| Float32 | Bfp8\_b | No + Yes | 86 tests (4 shapes), silicon + ttsim |
| Float32 | Bfp4\_b | No + Yes | 86 tests (4 shapes), silicon + ttsim |

All format/dest\_acc combinations pass. dest\_acc=Yes uses `if constexpr`
branches to override `unpack_A_dst` and `pack_src` to `Float16_b` (compat
16-bit DEST) only when `is_fp32_dest_acc_en` is true.

### Performance baseline

Fast-tilize is **2–3.5x faster** than standard tilize on BH (see
`BH_FAST_TILIZE_PERF_COMPARISON.md`). Steady-state: ~26 cyc/tile Float16\_b,
~36 cyc/tile BFP output, vs ~71+ cyc/tile standard.

## Call Chain (top → bottom)

```
ttnn::tilize()
  → TilizeDeviceOperation → ProgramFactory::create()
    → launches compute kernel: ttnn/cpp/ttnn/kernel/compute/tilize.cpp
      → compute_kernel_lib::tilize<...>()           [tilize_helpers.inl]
        → ckernel::fast_tilize_init/block/uninit()  [tt_metal/hw/inc/api/compute/tilize.h]
          → llk_*_fast_tilize_*()                   [llk_api/ wrappers]
            → _llk_*_fast_tilize_*()                [tt-llk LLK impl]
```

Today, the `#ifdef ARCH_BLACKHOLE` in `tilize.h` short-circuits the fast path
to standard tilize. The fix is to wire the BH LLK fast-tilize functions into
the three layers above it.

## Changes Required

### Layer 1: BH LLK API wrappers

**Files:**
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_tilize_api.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_datacopy_api.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h`

**What to add:** BH-specific `llk_*_fast_tilize_*` wrapper functions that
mirror the WH signatures but call the BH LLK implementations.

**Unpack:**
```cpp
inline void llk_unpack_fast_tilize_init(uint32_t operand, uint32_t ct_dim) {
    uint32_t operand_id = get_operand_id(operand);
    // BH fast-tilize forces compat 16-bit DEST.
    // When dest_acc=Yes, override unpack_dst to Float16_b.
    uint32_t dst_fmt = unpack_dst_format[operand_id];
    if constexpr (DST_ACCUM_MODE) {
        dst_fmt = static_cast<uint32_t>(DataFormat::Float16_b);
    }
    _llk_unpack_fast_tilize_init_(dst_fmt, ct_dim, ct_dim <= 1 ? 1 : 4);
}
```

**Math:**
```cpp
inline void llk_math_fast_tilize_init(uint32_t operand, uint32_t unit_dim) {
    uint32_t operand_id = get_operand_id(operand);
    // Remap + fp32 compat handled internally by _llk_math_fast_tilize_init_
    _llk_math_fast_tilize_init_<DST_ACCUM_MODE>(unpack_dst_format[operand_id], 4);
}
```

No remap ordering concern — `_llk_math_fast_tilize_init_` calls
`_llk_math_reconfig_remap_(true)` internally. This works because
`TTI_SEMINIT(max=2, min=0)` sets the semaphore to 0 (min), not 2 (max).
Same pattern as `pack_untilize_dest_init` in production Metal API.

**Pack:**
```cpp
inline void llk_pack_fast_tilize_init(uint32_t icb, uint32_t ocb, uint32_t unit_dim) {
    uint32_t output_id = get_output_id(ocb);
    // Fast-tilize always uses compat 16-bit DEST — override pack_src.
    if constexpr (DST_ACCUM_MODE) {
        uint32_t compat_src = static_cast<uint32_t>(DataFormat::Float16_b);
        _llk_pack_hw_configure_<DST_ACCUM_MODE>(compat_src, pack_dst_format[output_id], ...);
    } else {
        _llk_pack_hw_configure_<DST_ACCUM_MODE>(pack_src_format[output_id], pack_dst_format[output_id], ...);
    }
    _llk_pack_fast_tilize_init_<DST_SYNC_MODE, DST_ACCUM_MODE>(
        0, pack_dst_format[output_id], unit_dim <= 1 ? 1 : 4, 4);
}
```

### Layer 2: Metal Compute Kernel API

**File:** `tt_metal/hw/inc/api/compute/tilize.h`

Replace the BH fallback stubs with real implementations.

**`fast_tilize_init`:** Width-1 falls back to `tilize_init`. Width >= 2 calls
the BH LLK wrappers. No remap ordering issue — init is self-contained.

```cpp
ALWI void fast_tilize_init(uint32_t icb, uint32_t full_dim, uint32_t ocb, ...) {
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);
#ifdef ARCH_BLACKHOLE
    if (full_dim <= 1) {
        tilize_init(icb, full_dim, ocb, call_line);
    } else {
        UNPACK((llk_unpack_fast_tilize_init(icb, full_dim)));
        MATH((llk_math_fast_tilize_init(icb, 4)));
        PACK((llk_pack_fast_tilize_init(icb, ocb, 4)));
    }
#else
    // existing WH path
#endif
}
```

**`fast_tilize_block`:** Decompose width into `{4, 2, 3}` units. One
`section_done` per unit. Width-1 falls back to `tilize_block`.

**`fast_tilize_uninit`:** Width-1 falls back to `tilize_uninit`. Width >= 2
calls the BH uninit functions (remap disable handled internally).

### Layer 3: Tilize Helper Library

**File:** `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl`

**No changes required.** The helper already calls `fast_tilize_init/block/uninit`
and the compile-time gate `can_use_fast_tilize` already checks for
Float32/Float16\_b + 32x32 tiles + SyncHalf. Once the BH `#ifdef` fallback is
removed, BH automatically uses the fast path.

### Layer 4: Format support gating

The `has_supported_fast_tilize_format` check in `tilize_helpers.inl` only
allows Float32 and Float16\_b as input formats. Output format is not checked —
the pack stage handles BFP conversion transparently.

Unsupported formats (Int32, UInt16, etc.) and SyncFull mode automatically fall
back to standard tilize via the compile-time gate.

## Implementation Order

### Phase 1: Wire up BH fast-tilize

1. Add `llk_*_fast_tilize_*` wrappers in BH `llk_api/` headers
   - `if constexpr (DST_ACCUM_MODE)` branches for format overrides
   - `#include "llk_pack_fast_tilize.h"` in `llk_pack_api.h`
2. Replace `#ifdef ARCH_BLACKHOLE` fallback in `tilize.h`
   - Width-1: fall back to standard `tilize_init/block/uninit`
   - Width >= 2: call BH fast-tilize wrappers
3. Implement BH `fast_tilize_block` in `tilize.h` with `decompose_row`
4. Run existing ttnn tilize tests on BH — verify no regressions
5. Run perf comparison: ttnn tilize before/after

## Testing Plan

### Level 1: LLK accuracy (done)

86 tests in `test_fast_tilize_full.py` on silicon + ttsim. All 6 format
combos × both dest\_acc modes.

### Level 2: LLK perf (done)

`perf_fast_tilize_full.py` mirrors CI perf suite. No regressions.

### Running tt-metal tests on ttsim

tt-metal tests can run on the BH ttsim simulator instead of silicon. This
catches deadlocks and data errors without hanging the device.

**Setup:**
```bash
# Build ttsim (local clone)
cd /home/developer/ttsim-private/src
../make.py _out/release_bh/libttsim.so

# Deploy simulator binary + SoC descriptor
mkdir -p ~/sim
cp _out/release_bh/libttsim.so ~/sim/libttsim_bh.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml
```

**Running:**
```bash
# Requires slow dispatch mode + simulator path
export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so
export TT_METAL_SLOW_DISPATCH_MODE=1

python -m pytest <test_path> --timeout=60 [pytest_args]
```

**Notes:**
- `scripts/run_safe_pytest.sh` does NOT work with ttsim (it resets physical
  devices on timeout). Use `python -m pytest` directly with `--timeout`.
- `TT_METAL_SLOW_DISPATCH_MODE=1` is required — ttsim does not support fast
  dispatch.
- The SoC descriptor (`soc_descriptor.yaml`) must be in the same directory
  as the simulator `.so`.
- ttsim is slower than silicon but catches deadlocks as timeouts rather than
  device hangs that require `tt-smi -r 0` to recover.
- The LLK test infra uses `TT_LLK_TEST_TARGET=ttsim` instead (different
  mechanism, not tt-metal).

### Level 3: Metal compute kernel API (rollout)

Start with a single minimal test on ttsim before expanding:

1. **Single case on ttsim:** BFLOAT16 32×128 (width=4 tiles, single 4-wide unit)
   - Validates the full init → block → uninit path through LLK API wrappers
   - ttsim catches deadlocks and wrong data without hanging silicon
2. **Expand on ttsim:** Add width=2, width=8, multi-row, dest\_acc=Yes
3. **Single case on silicon:** Same 32×128 shape, verify no hang
4. **Full suite on silicon:**
   ```
   scripts/run_safe_pytest.sh tt_metal/tests/compute/ -k "tilize" --arch blackhole
   ```

### Level 4: ttnn op-level (rollout)

Same incremental approach:

1. **Single case on ttsim:** `test_tilize.py` with BFLOAT16 32×128
2. **Expand on ttsim:** Different shapes, dtypes, memory configs
3. **Single case on silicon** with `scripts/run_safe_pytest.sh`
4. **Full suite on silicon:**
   ```
   scripts/run_safe_pytest.sh ttnn/tests/ -k "tilize" --arch blackhole
   ```

### Level 5: ttnn op perf (rollout)

Before/after comparison on representative shapes. Only after Level 4 passes.

### Regression gate

The fast path is gated by compile-time checks. Standard tilize is never
removed. The `#ifdef` fallback can be restored per-case if needed.

## Risk Assessment

**Low risk.** Compile-time gating. LLK thoroughly tested (86 tests, silicon +
ttsim). Init/uninit self-contained (no external remap calls needed).

**Scheduling:** BH uses `unit_dim={4,2,3}` (not `{1,2,3}` like WH).
unit\_dim=4 gives ~2 cyc/tile better amortization. The `decompose_row` logic
from the test kernel provides the reference implementation.

## Files to Change

| File | Change |
|------|--------|
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_tilize_api.h` | Add `llk_unpack_fast_tilize_init/block/uninit` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_datacopy_api.h` | Add `llk_math_fast_tilize_init/block_/uninit` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h` | Add `llk_pack_fast_tilize_init/block/uninit` + include |
| `tt_metal/hw/inc/api/compute/tilize.h` | Replace BH `#ifdef` fallback with real impl |
| `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl` | No changes expected |
