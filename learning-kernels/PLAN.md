# FP32 SFPU Untilize: Learning Plan & Implementation Guide

## The Problem

When `ttnn.untilize()` or `ttnn.untilize_with_unpadding()` runs on an FP32 tensor **wider than 8 tiles (256 elements)**, it falls back to a slow kernel that **truncates FP32 to TF32**, losing precision. This is tracked in GitHub issue #33795.

PR #33904 partially fixed this: FP32 tensors **8 tiles wide or less** now preserve precision via the fast `pack_untilize` path. The remaining problem is the wide-tensor slow path.

### Why the slow path loses precision

In the factory files (e.g., `untilize_single_core_program_factory.cpp:145-160`), the code does:

```cpp
// Line 147: Set FP32 mode for the unpacker
unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;

// Line 151-157: But if we must use the slow kernel, OVERRIDE it back to Default (TF32)
if (!use_pack_untilize || (FLOAT32 && num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH)) {
    compute_kernel = "untilize.cpp";  // slow path
    unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::Default;  // <-- precision lost here
}
```

The override to `Default` is **intentional** - the slow untilize LLK was never built to support unpack-to-dest mode (see Root Cause Analysis below). Your job is to make it work.

---

## The Two Existing Code Paths

### Fast path: `pack_untilize.cpp`

**File:** `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp`

```
L1 input CB ──[Unpacker]──> DEST (tile order) ──[Packer with ADDR_MOD]──> L1 output CB (row-major)
```

- Tiles are unpacked into DEST in normal tile format
- The **packer hardware** uses address modification registers (`ADDR_MOD_0` through `ADDR_MOD_3`) to read DEST in a non-linear pattern, writing row-major data to L1
- Width limit: 4 tiles for FP32 in `block_ct_dim` (DEST can hold 4 FP32 tiles in half-sync)
- The `full_ct_dim` template parameter encodes the total row width so the packer can compute correct output addresses
- `MAX_PACK_UNTILIZE_WIDTH = 8` is the policy limit in the factory

### Slow path: `untilize.cpp`

**File:** `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp`

```
L1 input CB ──[Unpacker with Y-stride]──> DEST (row-major-ish) ──[Math datacopy]──> ──[Packer linear]──> L1 output CB
```

- The **unpacker** uses modified Y-stride to read tile data in a rearranged order, so data enters DEST already in row-major layout
- Math does a simple `A2D` datacopy (SrcA to Dest)
- Packer writes linearly from DEST to L1
- No width limit, but forces `UnpackToDestMode::Default` (TF32 truncation)

### Key difference

| Aspect | Fast (pack_untilize) | Slow (untilize) |
|--------|---------------------|-----------------|
| **Who rearranges** | Packer (ADDR_MOD hardware) | Unpacker (Y-stride config) |
| **When rearranged** | At pack time (DEST→L1) | At unpack time (L1→DEST) |
| **Data route** | L1 → Unpacker → SrcA → Math A2D → DEST → Packer (ADDR_MOD) → L1 | L1 → Unpacker (Y-stride) → SrcA → Math A2D → DEST → Packer (linear) → L1 |
| **Width limit** | 4 tiles FP32 / 8 tiles FP16 | None |
| **FP32 precision** | Preserved (after PR #33904) | Lost (forced TF32) |
| **Unpack-to-dest** | Configured via `set_dst_write_addr` in `llk_unpack_A` | **Never configured** - this is the root cause |

---

## Root Cause Analysis

### Why the slow path can't do FP32 (it's a software issue)

When `UnpackToDestFp32` is enabled, the unpacker must write **directly to DEST**, bypassing the SrcA register. This requires specific hardware configuration that the slow untilize LLK never sets up.

#### What the fast path does (in `llk_unpack_A.h`)

When `unpack_to_dest=true` and the input is 32-bit, `_llk_unpack_A_` calls `set_dst_write_addr()` which does:

```cpp
// cunpack_common.h:437-458
inline void set_dst_write_addr(const uint32_t &context_id, const uint32_t &unpack_dst_format) {
    uint32_t dst_byte_addr = 16 * (4 + mailbox_read(ThreadId::MathThreadId));
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0);           // 1. Disable address bit swizzle
    TTI_RDCFG(..., UNP0_ADDR_CTRL_ZW_REG_1_Zstride); // 2. Save current Z-stride
    // ... configure new Z-stride for DEST ...
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);  // 3. KEY: Enable unpack-to-dest
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);  // 4. Set DEST base addr
}
```

It also uses **different UNPACR instruction encodings**:
```cpp
// Normal (to SrcA):
UNPACR(SrcA, 0b1,        ..., 1 /*Dvalid*/, ...)   // Z-inc=1, sets data valid

// Unpack-to-dest (to DEST directly):
UNPACR(SrcA, 0b00010001, ..., 0 /*Dvalid*/, ...)   // Different Z-inc pattern, no Dvalid
```

And after the MOP completes, it calls `unpack_to_dest_tile_done()` for cleanup:
```cpp
// cunpack_common.h:414-435
inline void unpack_to_dest_tile_done(uint32_t &context_id) {
    t6_semaphore_post(semaphore::UNPACK_TO_DEST);       // Signal completion
    TTI_WRCFG(...);                                       // Restore Z-stride
    cfg_reg_rmw_tensix<Unpack_if_sel_cntx0_RMW>(0);      // Disable unpack-to-dest
    cfg_reg_rmw_tensix<Dest_cntx0_address_RMW>(4 * 16);  // Restore DEST address
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);                // Re-enable address swizzle

    // HARDWARE BUG WORKAROUND (TEN-3868):
    // Must issue one unpack-to-SrcA after last unpack-to-dest
    TT_UNPACR(SrcA, 0, 0, context_id, 0, 1, 0, ...);
}
```

#### What the slow path does (in `llk_unpack_untilize.h`)

`_llk_unpack_untilize_init_` and `_llk_unpack_untilize_pass_` do **none of the above**:
- Never call `set_dst_write_addr()`
- Never set `Unpack_if_sel = 1`
- Never set the DEST base address
- Never call `wait_for_dest_available()` (semaphore sync)
- Use the normal UNPACR encoding (to SrcA, with Dvalid)
- Never call `unpack_to_dest_tile_done()` for cleanup

The slow path always unpacks to **SrcA**, then relies on the Math thread's `A2D` datacopy to move data into DEST. When `UnpackToDestFp32` mode is enabled at the hardware level but the LLK doesn't configure the registers for it, the unpacker doesn't know to route data to DEST with full FP32 precision.

### The most promising fix: LLK modification

The fix is to modify `_llk_unpack_untilize_pass_` (or create a variant) to support unpack-to-dest mode. This requires:

1. **Register setup**: Call `set_dst_write_addr()` to set `Unpack_if_sel=1`, DEST base address, Z-stride
2. **Different UNPACR encoding**: Use the `0b00010001` Z-inc pattern instead of `0b01000001`, don't set Dvalid
3. **Semaphore sync**: Call `wait_for_dest_available()` before the MOP
4. **Cleanup**: Call `unpack_to_dest_tile_done()` after each pass
5. **HW bug workaround**: TEN-3868 requires one SrcA unpack after the last to-dest unpack
6. **Y-stride preservation**: The untilize Y-stride trick (which does the tile→row-major rearrangement) must work simultaneously with unpack-to-dest routing

The key insight: the Y-stride controls **how the unpacker reads from L1** (the rearrangement), while `Unpack_if_sel` controls **where the unpacker writes** (SrcA vs DEST). These should be independent — but this needs to be verified experimentally.

---

## Files You Need to Study

### Tier 1: The kernels themselves (read these first)

| File | What it is |
|------|-----------|
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp` | Slow kernel (28 lines, simple) |
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp` | Fast kernel (39 lines, also simple) |

### Tier 2: The APIs they call

| File | What it is |
|------|-----------|
| `tt_metal/include/compute_kernel_api/untilize.h` | Public API for slow untilize (`untilize_init`, `untilize_block`) |
| `tt_metal/include/compute_kernel_api/pack_untilize.h` | Public API for fast untilize (`pack_untilize_init`, `pack_untilize_block`) |
| `tt_metal/include/compute_kernel_api/compute_kernel_hw_startup.h` | `compute_kernel_hw_startup()` - called once at kernel start |

### Tier 3: LLK implementation (the hardware-level code) — THIS IS WHERE THE FIX LIVES

| File | What it is |
|------|-----------|
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_untilize.h` | **THE FILE TO MODIFY** — slow path unpacker, missing unpack-to-dest support |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_A.h` | **Reference implementation** — see `_llk_unpack_A_` for how unpack-to-dest is done correctly (lines 260-287) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/cunpack_common.h` | `set_dst_write_addr()` (line 437) and `unpack_to_dest_tile_done()` (line 414) — the register setup/cleanup you need |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h` | How the packer does the rearrangement in the fast path |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_datacopy_api.h` | The A2D datacopy math operation |

### Tier 4: Factory files (where the kernel is selected)

| File | What it is |
|------|-----------|
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_single_core_program_factory.cpp` | Simplest factory - start here |

There are ~11 factory files total (single core, multi core, sharded, etc.) that all have the same TODO. Once your kernel works, you'll need to update all of them. But start with single_core.

### Tier 5: SFPU reference (for writing your kernel)

| File | What it is |
|------|-----------|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_identity_kernel.cpp` | Simplest SFPU compute kernel pattern |
| `tt_metal/include/compute_kernel_api/eltwise_unary/eltwise_unary.h` | SFPU eltwise API |
| `tt_metal/include/compute_kernel_api/tile_move_copy.h` | `copy_tile`, `copy_tile_init` - tile movement APIs |

---

## Concepts You Need to Understand

### 1. Tile Layout
A 32x32 tile is stored as 4 "faces" of 16x16 elements:
```
Face 0 (rows 0-15, cols 0-15)   | Face 1 (rows 0-15, cols 16-31)
Face 2 (rows 16-31, cols 0-15)  | Face 3 (rows 16-31, cols 16-31)
```
Within each face, data is stored row-by-row (16 elements per row, 16 rows).

"Untilize" means converting from this face-based layout to standard row-major:
```
row 0:  [face0_row0 | face1_row0]   (32 elements)
row 1:  [face0_row1 | face1_row1]
...
row 15: [face0_row15 | face1_row15]
row 16: [face2_row0 | face3_row0]
...
row 31: [face2_row15 | face3_row15]
```

### 2. DEST Register
- The accumulator register where compute happens
- In half-sync mode: holds 4 FP32 tiles or 8 FP16 tiles
- In full-sync mode: holds 8 FP32 tiles or 16 FP16 tiles
- `DST_ACCUM_MODE` is defined when FP32/INT32/UINT32 are used

### 3. UnpackToDestMode
- `Default`: Unpacker sends data to **SrcA register**, then Math datacopy moves it to DEST. FP32 gets truncated to TF32 (19-bit mantissa) in this path.
- `UnpackToDestFp32`: Unpacker sends data **directly to DEST**, bypassing SrcA entirely. Preserves full 32-bit precision. **Incompatible** with unpacking to SrcA/SrcB — you can't do both at the same time.

### 4. Unpack-to-Dest Hardware Mechanism (CRITICAL FOR THIS FIX)
When `UnpackToDestFp32` is enabled, the hardware must be explicitly configured:
- **`Unpack_if_sel` register** = 1: Tells the unpacker hardware to route output to DEST instead of SrcA
- **DEST base address**: Must be set via `cfg_reg_rmw_tensix<Dest_cntx0_address_RMW>(addr)`
- **Address swizzle disabled**: `SETC16(SRCA_SET_Base_ADDR32, 0x0)`
- **Z-stride reconfigured**: Different stride for DEST vs SrcA addressing
- **UNPACR instruction encoding changes**: Different Z-inc bits, Dvalid=0 instead of 1
- **Semaphore sync**: Must call `wait_for_dest_available()` before and `unpack_to_dest_tile_done()` after
- **HW bug TEN-3868**: Must issue one normal SrcA unpack after the last to-dest unpack

All of this is implemented in `set_dst_write_addr()` and `unpack_to_dest_tile_done()` in `cunpack_common.h`. The slow untilize path does **none of it** — that's the bug.

### 5. Circular Buffers (CBs)
- `cb_wait_front(cb, n)`: Wait until `n` tiles are available to read from CB
- `cb_pop_front(cb, n)`: Release `n` tiles from the front of CB (done reading)
- `cb_reserve_back(cb, n)`: Reserve space for `n` tiles in CB output
- `cb_push_back(cb, n)`: Signal that `n` tiles have been written to CB

### 6. Three-Thread Model
Compute kernels run across three hardware threads simultaneously:
- **TRISC0 (Unpack)**: Moves data from L1 CBs into DEST
- **TRISC1 (Math)**: Does computation on data in DEST
- **TRISC2 (Pack)**: Moves data from DEST into L1 CBs

Code is tagged with `UNPACK((...))`, `MATH((...))`, `PACK((...))` macros.

### 7. SFPU (Special Function Processing Unit)
The SFPU sits on the Math thread and can do per-element operations on tiles in DEST. Operations like `exp`, `sqrt`, `relu`, etc. are SFPU ops. The key APIs:
- `tile_regs_acquire()` / `tile_regs_release()` - acquire/release DEST for SFPU use
- `copy_tile(cb, tile_index, dst_index)` - copy a tile from CB to DEST position
- `pack_tile(dst_index, cb)` - pack a tile from DEST position to output CB

---

## What to Try (In Order)

### Experiment 0: Reproduce the bug

Run the existing failing test to confirm the issue:
```bash
pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize.py::test_untilize_fp32_not_use_pack_untilize -v
```
This is marked `xfail` so it should "pass" (expected failure). Look at the actual precision difference.

Then run the diagnostic test for detailed measurements:
```bash
pytest learning-kernels/test_fp32_untilize.py::test_measure_precision_loss -v -s
```

### Experiment 1: Remove the Default override (sanity check)

Before doing any real work, test what happens if you just remove the override. This will almost certainly fail (for the reasons in the Root Cause Analysis), but it's important to confirm the failure mode.

In `untilize_single_core_program_factory.cpp`, change:
```cpp
unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::Default;
```
to:
```cpp
// Don't override - keep UnpackToDestFp32
// unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::Default;
```

Then build and run:
```bash
./build_metal.sh
pytest learning-kernels/test_fp32_untilize.py::test_slow_path_fp32_precision -v -s
```

**Expected outcome:** Crash, hang, or wrong results — because the slow untilize LLK doesn't configure unpack-to-dest registers. But record exactly what happens — the failure mode tells you what's missing.

### Experiment 2: Extend pack_untilize to wider FP32 (easiest possible fix)

The fast `pack_untilize` already chunks wide rows into blocks of `max_bct=4` tiles for FP32. The factory prevents using it when `num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH (8)`.

Try removing the width restriction:
```cpp
// In untilize_single_core_program_factory.cpp, change:
(a.dtype() == DataType::FLOAT32 && num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH)
// To:
false  // Always use pack_untilize for FP32
```

This tests whether the LLK `pack_untilize` hardware supports `full_ct_dim > 8`. The address offset math in `llk_pack_untilize.h` (line 144) uses `full_ct_dim`:
```cpp
const uint32_t output_addr_offset = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * (num_faces/2) * FACE_C_DIM);
```

If this works, it's the simplest fix — just change the constant or remove the check. **If it doesn't work**, you'll see wrong output addresses (garbled data) or a hang.

### Experiment 3: LLK modification — add unpack-to-dest support to untilize (THE MAIN APPROACH)

This is the real fix. Modify `_llk_unpack_untilize_pass_` in `llk_unpack_untilize.h` to support FP32 unpack-to-dest mode. Use `_llk_unpack_A_` in `llk_unpack_A.h` as your reference implementation.

#### Step 3a: Study the reference

Read `_llk_unpack_A_` (llk_unpack_A.h:236-291) end to end. Map out every operation it does when `unpack_to_dest=true`:

1. `set_dst_write_addr()` — register setup (before MOP)
2. `wait_for_dest_available()` — semaphore (before MOP)
3. MOP runs with different UNPACR encoding
4. `unpack_to_dest_tile_done()` — cleanup (after MOP)
5. Context switch

#### Step 3b: Understand the slow path's MOP

Read `_llk_unpack_untilize_pass_` (llk_unpack_untilize.h:88-169). It's more complex than `_llk_unpack_A_` because of the Y-stride rearrangement. Key structure:

```
For each of 16 rows:
    While tiles remain in this row:
        Run MOP (which does UNPACR with Y-stride)
        If we've filled 8 face-rows, signal data valid and reset
    Reset tile offset
```

The UNPACR instructions in the MOP (line 29-31):
```cpp
TTI_UNPACR(SrcA, 0b01000001, 0, 0, 0, 1, 0, RAREFYB_DISABLE, 0, 0, 0, 0, 1);
TTI_UNPACR(SrcA, 0b01000001, 0, 0, 0, 1, 0, RAREFYB_DISABLE, 0, 0, 0, 0, 1);
```

For unpack-to-dest, you'll likely need to change the Z-inc pattern and Dvalid flag in these UNPACR instructions, similar to how the fast path uses `0b00010001` instead of `0b1`.

#### Step 3c: Create a templated variant

Add an `unpack_to_dest` template parameter to `_llk_unpack_untilize_pass_`:

```cpp
template <bool first_pass = true, bool unpack_to_dest = false>
inline void _llk_unpack_untilize_pass_(const uint32_t base_address, const uint32_t block_tile_cols,
                                        const uint32_t unpack_src_format = 0, const uint32_t unpack_dst_format = 0) {
    // ... existing setup ...

    if constexpr (unpack_to_dest) {
        if (is_32bit_input(unpack_src_format, unpack_dst_format)) {
            set_dst_write_addr(unp_cfg_context, unpack_dst_format);
            wait_for_dest_available();
        }
    }

    // ... existing MOP code (with modified UNPACR encoding when unpack_to_dest) ...

    if constexpr (unpack_to_dest) {
        if (is_32bit_input(unpack_src_format, unpack_dst_format)) {
            unpack_to_dest_tile_done(unp_cfg_context);
        }
    }

    // ... existing cleanup ...
}
```

#### Step 3d: Wire it up

1. Update the LLK API layer to expose the new template parameter
2. Update `untilize_init` / `untilize_block` in `compute_kernel_api/untilize.h` to pass through the unpack-to-dest flag
3. Update (or create a variant of) the slow `untilize.cpp` compute kernel
4. Remove the `UnpackToDestMode::Default` override in the factory files
5. When using unpack-to-dest, the Math A2D datacopy may no longer be needed (data goes directly to DEST) — verify this

#### Step 3e: The big unknown

The Y-stride rearrangement (tile→row-major) is configured on the unpacker's read side. The `Unpack_if_sel` register controls the write destination (SrcA vs DEST). **These should be independent** — the rearrangement changes how data is read from L1, while the destination changes where it's written. But hardware can have unexpected interactions. This is the thing you can only verify by building and testing.

### Experiment 4: SFPU-based untilize kernel (fallback)

Only try this if Experiments 2 and 3 both fail. Write a new compute kernel that:

1. Unpacks tiles normally (no Y-stride) directly to DEST using `UnpackToDestFp32`
2. Uses SFPU or tile_move operations to rearrange data in DEST
3. Packs from DEST to output CB

This is the hardest approach and may not even be possible due to SFPU's tile-granularity limitations.

---

## Architecture Reference

```
                 ┌────────────────────────────────────────────┐
                 │              Tensix Core                    │
                 │                                            │
  L1 SRAM        │   ┌──────────┐                             │
 ┌──────────┐    │   │ Unpacker │──── Default ───> SrcA reg   │
 │ Input CB │────│──>│ (TRISC0) │                    │        │
 └──────────┘    │   │          │              Math A2D copy   │
                 │   │          │                    │        │
                 │   │          │                    ▼        │
                 │   │          │──── FP32 ────> DEST reg     │
                 │   └──────────┘  (Unpack_if_sel=1) (4 tiles)│
                 │                                   │        │
                 │                              ┌────┘        │
                 │                              ▼             │
                 │                        ┌──────────┐        │
 ┌──────────┐    │                        │  Packer  │        │
 │Output CB │<───│────────────────────────│ (TRISC2) │        │
 └──────────┘    │                        └──────────┘        │
                 └────────────────────────────────────────────┘

Default mode:    L1 → Unpacker → SrcA → Math A2D → DEST → Packer → L1
                 (FP32 truncated to TF32 at SrcA)

FP32 mode:       L1 → Unpacker → DEST directly → Packer → L1
                 (Full precision, but requires Unpack_if_sel=1 + register setup)

Current bug:     Slow untilize uses Default mode path even for FP32
                 because LLK never configures Unpack_if_sel or DEST addressing
```

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `PLAN.md` | This document |
| `test_fp32_untilize.py` | Test suite to measure your progress |
| `notes.md` | Your working notes (fill in as you go) |
