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

The override to `Default` is **intentional** - the slow `untilize.cpp` kernel apparently doesn't produce correct results with `UnpackToDestFp32`. Your job is to figure out why, and write a kernel that does.

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
| **Width limit** | 4 tiles FP32 / 8 tiles FP16 | None |
| **FP32 precision** | Preserved (after PR #33904) | Lost (forced TF32) |

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

### Tier 3: LLK implementation (the hardware-level code)

| File | What it is |
|------|-----------|
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_untilize.h` | How the unpacker does the rearrangement in the slow path |
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
- `Default`: Unpacker converts FP32 → TF32 (19-bit mantissa) when loading into DEST. This is faster and is the normal mode.
- `UnpackToDestFp32`: Unpacker preserves full 32-bit precision in DEST. Required for exact FP32 output.

### 4. Circular Buffers (CBs)
- `cb_wait_front(cb, n)`: Wait until `n` tiles are available to read from CB
- `cb_pop_front(cb, n)`: Release `n` tiles from the front of CB (done reading)
- `cb_reserve_back(cb, n)`: Reserve space for `n` tiles in CB output
- `cb_push_back(cb, n)`: Signal that `n` tiles have been written to CB

### 5. Three-Thread Model
Compute kernels run across three hardware threads simultaneously:
- **TRISC0 (Unpack)**: Moves data from L1 CBs into DEST
- **TRISC1 (Math)**: Does computation on data in DEST
- **TRISC2 (Pack)**: Moves data from DEST into L1 CBs

Code is tagged with `UNPACK((...))`, `MATH((...))`, `PACK((...))` macros.

### 6. SFPU (Special Function Processing Unit)
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

Then run the tests in `learning-kernels/test_fp32_untilize.py` (provided below) for more detailed measurements.

### Experiment 1: Try the naive fix first

Before writing a new kernel, test the obvious hypothesis: **what if you just remove the override to Default?**

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
pytest learning-kernels/test_fp32_untilize.py -v
```

**Possible outcomes:**
1. **It works** - The override was overly conservative. You just need to remove it from all 11 factories. Done.
2. **It crashes/hangs** - The slow untilize LLK genuinely can't handle FP32 DEST mode. You need a new kernel.
3. **Wrong results** - The unpacker Y-stride math is wrong for FP32 element size. May need LLK fix.

### Experiment 2: Understand why it fails (if Experiment 1 fails)

If enabling `UnpackToDestFp32` on the slow path produces wrong results, you need to understand WHERE it breaks. The slow path has three stages:

1. **Unpack** (`llk_unpack_untilize`): Uses Y-stride addressing. Check if the stride calculation accounts for FP32 element size. Look at `_llk_unpack_untilize_init_` line 64:
   ```cpp
   const uint32_t unpA_ch1_x_stride = (unpack_dst_format & 0x3) == (uint32_t)DataFormat::Float32 ? 4 : ...
   ```
   This looks like it DOES handle FP32. But does the DEST addressing work correctly in FP32 mode?

2. **Math** (`llk_math_eltwise_unary_datacopy`): This is an A2D datacopy. Should be format-agnostic.

3. **Pack** (`llk_pack`): Standard pack from DEST. Should work with FP32 DEST if configured correctly.

### Experiment 3: Write an SFPU-based untilize kernel

If the slow path fundamentally can't do FP32, write a new compute kernel. The approach:

1. **Unpack tiles normally** (not with untilize stride) into DEST using `UnpackToDestFp32`
2. **Use SFPU or tile_move operations** to rearrange the data in DEST from tile format to row-major
3. **Pack normally** from DEST to output CB

The challenge is step 2 - how do you rearrange within DEST? Options to explore:
- `copy_tile` with different src/dst indices
- SFPU operations that can permute data
- Process one tile at a time: unpack → pack with custom addressing

### Experiment 4: Extend pack_untilize to wider FP32

Alternative approach: the fast `pack_untilize` already chunks wide rows into blocks of 4 tiles for FP32. It uses `block_ct_dim = 4` and `full_ct_dim = total_width`. The factory prevents this when `full_ct_dim > 8`.

Try removing the width restriction:
```cpp
// In the factory, change:
(a.dtype() == DataType::FLOAT32 && num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH)
// To:
false  // Always use pack_untilize for FP32
```

This tests whether the LLK `pack_untilize` hardware can handle `full_ct_dim > 8`. The MOP configuration and address offset calculation in `llk_pack_untilize.h` use `full_ct_dim` for stride math. If the hardware supports it, this is the simplest fix.

---

## Architecture Reference

```
                    ┌─────────────────────────────────────┐
                    │           Tensix Core                │
                    │                                      │
   L1 SRAM          │   ┌──────────┐                      │
  ┌──────────┐      │   │ Unpacker │                      │
  │ Input CB │──────│──>│ (TRISC0) │──┐                   │
  └──────────┘      │   └──────────┘  │                   │
                    │                  ▼                   │
                    │            ┌──────────┐              │
                    │            │   DEST   │              │
                    │            │ Register │              │
                    │            │ (4 FP32  │              │
                    │            │  tiles)  │              │
                    │            └──────────┘              │
                    │           ┌──┘    └──┐              │
                    │           ▼          ▼              │
                    │   ┌──────────┐  ┌──────────┐       │
                    │   │  Math    │  │   SFPU   │       │
                    │   │ (TRISC1) │  │          │       │
                    │   └──────────┘  └──────────┘       │
                    │           └──┐    ┌──┘              │
                    │              ▼    ▼                  │
                    │            ┌──────────┐              │
  ┌──────────┐      │            │  Packer  │              │
  │Output CB │<─────│────────────│ (TRISC2) │              │
  └──────────┘      │            └──────────┘              │
                    └─────────────────────────────────────┘

Fast path: Unpacker loads tile-order → Packer reads DEST with ADDR_MOD → row-major output
Slow path: Unpacker loads with Y-stride (row-major into DEST) → Packer writes linearly
```

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `PLAN.md` | This document |
| `test_fp32_untilize.py` | Test suite to measure your progress |
| `notes.md` | Your working notes (fill in as you go) |
