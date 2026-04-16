# BH Fast-Tilize Conv2d Integration — Status Report

## Problem Summary

BH fast-tilize works correctly for standalone tilize operations (PCC=1.0) but produces incorrect matmul results (PCC=0.824) when used in the conv2d compute kernel's tilize→matmul flow. The issue only manifests with SyncHalf DEST mode and is specific to the fast-tilize→matmul transition.

## Test Results

| Test | Fast-Tilize Active? | PCC | Status |
|---|---|---|---|
| Standalone tilize (32x32, 320x384, 128x7328) | Yes | 1.0 | PASS |
| Conv2d 16ch HEIGHT_SHARDED bf16 filter=3 | Yes (in0_block_w=2) | 0.824 | FAIL |
| Conv2d 16ch HEIGHT_SHARDED bf16 filter=5 | Yes (in0_block_w=?) | 0.465 | FAIL |
| Conv2d 16ch HEIGHT_SHARDED bfp8 filter=3 | Yes | 0.511 | FAIL |
| Conv2d 128ch BLOCK_SHARDED bf16 | No (SyncFull) | 0.999 | PASS |
| Conv2d 353ch WIDTH_SHARDED bf16 | No (SyncFull) | 0.999 | PASS |

Key observation: 128ch and 353ch tests use **SyncFull** mode which disables fast-tilize via `can_use_fast_tilize()` returning false. They NEVER use fast-tilize. The 16ch HEIGHT_SHARDED test is the only one that uses SyncHalf AND fast-tilize.

## Root Cause Analysis

### What works
- Fast-tilize L1 output is **bit-identical** to standard tilize (confirmed via DPRINT on silicon, core 1,1)
- All config registers are **identical** between fast-tilize ON/OFF at matmul entry (confirmed via DPRINT + tt-exalens)
- Standalone tilize roundtrip (tilize + untilize) gives **PCC=1.0** for any width including non-4-aligned

### What we found
1. **`in0_block_w=2`** (not 5 as initially assumed). The conv2d op decomposes 16ch×3×3=144 datums into blocks of 2 tiles.
2. **`dest_offset_id` left at 1 after fast-tilize block** (should be 0). With `in0_block_w=2`: 1 fill → 1 section_done → odd toggle. Standard tilize does 0 section_dones.
3. **Fixing `dest_offset_id` to 0 did NOT fix PCC** — the fixup was verified via DPRINT but PCC remained 0.824.
4. **tt-exalens full tensix state dump** shows NO differences in config registers, DEST contents, SrcA/SrcB, thread config, semaphores, or address counters between ON and OFF. Only PACK GPR transient state (output_addr, output_addr_offset) differs, which gets overwritten by the matmul.
5. **ttsim is unreliable** for this specific issue — ttsim has a DEST offset execution ordering issue where MOVA2D stalls on SrcA validity, and the SrcB data divergence on ttsim is caused by corrupted tilize L1 output (from wrong DEST half reads), not from the real silicon issue.

### BH-specific differences from WH
- BH fast-tilize uses **DEST remap** (`remap_addrs`, `swizzle_32b`) — WH doesn't
- BH uninit calls `_llk_math_reconfig_remap_(false)` with `tensix_sync()` + semaphore drain — WH doesn't
- BH `unit_dim=4` (4 tiles per DEST half) vs WH `unit_dim=2` (2 tiles, 8 per DEST)
- BH fast-tilize uses `set_dst_write_addr<SrcRegs>` (SETC16) vs WH uses `<DestReg>` (mailbox)

## Fixes Applied (confirmed working)

1. **SrcB reconfig guard** — `#ifndef ARCH_BLACKHOLE` in `tilize_helpers.inl` prevents `reconfig_data_format_srcb(input_cb)` on BH (BH fast-tilize only uses SrcA)
2. **`fast_tilize_init_with_dt`** — BH path only reconfigures SrcA, not SrcB
3. **TileDescriptor X_dim restore** — `cfg_reg_rmw_tensix` in uninit forces X_dim=1024 (UNPACR clears it)
4. **Pack strides** — uninit calls `set_packer_strides<false,false>()` for correct Z=512 (was saving/restoring tilize strides Z=1024)
5. **`bh_used_fast_tilize` flag** — ensures init/uninit/block use consistent fast-tilize vs standard-tilize path
6. **`dest_offset_id` fixup** — compensates for odd section_done count when `n_fills` is odd

## What was tried and didn't fix PCC

| Attempt | Result |
|---|---|
| Reset `dest_offset_id` to 0 + SETC16 HW register write | PCC=0.824 unchanged |
| Remove `_llk_math_reconfig_remap_` tensix_sync/semaphore drain | PCC=0.824 unchanged |
| Leave DEST remap enabled (skip disable) | PCC=0.824 unchanged |
| CLR_ALL (clear entire DEST) after tilize | PCC=0.824 unchanged |
| Full `mm_block_init` (with pack_dest_init) after tilize | PCC=0.533 (worse — resets SyncHalf) |
| Full unpack+math reinit with hw_configure | PCC=0.720 (worse) |
| WH-style DEST management (section_done per fill) | PCC=0.824 unchanged |
| Reset all UNP_A address counters + carriage returns | PCC=0.824 unchanged |
| Various fill decompositions ([2,3], [3,2], 4-only) | PCC=0.824 unchanged |

## Current Gate

Temporarily restricted to `full_dim >= 4 && full_dim % 4 == 0` which makes all conv2d tests pass (16ch falls back to standard tilize, 128ch/353ch already use SyncFull). Standalone tilize works for all widths.

## Open Question

The tt-exalens state dump shows **zero observable differences** in hardware state between fast-tilize ON and OFF at the matmul entry point. Every register, every DEST row, every SrcA/SrcB value is identical. Yet PCC=0.824. This suggests:

1. A transient coprocessor pipeline effect not captured by register dumps
2. A race condition between threads that affects instruction execution order
3. Some hardware state not accessible via debug bus/tt-exalens

## Files Modified

### tt-metal (Metal API layer)
- `tt_metal/hw/inc/api/compute/tilize.h` — BH fast_tilize_init/block/uninit with `bh_used_fast_tilize` flag, WH-style fill decomposition, dest_offset_id fixup
- `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl` — `#ifndef ARCH_BLACKHOLE` guard on SrcB reconfig

### tt-llk (LLK layer)
- `tt_llk_blackhole/llk_lib/llk_unpack_tilize.h` — Save/restore TileDescriptor word 0, full counter+CR reset in uninit
- `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h` — dest_offset_id handling in uninit (no reset — preserve SyncHalf alternation)
- `tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h` — `set_packer_strides<false,false>()` in uninit, pack_src_format parameter

### BH Metal API wrappers
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_tilize_api.h` — Base address via SETDMAREG+WRCFG
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_datacopy_api.h` — Fast-tilize math API
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h` — Fast-tilize pack API with pack_src_format

### ttsim
- `ttsim-private/src/tensix.cpp` — Multiple debug traces (MVMUL config dump, SrcB write watchpoint, UNPACR trace)
- `ttsim-private/src/tile.cpp` — Disabled SEMPOST spam
- `ttsim-private/src/libttsim.cpp` — Instruction dispatch traces

## Next Steps

1. **Fix ttsim DEST offset issue** — MOVA2D stalls mask the real SrcB behavior, making ttsim unreliable
2. **Pipeline-level debugging** — Use tt-exalens debug bus or waveform capture to observe transient coprocessor state
3. **Minimal repro** — Create a standalone kernel (no conv2d op) that does tilize→matmul to isolate the issue from conv2d's complexity
4. **Try without DEST remap** — The remap enable/disable cycle is BH-specific. If we can make pack work without remap (different DEST stride pattern), it eliminates the BH-specific code path entirely
