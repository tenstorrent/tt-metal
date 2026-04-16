# BH Fast-Tilize PACR_FLUSH Overflow Fix Plan

## Hard Requirement

**The LLK must not write outside the provided output buffer.** No caller-side padding,
no Metal-level workarounds. The fix must be entirely within `llk_pack_fast_tilize.h` and
the LLK test infrastructure. The pack block function receives an L1 address and a tile
count — it must write exactly that many tiles and zero bytes beyond.

## Problem Statement

The flat-output pack MOP in `llk_pack_fast_tilize.h` emits a `PACR_FLUSH` (ZeroWrite=1,
Last=1) as `last_outer_loop_instr`. This writes **128 bytes** past the valid tile data
into adjacent L1 memory after every unit. In conv2d, this corrupts the matmul
partial-results buffer, causing accumulation failures starting at kernel_h=3.

## Root Cause

**Standard pack** has `outerloop = num_faces * num_tiles` and `innerloop = face_r_dim / 4`.
Its `last_outer_loop_instr` is a normal PACR (Last=1) that writes the **last valid row**
(row 63) of the tile. No overflow because the write is within bounds.

**Fast-tilize flat MOP** has `outerloop = 1` and `innerloop = unit_dim`. The 16-PACR tile
replay has no Last=1. PACR_FLUSH is appended as `last_outer_loop_instr` to close the tile.
PACR_FLUSH writes an **extra row** (128 bytes across 4 faces) past the last valid tile.

**BFP fast-tilize MOP** has `outerloop = 1`, `innerloop = 1` and Last=1 **inside the replay
buffer** on the 16th PACR. No PACR_FLUSH, no overflow. Runs one MOP per tile with STALLWAIT
between them.

## Constraints (proven by silicon testing)

1. **PACR_FLUSH is required on real silicon** — removing it breaks accuracy for ct_dim >= 3.
   The pack hardware needs Last=1 to close its internal tile state.
2. **Last=1 in the flat replay breaks multi-tile addressing** — tested: tile data reorders
   (tile 0 gets tile 1's columns). ADDR_MOD_1 (y_clr + z_clr) on the Last PACR interacts
   badly with the MOP's ADDRCRZW increment between inner iterations.
3. **`set_last_inner_loop_instr` is ignored when outerloop=1** — the hardware priority rule:
   `if (last_inner && last_outer)` uses `m_loop0_last_instr`, not `m_loop1_last_instr`.
   With outerloop=1, every "last inner" is also "last outer", so `last_inner_loop_instr`
   never fires.
4. **Per-tile STALLWAIT disrupts SyncHalf timing** — BFP-style per-tile execution with stalls
   between MOP runs broke Metal conv2d kernel_h=2 in earlier testing.

## Measured Overflow

| Output Format | Overflow per unit | Overflows per row | Total overflow |
|---|---|---|---|
| Float16_b | 128 bytes (64 uint16) | 1 (last unit only) | 128 bytes |
| Bfp8_b | 0 | 0 | 0 |
| Bfp4_b | 0 | 0 | 0 |

For multi-unit rows, intermediate PACR_FLUSH overflow is overwritten by the next unit's
data. Only the **last unit's** PACR_FLUSH persists into adjacent L1 (128 bytes total).

---

## Fix Options

### Option A: Per-tile MOP execution (BFP-style for all formats)

Eliminate the flat MOP entirely. Use the BFP MOP config (outerloop=1, innerloop=1, BFP
replay with Last=1) for **all** output formats. Execute one MOP `run()` per tile from the
block function.

The BFP path currently uses `TTI_STALLWAIT(PACK | THCON)` between MOP runs to ensure the
REG2FLOP L1_Dest_addr update completes before the next tile's first PACR reads it.
Try without STALLWAIT first — the RISC-V loop overhead (branch, counter increment,
SETADCZW) may provide enough natural latency. If not, add minimal stall.

**Changes to `tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h`:**

1. **Remove** `_llk_pack_fast_tilize_mop_config_flat_()` and the `PACR_FLUSH` constant.

2. **Init** (`_llk_pack_fast_tilize_init_`): always call `_llk_pack_fast_tilize_mop_config_bfp_()`.
   Remove the `if (fast_tilize_bfp_mode)` branch for MOP selection. Change
   `OUTPUT_ADDR_OFFSET` to always be **per-tile** (`tile_l1_size`) instead of per-unit
   (`unit_dim * tile_l1_size`) — the current code at line 241 sets
   `pacr_l1_size = fast_tilize_bfp_mode ? tile_l1_size : (unit_dim * tile_l1_size)`;
   this becomes unconditionally `tile_l1_size`.

3. **Reinit** (`_llk_pack_fast_tilize_reinit_unit_dim_`): the current flat branch at
   line 296 reprograms `OUTPUT_ADDR_OFFSET = new_unit_dim * tile_l1_size` and calls
   `_llk_pack_fast_tilize_mop_config_flat_`. With the flat MOP gone, reinit becomes a
   no-op for all formats (BFP MOP is not parameterized by unit_dim, and
   `OUTPUT_ADDR_OFFSET` is always per-tile).

4. **Block function** (`_llk_pack_fast_tilize_block_`): remove the `if (fast_tilize_bfp_mode)`
   branch. Single path for all formats:
   ```cpp
   for (uint32_t u = 0; u < num_units; u++) {
       TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
       for (uint32_t t = 0; t < unit_dim; t++) {
           ckernel::ckernel_template::run();  // BFP replay (Last=1)
           // Try without STALLWAIT. If accuracy fails, add:
           // TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
       }
   }
   ```

5. **Uninit** (`_llk_pack_fast_tilize_uninit_`): remove `fast_tilize_bfp_mode` flag and
   all conditional logic. Single cleanup path.

**Pros:** Eliminates PACR_FLUSH entirely. Zero overflow. Clean architecture (one MOP, one
block path, one init/uninit for all formats). BFP path is proven working.
**Cons:** Risk of SyncHalf timing regression without STALLWAIT. Must verify with Metal
conv2d. If timing fails, stalls reduce perf.

**Effort:** Medium. Straightforward code deletion + unification.

---

### Option B: Restructure flat MOP with outerloop=unit_dim

Change the flat MOP from `outerloop=1, innerloop=unit_dim` to
`outerloop=unit_dim, innerloop=1`. Use the BFP replay (Last=1). Keep multi-tile MOP
execution (no per-tile RISC-V loop).

With outerloop>1 and innerloop=1, `last_outer_loop_instr` fires only on the **last outer
iteration** (last tile). `last_inner_loop_instr` fires on all other iterations.

MOP structure:
- `loop_op0`: BFP replay (16 PACRs, Last=1 closes each tile)
- `loop_op1` / `last_inner_loop_instr`: ADDRCRZW (advance DEST W counter between tiles)
- `last_outer_loop_instr`: ADDRCRZW or NOP (no PACR_FLUSH)
- `start_op`: SETADCZW (reset Z/W)
- `end_ops`: ADDDMAREG + REG2FLOP (L1 address advance)

**Critical question:** Does the pack hardware auto-advance L1_Dest_addr after Last=1?

Each tile's replay ends with Last=1. The next tile's first PACR needs to write to the
correct L1 offset. `end_ops` (ADDDMAREG + REG2FLOP) only run **once** after the entire
loop, not per tile. So L1_Dest_addr is not software-advanced between tiles.

If the hardware auto-advances L1_Dest_addr by tile_size on Last=1, tiles land at the
correct L1 offsets. If not, all tiles overwrite the same L1 location.

**How to test — 2-tile memory-layout experiment (not register probe):**

A register probe (reading L1_Dest_addr after BFP MOP) gives a false signal: the current
BFP MOP already advances L1_Dest_addr in software via end_ops (ADDDMAREG + REG2FLOP at
`llk_pack_fast_tilize.h:203`). Cannot distinguish HW auto-advance from SW advance.

Instead: write a **special probe MOP with no end_ops** that packs 2 tiles using the BFP
replay, then read both tile locations in L1:
- If tile 0 and tile 1 contain different correct data → HW auto-advanced
- If tile 1 is zeros or duplicate of tile 0 → HW does NOT auto-advance

**Changes to `tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h`** (if HW auto-advances):

1. **New MOP config** replaces both flat and BFP configs:
   ```cpp
   inline void _llk_pack_fast_tilize_mop_config_(const uint32_t unit_dim) {
       ckernel_template tmp(
           unit_dim,  // outerloop: one iteration per tile
           1,         // innerloop: single replay per tile
           replay_insn(REPLAY_BFP_TILE_OFFSET, REPLAY_BFP_TILE_LEN),
           ADDRCRZW(PAC, 0, 0, 1, 0, 0b0010));  // loop_op1: advance W
       tmp.set_start_op(SETADCZW(PAC, 0, 0, 0, 0, 0b0011));
       // No PACR_FLUSH. No end_ops for L1 advance (HW handles it).
       // end_ops still needed for OUTPUT_ADDR GPR bookkeeping if block
       // function uses it for multi-unit addressing.
       tmp.set_end_ops(
           ADDDMAREG(OUTPUT_ADDR, OUTPUT_ADDR, OUTPUT_ADDR_OFFSET),
           REG2FLOP(L1_Dest_addr, OUTPUT_ADDR));
       tmp.program();
   }
   ```
   `OUTPUT_ADDR_OFFSET` = `unit_dim * tile_l1_size` (end_ops advance by full unit for
   the next block call, matching the HW auto-advance that happened per-tile).

2. **Reinit** (`_llk_pack_fast_tilize_reinit_unit_dim_`): reprogram the unified MOP with
   new `unit_dim` (changes outerloop count). Update `OUTPUT_ADDR_OFFSET` to
   `new_unit_dim * tile_l1_size`.

3. **Block function**: single path, one `run()` per unit (MOP handles all tiles internally):
   ```cpp
   for (uint32_t u = 0; u < num_units; u++) {
       ckernel::ckernel_template::run();
   }
   ```
   No per-tile loop, no STALLWAIT. Same perf as current flat MOP.

4. **Init**: `OUTPUT_ADDR_OFFSET = unit_dim * tile_l1_size` (per-unit, same as current flat).

**Pros:** Zero overflow. No per-tile RISC-V overhead. No stalls. Single MOP for all formats.
Same perf as current flat MOP.
**Cons:** Depends on unverified HW auto-advance behavior. Falls back to Option A if
HW doesn't auto-advance.

**Effort:** Medium-high. HW probe + LLK changes + full test sweep.

---

## Recommended Path

### Step 1: Probe HW auto-advance (gates Option B)

Write a 2-tile memory-layout experiment on silicon. **Do not use a register probe** — the
existing BFP MOP's end_ops (ADDDMAREG + REG2FLOP at `llk_pack_fast_tilize.h:203`) already
advance L1_Dest_addr in software, making a register read ambiguous.

**Experiment design:**
1. Create a probe MOP: `outerloop=2, innerloop=1`, BFP replay, ADDRCRZW as loop_op1,
   **no end_ops** (no ADDDMAREG, no REG2FLOP).
2. Load two different tiles into DEST (tile 0 = constant A, tile 1 = constant B).
3. Run the probe MOP once. Read L1 at tile 0 and tile 1 offsets.
4. If L1[tile_0] == A and L1[tile_1] == B → HW auto-advances. Proceed with Option B.
5. If L1[tile_1] is zeros or == A → HW does NOT auto-advance. Proceed with Option A.

**Run per output format.** Tile L1 size varies by `pack_dst_format`
(`GET_L1_HEADERLESS_TILE_SIZE` at `llk_pack_fast_tilize.h:237`). Auto-advance amount
(if any) may be format-dependent. Probe must cover at minimum Float16_b and Bfp8_b.
If auto-advance works for Float16_b but not Bfp8_b, Option B can only replace the flat
path — the BFP path stays as-is (it already works without overflow).

This can be a standalone LLK test in
`tt_metal/third_party/tt_llk/tests/sources/` + `tt_metal/third_party/tt_llk/tests/python_tests/`.

### Step 2: Implement the fix

**If Option B (restructured MOP, HW auto-advance confirmed):**
1. Replace `_llk_pack_fast_tilize_mop_config_flat_` and `_llk_pack_fast_tilize_mop_config_bfp_`
   with single `_llk_pack_fast_tilize_mop_config_` taking `unit_dim`.
2. Remove `PACR_FLUSH` constant and `fast_tilize_bfp_mode` flag.
3. Init: `OUTPUT_ADDR_OFFSET = unit_dim * tile_l1_size` (per-unit, HW handles per-tile).
4. Reinit: reprogram MOP outerloop + `OUTPUT_ADDR_OFFSET` for new unit_dim.
5. Block: single `run()` per unit, no per-tile loop, no STALLWAIT.
6. Uninit: remove BFP-mode conditional, single cleanup path.

**If Option A (per-tile execution, HW does NOT auto-advance):**
1. Remove `_llk_pack_fast_tilize_mop_config_flat_` and `PACR_FLUSH` constant.
2. Init: always call `_llk_pack_fast_tilize_mop_config_bfp_()`.
   Change `OUTPUT_ADDR_OFFSET` to **per-tile** unconditionally (`tile_l1_size` instead of
   `unit_dim * tile_l1_size` at line 241).
3. Reinit: becomes no-op (BFP MOP not parameterized by unit_dim, `OUTPUT_ADDR_OFFSET` is
   always per-tile).
4. Block: single path — per-tile `run()` loop for all formats. Try without STALLWAIT first.
   If accuracy fails, add `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON)`.
5. Uninit: remove `fast_tilize_bfp_mode` conditional, single cleanup path.

### Step 3: Validate

Test tiers (each must pass before proceeding to the next):

1. **LLK overflow guard** — `tt_metal/third_party/tt_llk/tests/python_tests/test_fast_tilize_full.py::test_fast_tilize_overflow_guard`:
   0 corruption for Float16_b, Bfp8_b (silicon, individual runs).
   Current guard matrix covers (1,1) (1,2) (1,3) (1,4) (2,2) (2,4) (4,4).
   **Before merging, expand guard dimensions to include width tails: (1,5) (1,6) (1,7)
   (1,8) (1,9).** These exercise `reinit_unit_dim` and "last unit" boundary behavior
   (decompose_row produces 2+3, 2+4→err, 3+4, 4+4, 4+2+3 splits) that the current
   matrix does not cover.
2. **LLK accuracy** — `tt_metal/third_party/tt_llk/tests/python_tests/test_fast_tilize_full.py::test_fast_tilize_full`:
   all 18 Float16_b dims + Bfp8_b + Float32 (silicon, batch)
3. **LLK tilize+matmul repro** — `tt_metal/third_party/tt_llk/tests/python_tests/test_tilize_matmul_repro.py`:
   fast_tilize_matmul_accum tests with reload pattern (tests 5-8)
4. **Metal tilize op** — `tests/ttnn/unit_tests/operations/data_movement/test_tilize.py`
5. **Metal conv2d** — `tests/ttnn/unit_tests/operations/conv/test_conv2d_ones.py` kernel_h=1,2,3
6. **Perf** — `tt_metal/third_party/tt_llk/tests/python_tests/perf_fast_tilize_full.py`:
   run `test_perf_fast_tilize` (L1_TO_L1 + PACK_ISOLATE) for Float16_b ct_dim=2,4,8
   before and after the change. No regression > 5% in cycles/tile.

### Acceptance Criteria

- Overflow guard tests: **0 corrupted** for all guarded dimensions (including expanded
  width-tail set) and all output formats on silicon
- All accuracy tests pass on silicon in batch mode
- conv2d passes for kernel_h=1,2,3 on silicon
- No perf regression > 5% in `perf_fast_tilize_full.py` cycles/tile (L1_TO_L1 + PACK_ISOLATE)

## Files to Change

### Option A (per-tile execution)
| File | Change |
|---|---|
| `tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h` | Remove flat MOP config + PACR_FLUSH. Use BFP MOP for all. `OUTPUT_ADDR_OFFSET` → per-tile. Reinit → no-op. Block → per-tile `run()`. Uninit → remove bfp_mode conditional. |
| `tt_metal/third_party/tt_llk/tests/sources/fast_tilize_bh_test.cpp` | Update pack thread if block API signature changes |
| `tt_metal/third_party/tt_llk/tests/python_tests/test_fast_tilize_full.py` | Verify overflow guard shows 0 on silicon |

### Option B (restructured MOP)
| File | Change |
|---|---|
| `tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h` | Replace both MOP configs with unified `_llk_pack_fast_tilize_mop_config_(unit_dim)`. Remove PACR_FLUSH + bfp_mode. Reinit reprograms MOP outerloop + OFFSET. Block → one `run()` per unit. |
| `tt_metal/third_party/tt_llk/tests/sources/fast_tilize_bh_test.cpp` | Update pack thread if block API signature changes |
| `tt_metal/third_party/tt_llk/tests/python_tests/test_fast_tilize_full.py` | Verify overflow guard shows 0 on silicon |
| `tt_metal/third_party/tt_llk/tests/sources/` + `tests/python_tests/` | New probe test for HW auto-advance experiment |
