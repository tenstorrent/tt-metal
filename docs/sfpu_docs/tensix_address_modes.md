# Tensix Address Modes

Address modes (address modifiers) are a hardware indirection mechanism that
automatically increments or resets the **SrcA**, **SrcB**, and **Dest**
Read/Write Counters (RWCs) after each math instruction executes. This removes
the need for software to manually update register pointers between operations.

## Overview

There are **8 address modifier slots** (`ADDR_MOD_0`..`ADDR_MOD_7`).
Each is pre-configured before the math loop begins. Every math instruction
carries a 2-bit `addr_mod` field that selects which slot to apply after the
instruction completes.

The 8 slots are divided into two **sets** of 4:

```
    addr_mod_t slots
    +------+------+------+------+------+------+------+------+
    |  0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |
    +------+------+------+------+------+------+------+------+
    |<---- SET 0 (base=0) ---->||<---- SET 1 (base=1) ----->|
    |   FPU default set        ||   SFPU default set (WH)   |
    +------+------+------+------+------+------+------+------+

    A 2-bit addr_mod field in each instruction selects index 0..3
    within the ACTIVE set. Which set is active depends on the
    ADDR_MOD_SET_Base config register bit:

        base=0  -->  2-bit index 0..3 maps to slots 0..3
        base=1  -->  2-bit index 0..3 maps to slots 4..7

    Math instruction example:

        TT_OP_ELWADD(clr_src, acc, bcast, addr_mod, ...)
                                          ^^^^^^^^
                                          2-bit index
                                          selects slot within active set
```

### Wormhole vs Blackhole: ADDR_MOD_SET_Base

The two architectures handle the set switching differently, which affects
how FPU and SFPU address modifiers coexist without collision:

```
    +===============+==============================================+
    |               |  How ADDR_MOD_SET_Base is managed            |
    +===============+==============================================+
    |               |                                              |
    |  Wormhole     |  Explicit helpers in cmath_common.h:         |
    |  (WH B0)      |    math::set_addr_mod_base()                 |
    |               |      -> TTI_SETC16(ADDR_MOD_SET_Base, 1)     |
    |               |    math::clear_addr_mod_base()               |
    |               |      -> TTI_SETC16(ADDR_MOD_SET_Base, 0)     |
    |               |                                              |
    |               |  Called at SFPU start/done boundaries:        |
    |               |    _sfpu_start_  -> set_addr_mod_base()      |
    |               |    _sfpu_done_   -> stall; clear_addr_mod_   |
    |               |                     base()                   |
    |               |                                              |
    +---------------+----------------------------------------------+
    |               |                                              |
    |  Blackhole    |  No set_addr_mod_base / clear_addr_mod_base  |
    |  (BH)         |  helpers in cmath_common.h.                  |
    |               |                                              |
    |               |  Unary/binary SFPU: no base switching at all |
    |               |    _sfpu_start_  -> (no base switch)         |
    |               |    _sfpu_done_   -> (no base switch)         |
    |               |                                              |
    |               |  Ternary SFPU: uses raw register write       |
    |               |    _sfpu_done_   -> TTI_SETC16(2, 0)         |
    |               |    (hardcoded register address, no helper)   |
    |               |                                              |
    +===============+==============================================+
```

**Why this matters:**

```
    Wormhole SFPU lifecycle               Blackhole SFPU lifecycle
    ========================              ========================

    _sfpu_start_:                         _sfpu_start_:
      set_dst_write_addr(idx)               set_dst_write_addr(idx)
      set_addr_mod_base()  <-- SET 1        (no base switch)
      STALL_SFPU                            STALL_SFPU

    ... SFPU uses ADDR_MOD_4..7 ...       ... SFPU uses ADDR_MOD_0..7 ...
    ... (2-bit index 3 -> slot 7)  ...    ... (slots accessed directly) ...

    _sfpu_done_:                          _sfpu_done_:
      clear_dst_reg_addr()                  clear_dst_reg_addr()
      STALL_CFG, WAIT_SFPU                  (no stall, no base switch)
      clear_addr_mod_base() <-- SET 0

    On WH, base switching prevents        On BH, SFPU and FPU share the
    SFPU addr mods (4..7) from            same slot namespace. The kernel
    colliding with FPU addr mods          must avoid slot conflicts by
    (0..3). They live in separate          convention (SFPU uses 6,7;
    halves of the same table.             FPU uses 0..3).
```

### Practical Consequence for Kernel Authors

```
    Architecture    SFPU configures       FPU configures    Conflict?
    ============    ===============       ==============    =========
    Wormhole        ADDR_MOD_7 (maps      ADDR_MOD_0..3    No - separate
                    to slot 7 via                           sets via
                    Set 1 base bit)                         base bit

    Blackhole       ADDR_MOD_6, _7        ADDR_MOD_0..3    No - different
                    (absolute slot                          slot indices
                    indices)                                (same set)

    Both use ADDR_MOD_7 for SFPU, but WH maps it via set_addr_mod_base()
    while BH accesses slot 7 directly. The code comment in both versions
    says: "this kernel is typically used in conjunction with A2D, which is
    using ADDR_MOD_0 and ADDR_MOD_2, so use one that doesn't conflict!"
```


## The `addr_mod_t` Structure

Defined identically in both architectures:
- `tt_llk_wormhole_b0/common/inc/ckernel_addrmod.h`
- `tt_llk_blackhole/common/inc/ckernel_addrmod.h`

```
    addr_mod_t
    +--------------------------------------------------+
    |  srca      : addr_mod_src_t      (SrcA control)  |
    |  srcb      : addr_mod_src_t      (SrcB control)  |
    |  dest      : addr_mod_dest_t     (Dest control)  |
    |  fidelity  : addr_mod_fidelity_t (phase control) |
    |  bias      : addr_mod_bias_t     (bias control)  |
    +--------------------------------------------------+
```

### Per-Register Fields

```
    addr_mod_src_t  (SrcA, SrcB)        addr_mod_dest_t  (Dest)
    +---------------------------+       +-----------------------------------+
    | incr : 6 bits  (0..63)   |       | incr   : 10 bits  (0..1023)      |
    | clr  : 1 bit   (reset)   |       | clr    : 1 bit    (reset)        |
    | cr   : 1 bit   (carry)   |       | cr     : 1 bit    (carry/reset)  |
    +---------------------------+       | c_to_cr: 1 bit    (save counter) |
                                        +-----------------------------------+

    addr_mod_fidelity_t                 addr_mod_bias_t
    +---------------------------+       +---------------------------+
    | incr : 2 bits  (0..3)    |       | incr : 4 bits  (0..15)   |
    | clr  : 1 bit   (reset)   |       | clr  : 1 bit   (reset)   |
    +---------------------------+       +---------------------------+
```

**Field semantics:**

| Field    | Effect when set                                               |
|----------|---------------------------------------------------------------|
| `incr`   | Advance the register row pointer by this many rows            |
| `clr`    | Reset the row pointer to 0 (overrides `incr`)                 |
| `cr`     | Use carry/reset: wrap around when pointer hits the boundary   |
| `c_to_cr`| (Dest only) Copy current counter into carry register first    |

### Hardware Register Encoding (`set()` method)

The `set()` method writes two 16-bit config registers per slot. The bit
packing is identical on WH and BH:

```
    Source register (16 bits):
    +---+---+------+---+---+------+
    | 7 | 6 | 5..0 | F | E | D..8 |     (bit positions)
    +---+---+------+---+---+------+
    |clr|cr | incr | clr|cr| incr |
    |<-- SrcA ---->|<--- SrcB --->|

    Dest + Fidelity register (16 bits):
    +-----+---+---+---+----------+
    |15.13|12 |11 |10 | 9..0     |     (bit positions)
    +-----+---+---+---+----------+
    | fid |c2c|clr|cr | incr     |
    |     |   |   |   |          |
    | fidelity.val()  dest.val() |
```


## Tile Geometry and Row Addressing

A 32x32 tile is divided into four 16x16 **faces**. The FPU processes each
face in two steps of 8 rows. The address modifier `incr` values correspond
to jumps within this layout:

```
    One 32x32 tile in Dest register
    ================================

    Row 0  +----------------+----------------+
           |                |                |
           |   Face 0       |   Face 1       |
           |   (16x16)      |   (16x16)      |
           |                |                |
    Row 15 +----------------+----------------+
    Row 16 +----------------+----------------+
           |                |                |
           |   Face 2       |   Face 3       |
           |   (16x16)      |   (16x16)      |
           |                |                |
    Row 31 +----------------+----------------+

    Each face is processed as two 8-row halves:

    Face 0:
      rows  0.. 7  <-- first half   (8 rows)
      rows  8..15  <-- second half  (8 rows)
                        ^
                        |
                   incr = 8 advances from first half to second half
```


## FPU Example: Element-Wise Binary (Annotated Source)

Both WH and BH use identical address modifier configuration for FPU
element-wise binary operations. The source below is from Blackhole;
the Wormhole version is character-for-character the same.

### Address Modifier Configuration

```
    ADDR_MOD_0  (inner loop)     srca.incr=8  srcb.incr=8  dest.incr=8
    ADDR_MOD_1  (no-op)          srca.incr=0  srcb.incr=0  dest.incr=0
    ADDR_MOD_2  (end of face)    srca.clr=1   srcb.clr=1   dest.cr=1     fidelity.incr++
    ADDR_MOD_3  (end of tile)    srca.clr=1   srcb.clr=1   dest.c_to_cr  fidelity.clr=1
```

**File:** `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h`
(WH equivalent: `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h` -- identical)

```cpp
// === eltwise_binary_configure_addrmod (lines 18-42) ===
//
// Called once during _llk_math_eltwise_binary_init_ to program the
// four address modifier slots before any math instructions execute.

template <EltwiseBinaryType eltwise_binary_type,
          BroadcastType bcast_type,
          std::uint32_t FIDELITY_INCREMENT>
inline void eltwise_binary_configure_addrmod()
{
    // SrcB increment depends on broadcast type:
    //   NONE or COL -> advance normally (8 rows per step)
    //   ROW or SCALAR -> stay at row 0 (reuse same data)
    constexpr std::uint32_t srcb_incr =
        (bcast_type == BroadcastType::NONE ||
         bcast_type == BroadcastType::COL) ? 8 : 0;

    // ADDR_MOD_0: INNER LOOP
    // Used by every math instruction in the inner loop body.
    // After processing rows 0..7, this advances all three
    // pointers to rows 8..15 (second half of the face).
    addr_mod_t {
        .srca = {.incr = 8},             // SrcA: +8 rows
        .srcb = {.incr = srcb_incr},     // SrcB: +8 (or 0 for row/scalar bcast)
        .dest = {.incr = 8},             // Dest: +8 rows
    }
        .set(ADDR_MOD_0);

    // ADDR_MOD_1: NO-OP
    // No pointer movement at all. Used by special operations like
    // dest-reuse (move_d2a / move_d2b) and ZEROACC where pointers
    // must remain in place.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    // ADDR_MOD_2: END OF FACE
    // Applied after the last inner-loop instruction of each face.
    // SrcA and SrcB clear back to row 0 so they can reload the
    // next face's input data. Dest uses cr=1 (carry/reset) to
    // continue advancing into the next face WITHOUT clearing --
    // the carry register holds the base of the next face.
    // Fidelity increments for multi-pass accumulation modes.
    addr_mod_t {
        .srca = {.incr = 0, .clr = 1},  // SrcA: reset to row 0
        .srcb = {.incr = 0, .clr = 1},  // SrcB: reset to row 0
        .dest = {.incr = 0, .clr = 0,
                 .cr = 1},              // Dest: carry -> next face base
        .fidelity = {.incr = FIDELITY_INCREMENT}  // fidelity phase++
    }
        .set(ADDR_MOD_2);

    // ADDR_MOD_3: END OF TILE
    // Applied at the very last instruction of the entire tile.
    // Sources clear. Dest uses c_to_cr=1 to snapshot the current
    // counter into the carry register BEFORE resetting, so that
    // the next tile can start from the correct position.
    // Fidelity clears back to phase 0.
    addr_mod_t {
        .srca = {.incr = 0, .clr = 1},  // SrcA: reset to row 0
        .srcb = {.incr = 0, .clr = 1},  // SrcB: reset to row 0
        .dest = {.incr = 8, .clr = 0,
                 .cr = 0,
                 .c_to_cr = 1},         // Dest: save counter, then +8
        .fidelity = {.incr = 0,
                     .clr = 1}           // Fidelity: reset to phase 0
    }
        .set(ADDR_MOD_3);
}
```

### MOP (Micro-Op Program) Configuration

**File:** `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h`
(WH equivalent: `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h` -- identical)

```cpp
// === eltwise_binary_configure_mop (lines 201-239) ===
//
// Programs the MOP (Micro-Op Program) that the hardware will execute
// as a nested loop of math instructions. The MOP runs automatically
// once triggered by ckernel_template::run().

template <EltwiseBinaryType eltwise_binary_type,
          BroadcastType bcast_type,
          MathFidelity math_fidelity,
          EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_configure_mop(
    const std::uint32_t acc_to_dest = 0,
    const std::uint32_t num_faces = 4)
{
    // All inner-loop instructions use ADDR_MOD_0 (the +8 step).
    const std::uint32_t addr_mod = ADDR_MOD_0;

    // innerloop = 16 >> 3 = 2
    // Each iteration processes 8 rows. 2 iterations = 16 rows = 1 face.
    constexpr std::uint32_t innerloop = 16 >> 3;

    // outerloop = number of faces (typically 4).
    // COL broadcast: MOP processes 2 faces, called twice externally.
    const std::uint32_t outerloop =
        (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? 1
        : (bcast_type == BroadcastType::COL) ? 2
        : num_faces;

    // ---- ADD / SUB ----
    // All inner instructions use ADDR_MOD_0 (+8 each).
    // end_op fires after each outerloop iteration:
    //   SETRWC clears sources and applies carry/reset on Dest.
    if constexpr ((eltwise_binary_type == ELWADD) ||
                  (eltwise_binary_type == ELWSUB))
    {
        ckernel_template tmp(outerloop, innerloop,
            eltwise_binary_func<eltwise_binary_type>(
                0, acc_to_dest, broadcast_type, addr_mod));

        // end_op: clear SrcA (and SrcB unless bcast), carry/reset Dest
        tmp.set_end_op(TT_OP_SETRWC(
            CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    // ---- MUL (high fidelity) ----
    // Last inner instruction uses ADDR_MOD_2 (incr fidelity).
    // Last outer instruction uses ADDR_MOD_3 (clear everything).
    else if constexpr (eltwise_binary_type == ELWMUL)
    {
        ckernel_template tmp(
            high_fidelity ? to_underlying(math_fidelity) : outerloop,
            innerloop,
            eltwise_binary_func<ELWMUL>(
                0, 0, broadcast_type, addr_mod));

        if constexpr (high_fidelity)
        {
            // Last instruction of inner loop -> ADDR_MOD_2:
            //   incr fidelity phase, clr sources, carry dest
            tmp.set_last_inner_loop_instr(
                eltwise_binary_func<ELWMUL>(
                    0, 0, broadcast_type, ADDR_MOD_2));

            // Last instruction of outer loop -> ADDR_MOD_3:
            //   clr sources, c_to_cr on dest, clr fidelity
            tmp.set_last_outer_loop_instr(
                eltwise_binary_func<ELWMUL>(
                    CLR_SRC, 0, broadcast_type, ADDR_MOD_3));
        }
        else
        {
            tmp.set_end_op(TT_OP_SETRWC(
                CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        }
        tmp.program();
    }
}
```

### MOP Execution Diagram

```
    MOP structure (ADD/SUB, 4 faces, no broadcast):
    ================================================

    outerloop = 4 (one per face)
    innerloop = 2 (two 8-row halves per face)

    outer=0  +-- inner=0: ELWADD  addr_mod=MOD_0  (SrcA+=8, SrcB+=8, Dest+=8)
    (face 0) +-- inner=1: ELWADD  addr_mod=MOD_0
             +-- end_op:  SETRWC  (CLR_AB, CR_AB)  --> SrcA=0, SrcB=0, Dest carries

    outer=1  +-- inner=0: ELWADD  addr_mod=MOD_0
    (face 1) +-- inner=1: ELWADD  addr_mod=MOD_0
             +-- end_op:  SETRWC  (CLR_AB, CR_AB)

    outer=2  +-- inner=0: ELWADD  addr_mod=MOD_0
    (face 2) +-- inner=1: ELWADD  addr_mod=MOD_0
             +-- end_op:  SETRWC  (CLR_AB, CR_AB)

    outer=3  +-- inner=0: ELWADD  addr_mod=MOD_0
    (face 3) +-- inner=1: ELWADD  addr_mod=MOD_0
             +-- end_op:  SETRWC  (CLR_AB, CR_AB)  --> tile complete


    MOP structure (MUL, high fidelity HiFi4, 4 phases):
    ====================================================

    outerloop = 4 (math_fidelity phases, not faces)
    innerloop = 2

    outer=0  +-- inner=0: ELWMUL  addr_mod=MOD_0      (normal +8 step)
    (phase 0)+-- inner=1: ELWMUL  addr_mod=MOD_2 ***   (last inner -> incr fidelity)

    outer=1  +-- inner=0: ELWMUL  addr_mod=MOD_0
    (phase 1)+-- inner=1: ELWMUL  addr_mod=MOD_2 ***

    outer=2  +-- inner=0: ELWMUL  addr_mod=MOD_0
    (phase 2)+-- inner=1: ELWMUL  addr_mod=MOD_2 ***

    outer=3  +-- inner=0: ELWMUL  addr_mod=MOD_0
    (phase 3)+-- inner=1: ELWMUL  addr_mod=MOD_3 ***   (last outer -> clear all)
```

### Pointer Trace (ADD, 4 faces, no broadcast)

```
    Time -->

    Instruction    addr_mod    SrcA ptr    SrcB ptr    Dest ptr    Action
    ===========    ========    ========    ========    ========    ======
    ELWADD #0      MOD_0          0           0           0       rows 0..7
                                  |           |           |
                               +8 incr     +8 incr     +8 incr
                                  v           v           v
    ELWADD #1      end_op         8           8           8       rows 8..15
                                  |           |           |          (end face 0)
                               clr->0      clr->0      cr->16
                                  v           v           v
    ELWADD #2      MOD_0          0           0          16       face 1, rows 0..7
                                  |           |           |
                               +8 incr     +8 incr     +8 incr
                                  v           v           v
    ELWADD #3      end_op         8           8          24       face 1, rows 8..15
                                  |           |           |          (end face 1)
                               clr->0      clr->0      cr->32
                                  v           v           v
    ELWADD #4      MOD_0          0           0          32       face 2, rows 0..7
                                  |           |           |
                               +8 incr     +8 incr     +8 incr
                                  v           v           v
    ELWADD #5      end_op         8           8          40       face 2, rows 8..15
                                  |           |           |          (end face 2)
                               clr->0      clr->0      cr->48
                                  v           v           v
    ELWADD #6      MOD_0          0           0          48       face 3, rows 0..7
                                  |           |           |
                               +8 incr     +8 incr     +8 incr
                                  v           v           v
    ELWADD #7      end_op         8           8          56       face 3, rows 8..15
                                  |           |           |          (end tile)
                               clr->0      clr->0      cr->64
```

### Visualized on the Tile

```
    SrcA tile          SrcB tile          Dest tile
    +--------+         +--------+         +--------+--------+
    |  F0-hi |<--+     |  F0-hi |<--+     |  F0-hi | F1-hi |
    |  F0-lo |   |     |  F0-lo |   |     |  F0-lo | F1-lo |
    +--------+   |     +--------+   |     +--------+--------+
    |  F1-hi |   |     |  F1-hi |   |     |  F2-hi | F3-hi |
    |  F1-lo |   |     |  F1-lo |   |     |  F2-lo | F3-lo |
    +--------+   |     +--------+   |     +--------+--------+
                 |                  |         ^
       clr=1 resets            clr=1 resets   |
       back to top             back to top    dest only goes forward
       each face               each face      (cr=1 carry, never clr)

    Key insight:
      - SrcA/SrcB reload from row 0 for each new face  (clr=1 via end_op)
      - Dest advances continuously across all 4 faces   (cr=1 via end_op)
```


## SFPU Example: Unary Operations (Annotated Source)

SFPU operations use a fundamentally different pattern. The SFPU reads/writes
Dest directly via `dst_reg[]` indexing in SFPU instructions, so hardware
auto-increment is **disabled** (all zeros).

### Address Modifier Configuration

The configuration code is identical on WH and BH:

**File (BH):** `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
**File (WH):** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`

```cpp
// === eltwise_unary_sfpu_configure_addrmod (lines 22-57, both WH and BH) ===

template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_addrmod()
{
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    // ADDR_MOD_7: DEFAULT SFPU
    // All increments are zero. The SFPU manages its own Dest
    // addressing via dst_reg[] indexing in SFPU instructions,
    // so no hardware auto-increment is needed.
    addr_mod_t {
        .srca = {.incr = 0},    // no SrcA movement
        .srcb = {.incr = 0},    // no SrcB movement
        .dest = {.incr = 0},    // no Dest movement (SFPU handles it)
    }
        .set(ADDR_MOD_7);

    // ADDR_MOD_6: SPECIAL CASES
    // Some SFPU ops need hardware-assisted Dest pointer advancement.

    // topk_local_sort: jump an entire 32-row tile in Dest
    if constexpr (sfpu_op == SfpuType::topk_local_sort)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 32},    // +32 rows = full tile jump
        }
            .set(ADDR_MOD_6);
    }

    // typecast, reciprocal (BH only), unary_max/min variants:
    // small 2-row Dest steps for interleaved read/write patterns.
    //
    // >>> BH includes SfpuType::reciprocal here; WH does not <<<
    if constexpr (
        sfpu_op == SfpuType::reciprocal ||  // BH only (absent in WH)
        sfpu_op == SfpuType::typecast ||
        sfpu_op == SfpuType::unary_max ||
        sfpu_op == SfpuType::unary_min ||
        sfpu_op == SfpuType::unary_max_int32 ||
        sfpu_op == SfpuType::unary_min_int32 ||
        sfpu_op == SfpuType::unary_max_uint32 ||
        sfpu_op == SfpuType::unary_min_uint32)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},     // +2 rows per step
        }
            .set(ADDR_MOD_6);
    }
}
```

### SFPU Start / Done (WH vs BH difference)

**File (WH):** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`

```cpp
// === WH _sfpu_start_ and _sfpu_done_ (lines 60-74) ===

template <DstSync Dst>
inline void _llk_math_eltwise_unary_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32,
                             UnpackDestination::SrcRegs>(dst_index);

    // >>> WH ONLY: switch to Set 1 (ADDR_MOD_4..7) <<<
    // Makes the 2-bit addr_mod field in SFPU instructions index
    // into slots 4..7 instead of 0..3, preventing collision with
    // FPU addr mods that live in slots 0..3.
    math::set_addr_mod_base();     // TTI_SETC16(ADDR_MOD_SET_Base, 1)

    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_unary_sfpu_done_()
{
    math::clear_dst_reg_addr();

    // >>> WH ONLY: wait for SFPU pipeline to drain, then restore Set 0 <<<
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();   // TTI_SETC16(ADDR_MOD_SET_Base, 0)
}
```

**File (BH):** `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`

```cpp
// === BH _sfpu_start_ and _sfpu_done_ (lines 61-71) ===

template <DstSync Dst>
inline void _llk_math_eltwise_unary_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32,
                             UnpackDestination::SrcRegs>(dst_index);

    // >>> BH: NO base switch <<<
    // SFPU uses slots 6,7 directly (same set as FPU's 0..3).
    // No collision because different slot indices by convention.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_unary_sfpu_done_()
{
    // >>> BH: NO stall, NO base switch <<<
    // Just clear the Dest address register.
    math::clear_dst_reg_addr();
}
```

### WH Base-Switch Helpers

**File:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/cmath_common.h`

```cpp
// === set_addr_mod_base / clear_addr_mod_base (lines 173-181) ===
// These helpers exist ONLY in the Wormhole cmath_common.h.
// Blackhole has no equivalent.

inline void set_addr_mod_base()
{
    // Set base=1: 2-bit addr_mod indices now map to slots 4..7
    TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1);
}

inline void clear_addr_mod_base()
{
    // Set base=0: 2-bit addr_mod indices map back to slots 0..3
    TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0);
}
```

### Face Advancement (identical on WH and BH)

```cpp
// Both architectures
inline void _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()
{
    // Two +8 increments = +16 rows = advance past one 16x16 face.
    // Called by the compute kernel between face iterations.
    math::inc_dst_addr<8>();   // +8 rows
    math::inc_dst_addr<8>();   // +8 rows (total +16)
}
```

### SFPU Execution Diagram

```
    SFPU tile processing (e.g. unary exp):
    =======================================

    _sfpu_start_(dst_index)
      |
      |  [WH: set_addr_mod_base() -> use Set 1, slots 4..7]
      |  [BH: no base switch, slots accessed directly]
      v
    Face 0:  SFPU kernel runs, reads/writes dst_reg[0..15]
             (ADDR_MOD_7: all zero -> no auto-increment)
      |
      v
    inc_dst_face_addr()  --> Dest ptr += 16
      |
      v
    Face 1:  SFPU kernel runs, reads/writes dst_reg[0..15]
             (now operating on the next 16 rows of Dest)
      |
      v
    inc_dst_face_addr()  --> Dest ptr += 16
      |
      v
    Face 2:  SFPU kernel runs
      |
      v
    inc_dst_face_addr()  --> Dest ptr += 16
      |
      v
    Face 3:  SFPU kernel runs
      |
      v
    _sfpu_done_()
      |
      |  [WH: stall SFPU; clear_addr_mod_base() -> back to Set 0]
      |  [BH: clear_dst_reg_addr() only]
      v
    Done.

    Pointer trace:
    ==============
    Face 0:  Dest = base + 0    SFPU accesses dst_reg[0..15]
    Face 1:  Dest = base + 16   SFPU accesses dst_reg[0..15]  (at new base)
    Face 2:  Dest = base + 32   SFPU accesses dst_reg[0..15]
    Face 3:  Dest = base + 48   SFPU accesses dst_reg[0..15]
```


## Broadcast Variations

When SrcB is broadcast, its increment in ADDR_MOD_0 changes:

```
    Broadcast type      srcb.incr in ADDR_MOD_0    Behavior
    ==============      =======================    ========
    NONE                8                          normal: advance with SrcA
    COL                 8                          advance, but clear manually
    ROW                 0                          stay at row 0 (reuse same row)
    SCALAR              0                          stay at row 0 (reuse same value)

    Row broadcast example:

    SrcA              SrcB (row bcast)     Dest
    +----------+      +----------+         +----------+
    | row 0..7 |  op  | row 0..7 |----+    | row 0..7 |
    +----------+      +----------+    |    +----------+
    | row 8..15|  op  | row 0..7 |<---+    | row 8..15|
    +----------+      +----------+         +----------+
       incr=8            incr=0               incr=8
                         (stays put)
```


## Carry/Reset and `c_to_cr` Mechanics

The carry/reset (`cr`) and copy-to-carry (`c_to_cr`) fields enable Dest to
wrap correctly across face and tile boundaries without a full clear:

```
    c_to_cr = 1:  Save current counter value into carry register
                  before applying other operations.

    cr = 1:       Use carry register value as new base after reset.

    Example: processing a 4-face tile
    ==================================

                    Dest ptr    carry_reg    What happened
                    --------    ---------    ------------
    Start face 0:      0           0
    After MOD_0:       8           0         +8 (inner step)
    After end_op:     16           0         CR: ptr = carry + 16  --> 16
                                             (carry stays 0, ptr wraps forward)

    Start face 1:     16           0
    After MOD_0:      24           0         +8 (inner step)
    After end_op:     32           0         CR: ptr = carry + 32  --> 32

    Start face 2:     32           0
    After MOD_0:      40           0         +8 (inner step)
    After end_op:     48           0         CR: ptr = carry + 48  --> 48

    Start face 3:     48           0
    After MOD_0:      56           0         +8 (inner step)
    After end_op:     64           0         CR: ptr = carry + 64  --> 64
                                             (tile done, next tile starts at 64)
```


## Summary

```
    +=====================+=============================================+
    | Concept             | Key Point                                   |
    +=====================+=============================================+
    | 8 slots             | ADDR_MOD_0..7, pre-configured, selected     |
    |                     | per-instruction via 2-bit index              |
    +---------------------+---------------------------------------------+
    | Two sets            | Set 0 = slots 0..3, Set 1 = slots 4..7      |
    |                     | ADDR_MOD_SET_Base selects active set         |
    +---------------------+---------------------------------------------+
    | WH base switching   | SFPU calls set_addr_mod_base() to use       |
    |                     | Set 1 (4..7), isolating from FPU's Set 0    |
    +---------------------+---------------------------------------------+
    | BH base switching   | No base switching for unary/binary SFPU.    |
    |                     | SFPU uses slots 6,7 directly. Ternary SFPU  |
    |                     | uses raw TTI_SETC16(2, 0) in _done_().      |
    +---------------------+---------------------------------------------+
    | SrcA/SrcB incr      | 0..63 rows, clr resets to 0                 |
    +---------------------+---------------------------------------------+
    | Dest incr           | 0..1023 rows, cr for carry, c_to_cr to      |
    |                     | save counter before reset                    |
    +---------------------+---------------------------------------------+
    | FPU pattern         | incr=8 per step, clr sources each face,     |
    |                     | Dest advances via carry across all faces     |
    +---------------------+---------------------------------------------+
    | SFPU pattern        | All zeros (disabled), Dest managed by        |
    |                     | software via inc_dst_addr<8>()               |
    +---------------------+---------------------------------------------+
    | Fidelity            | Increments accumulation phase for multi-     |
    |                     | pass high-fidelity math modes                |
    +---------------------+---------------------------------------------+
```
