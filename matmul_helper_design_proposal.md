# Matmul Helper API — Unified Design Proposal

**Date:** 2026-04-14
**Branches analyzed:** `wransom/matmul_op_integ_verf` (comprehensive), `wransom/llk5` (encapsulated)

---

## Problem Statement

We have two matmul helper implementations with complementary strengths:

| | `matmul_op_integ_verf` | `llk5` |
|---|---|---|
| **Coverage** | ~28 production kernels | ~3 kernels |
| **DST encapsulation** | No — caller manages `tile_regs_*` | Yes — helper owns full 4-phase cycle |
| **CB encapsulation** | No — caller manages `wait/pop/reserve/push` | Yes — helper owns all CB operations |
| **Pack encapsulation** | No — caller runs `pack_tile` loops | Yes — helper packs internally |
| **Spill/reload** | Partial (`matmul_reload_partials`) | Full (automatic per K-block) |
| **L1_ACC / RELU** | Manual in kernel | Compile-time template params |

**Goal:** An API that gives us llk5's encapsulation for the majority of call sites, while still covering the complex fused kernels (SDPA, conv, MoE) that llk5 couldn't reach.

**Why this matters:** The long-term purpose is enabling Claude to generate correct matmul kernels automatically. DST and CB mismanagement causes hangs that are extremely difficult to debug. Every call site where the helper owns the DST/CB lifecycle is a call site where Claude cannot introduce a hang.

---

## Proposed Architecture: Three Tiers

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 1: Complete Helpers                                   │
│  Full DST + CB + pack + spill/reload encapsulation          │
│  CB IDs as compile-time template params                     │
│  Functor callbacks for extensibility                        │
│  → For all standard matmul patterns                         │
│  → 10 production call sites (~83%)                          │
├─────────────────────────────────────────────────────────────┤
│  Tier 3: Building Blocks                                    │
│  No DST/CB encapsulation — caller manages everything        │
│  → For SDPA only (absolute-offset packing, bidirectional CB)│
│  → 2 production call sites (~17%)                           │
└─────────────────────────────────────────────────────────────┘

Tier 2 (phase helpers) eliminated — all candidates were either
promoted to Tier 1 or excluded as experimental.
```

### Design Principles

1. **Maximize encapsulation surface.** Every call site that CAN use Tier 1 SHOULD — even if it requires a callback. Tier 3 is a last resort.
2. **Compile-time CB params for Tier 1** — enables `static_assert` validation, matches existing kernel_lib conventions (tilize, reduce, binary_op).
3. **4-phase DST everywhere** — `acquire/commit/wait/release`, consistent with all kernel_lib helpers.
4. **Functor callbacks over separate functions** — `PreKBlockFn` and `PostComputeFn` let Tier 1 absorb patterns that would otherwise require Tier 3 (e.g., conv's tilize-before-matmul).
5. **No name collisions** — Tier 1 names describe the *full operation* (`matmul_block`, `matmul_tile`). Tier 3 names describe the *primitive* (`matmul_single`, `matmul_accumulate`).

---

## Tier 1: Complete Helpers

### `matmul_block` — Block-mode matmul with full automation

From llk5, with extensions. Handles the entire K-blocking loop including DST, CB, spill/reload, L1_ACC, RELU, and callbacks.

```cpp
template <
    uint32_t in0_cb,          // Input A circular buffer
    uint32_t in1_cb,          // Input B circular buffer
    uint32_t out_cb,          // Output circular buffer
    uint32_t interm_cb,       // Intermediate CB for spill/reload
    bool transpose = false,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,  // For bias path: pack to interm, not out
    bool pack_relu = false,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock>
ALWI void matmul_block(
    uint32_t block_w,             // Inner dim block size (tiles)
    uint32_t in0_num_subblocks,   // M-dim subblocks
    uint32_t in1_num_subblocks,   // N-dim subblocks
    uint32_t num_k_blocks,        // K-dim blocks
    uint32_t out_subblock_h,      // Output subblock height (tiles)
    uint32_t out_subblock_w,      // Output subblock width (tiles)
    uint32_t batch = 1,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {});
```

**Internally manages:**
- `tile_regs_acquire/commit/wait/release` per subblock
- `cb_wait_front/cb_pop_front` for in0, in1 per K-block
- `cb_reserve_back/cb_push_back` for output/interm per subblock
- `copy_tile_to_dst` + `copy_block_matmul_partials` for reload
- `llk_pack_reconfig_l1_acc` toggling
- `llk_pack_relu_config` on last K-block
- `pack_reconfig_data_format` for FP32_DEST_ACC_EN
- Out CB reservation guard (prevents interm/out overlap corruption)

### `matmul_tile` — Tile-mode matmul with full automation

From llk5. Simple Mt × Nt × Kt loop with per-tile CB management.

```cpp
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    bool transpose = false,
    typename PostComputeFn = NoPostCompute>
ALWI void matmul_tile(
    uint32_t Mt, uint32_t Nt, uint32_t Kt,
    uint32_t batch = 1,
    PostComputeFn post_compute = {});
```

**Internally manages:** Full 4-phase DST per output tile, all CB wait/pop/reserve/push, pack.

### `add_bias_bcast_rows` — Bias addition phase

From llk5. Composes with `matmul_block<..., pack_last_to_interm=true>`.

```cpp
template <
    uint32_t partials_cb,
    uint32_t bias_cb,
    uint32_t out_cb,
    typename PostBiasFn = NoPostBias>
ALWI void add_bias_bcast_rows(
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t bias_width_tiles,
    PostBiasFn post_bias = {});
```

**Internally manages:** Full DST cycle, format reconfig, bias CB wait, partials CB pop, output CB reserve/push.

### Tier 1 Call Site Coverage (production only)

| Call site | Tier 1 helper | Notes |
|-----------|--------------|-------|
| `bmm.cpp` | `matmul_tile` | Direct replacement, ~4 lines of kernel logic |
| `bmm_large_block_zm.cpp` | `matmul_block` | Direct replacement, ~8 lines of kernel logic |
| `bmm_large_block_zm_fused_bias_activation.cpp` | `matmul_block` + `add_bias_bcast_rows` | PostComputeFn for SFPU, PreKBlockFn for transpose |
| `bmm_large_block_zm_fused_bias_activation_gathered.cpp` | `matmul_block` + `add_bias_bcast_rows` | CB alternation — see Q2 |
| `conv_bmm_tilize.cpp` | `matmul_block` with PreKBlockFn | Tilize functor; L1_ACC pointer mgmt — see Q1 |
| `bmm_tilize_untilize.cpp` | `matmul_block` with PreKBlockFn | Similar to conv |
| `moreh_matmul.cpp` | `matmul_tile` | Mt×Nt×Kt loop |
| `reduce_w.cpp` | `matmul_reduce_w` (new) | Own Tier 1 helper — different loop shape |
| `moreh_mean_w.cpp` | `matmul_reduce_w` w/ reinit | Same helper, `reinit_per_tile=true` |
| `moreh_sum_w.cpp` | `matmul_reduce_w` w/ reinit | Same helper, `reinit_per_tile=true` |

**Tier 1 coverage: 10 of 12 production call sites (83%)**

---

## ~~Tier 2: Phase Helpers~~ — Eliminated

Originally proposed for specialized patterns (reduce-W, attention, MoE). After filtering:
- **reduce_w family** → promoted to Tier 1 (own `matmul_reduce_w` helper, no real barrier)
- **attention matmul** → all call sites experimental, excluded
- **MoE** → all call sites experimental, excluded

No production call sites remain that need Tier 2. The concept is sound (encapsulate one acquire→pack→release cycle) and could be revived if experimental ops graduate to production.

---

## Tier 3: Building Blocks

For kernels with deeply non-standard control flow where the caller MUST own DST/CB. These are kept from `matmul_op_integ_verf` with minor naming adjustments.

```cpp
// Single LLK dispatch (no DST/CB management)
template <MatmulMode mode>
ALWI void matmul_single(const MatmulConfig& cfg, uint32_t in0, uint32_t in1, uint32_t dst);

// Strided accumulation loop (no DST/CB management)
template <MatmulMode mode>
ALWI void matmul_accumulate(const MatmulConfig& cfg, ...);

// Subblock accumulation (no DST/CB management)
template <MatmulMode mode>
ALWI void matmul_accumulate_subblock(const MatmulConfig& cfg, ...);

// BH-specific no-MOP variant
#ifdef ARCH_BLACKHOLE
template <MatmulMode mode>
ALWI void matmul_accumulate_no_mop(const MatmulConfig& cfg, ...);
#endif

// Pack phase only: commit → reserve → wait → pack → release → push
// (Manages the second half of the DST cycle — prevents commit/release bugs)
ALWI void matmul_pack_to_cb(uint32_t dest_cb_id, uint32_t num_tiles);

// Reload partials into DST (must be called after tile_regs_acquire)
template <MatmulMode mode>
ALWI void matmul_reload_partials(const MatmulConfig& cfg, uint32_t num_tiles);

// All init variants
template <MatmulMode mode> ALWI void matmul_init(const MatmulConfig& cfg);
template <MatmulMode mode> ALWI void matmul_init_short(const MatmulConfig& cfg);
template <MatmulMode mode> ALWI void matmul_init_short_with_dt(const MatmulConfig& cfg, uint32_t old);
template <MatmulMode mode> ALWI void matmul_init_short_with_both_dt(const MatmulConfig& cfg, ...);
```

### Tier 3 Call Site Coverage

| Call site | Why Tier 3 is needed |
|-----------|---------------------|
| SDPA `compute_streaming.hpp` | Absolute-offset packing, sub_exp interleaving between subblocks, hardware semaphores, `cb_push_back_hold_wr_ptr` |
| SDPA `compute_common.hpp` (`matmul_blocks`) | Legacy wrapper with its own absolute-offset pack and subblock loop; worth refactoring later |
| `topk_router_gpt/compute.cpp` | Interleaved with TopK operations |
| DeepSeek `matmul_wo/compute.cpp` | PACK-thread SFPU via TTI macros |
| DeepSeek `rope.hpp` | Uses matmul within larger fused kernel |

**Estimated Tier 3 coverage: ~5 call sites**

---

## Shared Types (used by all tiers)

### `MatmulConfig` (from `matmul_op_integ_verf`, for Tiers 2-3)

```cpp
struct MatmulConfig {
    uint32_t in0_cb_id, in1_cb_id, out_cb_id;
    uint32_t ct_dim = 1, rt_dim = 1, kt_dim = 1;
    bool transpose = false;
    uint32_t partials_cb_id = 0;

    static constexpr MatmulConfig tile(uint32_t in0, uint32_t in1, uint32_t out, bool trans = false);
    static constexpr MatmulConfig block(uint32_t in0, uint32_t in1, uint32_t out,
                                         uint32_t ct, uint32_t rt, uint32_t kt, ...);
};
```

### `MatmulMode` enum (from `matmul_op_integ_verf`, for Tiers 2-3)

```cpp
enum class MatmulMode { TILE, BLOCK };
inline constexpr MatmulMode TILE = MatmulMode::TILE;
inline constexpr MatmulMode BLOCK = MatmulMode::BLOCK;
```

### Functor types (from llk5, for Tier 1)

```cpp
struct NoPostCompute { ALWI void operator()(uint32_t) const {} };
struct NoPreKBlock   { ALWI void operator()(uint32_t, uint32_t, bool) const {} };
struct NoPostBias    { ALWI void operator()(uint32_t) const {} };
```

---

## File Organization

| File | Contents | Tier |
|------|----------|------|
| `matmul_block_helpers.hpp/inl` | `matmul_block`, functor types | Tier 1 (block) |
| `matmul_tile_helpers.hpp/inl` | `matmul_tile` | Tier 1 (tile) |
| `bias_add_helpers.hpp/inl` | `add_bias_bcast_rows` | Tier 1 (bias) |
| `matmul_reduce_w_helpers.hpp/inl` | `matmul_reduce_w` | Tier 1 (reduce-W) |
| `matmul_helpers_compute.hpp/inl` | `MatmulConfig`, `MatmulMode`, Tier 3 building blocks | Tier 3 (SDPA) |

Tier 1 files include only what they need (LLK headers, `dest_helpers.hpp`, `cb_helpers.hpp`).
Tier 3 file includes LLK headers and defines shared types.
Most kernels include only Tier 1 files. Only SDPA includes the Tier 3 file.

---

## Experimental Call Sites — Excluded from Design Scope

11 of the original ~28 migrated call sites live under `experimental/` paths. These are not production code and should not drive API design decisions. They will still benefit from whatever helpers exist (Tier 1 if compatible, Tier 3 building blocks otherwise), but we do not design around their constraints.

| Experimental call site | Original tier | Why excluded |
|------------------------|--------------|--------------|
| `experimental/matmul/attn_matmul/transformer_attn_matmul.cpp` | Was Tier 2 | Experimental |
| `experimental/matmul/group_attn_matmul/transformer_group_attn_matmul.cpp` | Was Tier 2 | Experimental |
| `experimental/conv3d/compute.cpp` | Was Tier 1 | Experimental |
| `experimental/deepseek/mla/matmul_wo/compute.cpp` | Was Tier 3 | Experimental; pack-thread SFPU |
| `experimental/deepseek/moe/moe_gate_mm/compute.cpp` | Was Tier 2 | Experimental |
| `experimental/ccl/moe_compute/compute.cpp` | Was Tier 2 | Experimental |
| `experimental/ccl/moe_gpt/compute.cpp` | Was Tier 2 | Experimental |
| `experimental/minimal_matmul/compute.cpp` | Was Tier 1 | Experimental |
| `experimental/ccl/all_gather_minimal_matmul_async/compute.cpp` | Was Tier 1 | Experimental |
| `experimental/ccl/llama_all_gather_matmul_async/.../gathered.cpp` | Was Tier 1 | Experimental |
| `experimental/topk_router_gpt/compute.cpp` | Was Tier 3 | Experimental; TopK interleaving |

**Also excluded:** `models/demos/deepseek_v3_b1/unified_kernels/rope.hpp` (model demo, not a production op) and `tt-train/` SDPA kernels (separate project, follows SDPA patterns).

**Note on MoE helpers:** All MoE call sites are experimental. The `matmul_moe_*` specialized helpers from `matmul_op_integ_verf` (DM1 cycling, bias-on-limit-hit) are too MoE-specific to justify as a designed API surface. If MoE kernels graduate from experimental, they can use Tier 3 building blocks or get a purpose-built helper at that time.

---

## Coverage Summary (Production Call Sites Only)

After excluding experimental/demo/tt-train sites, **~13 production call sites** remain:

| Call site | Tier | Helper | DST encap | CB encap | Barrier to Tier 1 |
|-----------|------|--------|-----------|----------|--------------------|
| `bmm.cpp` | **1** | `matmul_tile` | Yes | Yes | None |
| `bmm_large_block_zm.cpp` | **1** | `matmul_block` | Yes | Yes | None |
| `bmm_large_block_zm_fused_bias_activation.cpp` | **1** | `matmul_block` + `add_bias_bcast_rows` | Yes | Yes | None (PreKBlockFn for transpose, PostComputeFn for SFPU) |
| `bmm_large_block_zm_fused_bias_activation_gathered.cpp` | **1** | `matmul_block` | Yes | Yes | CB alternation — see Q2 |
| `conv_bmm_tilize.cpp` | **1** | `matmul_block` w/ PreKBlockFn | Yes | Yes | Tilize in PreKBlockFn; L1_ACC pointer mgmt — see Q1 |
| `bmm_tilize_untilize.cpp` | **1** | `matmul_block` w/ PreKBlockFn | Yes | Yes | Similar to conv |
| `moreh_matmul.cpp` | **1** | `matmul_tile` | Yes | Yes | None (Mt×Nt×Kt loop) |
| `reduce_w.cpp` | **1** | `matmul_reduce_w` (new Tier 1) | Yes | Yes | None — different loop shape, own helper |
| `moreh_mean_w.cpp` | **1** | `matmul_reduce_w` w/ reinit | Yes | Yes | None — same helper, `reinit_per_tile=true` |
| `moreh_sum_w.cpp` | **1** | `matmul_reduce_w` w/ reinit | Yes | Yes | None — same as moreh_mean_w |
| SDPA `compute_streaming.hpp` | **3** | Tier 3 building blocks | No | No | **Real barrier:** absolute-offset packing, bidirectional CB (sub_exp interleaving), hardware semaphores, `cb_push_back_hold_wr_ptr` |
| SDPA `compute_common.hpp` | **3** | Tier 3 building blocks | No | No | **Real barrier:** absolute-offset packing into pre-reserved CB region; `matmul_reduce_subblock_inplace` already encapsulates its own DST |
| `rope.hpp` (DeepSeek demo) | — | Excluded | — | — | Model demo, not production op |

### Summary

| Tier | DST encapsulated | CB encapsulated | Production call sites | % of production |
|------|------------------|-----------------|----------------------|-----------------|
| Tier 1 | Yes | Yes | **10** | **~83%** |
| Tier 3 | No | No | **2** (SDPA only) | **~17%** |
| **Total** | | | **12** | **100%** |

**Tier 2 is eliminated.** The patterns that were Tier 2 (reduce_w, attention) are now promoted to Tier 1 as their own fully-encapsulated helpers. The remaining Tier 2 candidates (MoE, attention) were all experimental.

**~83% of production call sites get full DST+CB encapsulation** — and the only holdouts are the two SDPA kernel files, which have genuine hardware-level barriers.

---

## Per-Call-Site Analysis: What Gets Absorbed

### Tier 1 Call Sites

**`bmm_large_block_zm_fused_bias_activation.cpp`** (477 lines → ~80 lines estimated)

Currently has ~10 manual `tile_regs_*` call sites, ~12 manual CB operations, ~4 pack loops. With Tier 1:

```cpp
// Non-bias path: single call replaces the entire K-blocking loop
matmul_block<in0_cb, in1_cb, mm_out_cb, mm_partials_cb,
             in1_transpose, l1_acc, false, do_relu,
             PostMatmulSFPU, PreFn>(
    in0_block_w, in0_num_subblocks, in1_num_subblocks,
    num_blocks_inner, out_subblock_h, out_subblock_w, 1,
    PostMatmulSFPU{}, PreFn{});

// Bias path: matmul packs to interm, then bias helper reads from interm
matmul_block<in0_cb, in1_cb, mm_out_cb, mm_partials_cb,
             in1_transpose, l1_acc, true, false,
             NoPostCompute, PreFn>(...);
add_bias_bcast_rows<mm_partials_cb, bias_cb, out_cb, PostBiasSFPU>(...);
```

**Absorbed:** All DST management, all CB management, all pack loops, spill/reload, L1_ACC toggling, RELU config, out-CB reservation guard.
**Remains in kernel:** Argument parsing, `#ifdef` dispatch, `SKIP_COMPUTE` path (fundamentally different), transpose functor definition, SFPU functor definition, untilize-out path (post-matmul).

**`conv_bmm_tilize.cpp`** (601 lines → ~150 lines estimated)

The matmul phase uses `PreKBlockFn` to handle tilize + re-init:

```cpp
struct TilizePreKBlock {
    ALWI void operator()(uint32_t block, uint32_t num_k_blocks, bool last) const {
        // Tilize activation block from untilized CB to tilized CB
        tilize_block<untilized_cb, tilized_cb>(block_h, block_w);
        // Re-init matmul after tilize changed unpack data format
        mm_block_init_short_with_both_dt(in0_cb, in1_cb, tilized_cb, untilized_cb, ...);
    }
};
```

**Absorbed:** Subblock DST loop, CB management for matmul inputs/outputs, spill/reload, L1_ACC.
**Remains in kernel:** Tilize functor, bias fuse phase, untilize phase, partials CB pointer management for L1_ACC mode (**see Question 1**).

**`reduce_w.cpp`** — Promoted from Tier 2 to Tier 1. Gets its own `matmul_reduce_w` helper:

```cpp
matmul_reduce_w<in0_cb, in1_cb, out_cb>(Wt, dst_idx);
```

**Absorbed:** DST acquire/release, per-tile in0 wait/pop, static in1 hold, pack, CB reserve/push.
**No technical barrier** — just a different loop shape (per-tile in0 streaming, in1 never popped). The moreh variants use `reinit_per_tile=true` for data format reconfig.

### Tier 3 Call Sites (SDPA only)

**SDPA `compute_streaming.hpp`** — The `blocked_matmul_and_pack` function already manages DST internally. The barriers are real and hardware-level:

1. **Absolute-offset packing:** Multiple subblocks write into different offsets of the SAME pre-reserved CB region via `pack_tile<true>(row * cols + col, cb)`. Tier 1 helpers pack sequentially.
2. **Bidirectional CB / sub_exp interleaving:** The consumer of matmul output (sub_exp) interleaves with the producer of the next matmul subblock. The matmul data format changes after sub_exp, requiring mid-loop `mm_block_init_short` re-issue.
3. **`cb_push_back_hold_wr_ptr`:** Hardware trick advancing read pointer without write pointer. No standard helper has this concept.
4. **Hardware semaphore in pack loop:** `t6_semaphore_post` triggers early reduce before all tiles are packed.

**SDPA `compute_common.hpp`** — Same absolute-offset packing barrier. `matmul_reduce_subblock_inplace` (already encapsulates its own DST cycle) is the one helper that works well here.

**What Tier 3 provides for SDPA:** `matmul_accumulate<BLOCK>` / `matmul_accumulate_no_mop<BLOCK>` replace raw `ckernel::matmul_block` calls. Init wrappers replace raw LLK calls. Config struct for consistency. But the SDPA-local `blocked_matmul_and_pack` remains the real orchestrator.

---

## Migration Path

### Phase 1: Implement Tier 1 helpers
Start from llk5's proven implementations.
- Port `matmul_block_helpers.hpp/inl` from llk5
- Port `matmul_tile_helpers.hpp/inl` from llk5
- Port `bias_add_helpers.hpp/inl` from llk5
- Create new `matmul_reduce_w_helpers.hpp/inl` (Tier 1 for reduce-W pattern)
- Migrate the 10 Tier 1 production call sites

### Phase 2: Implement Tier 3 building blocks for SDPA
Keep only what SDPA needs from `matmul_op_integ_verf`:
- `matmul_single`, `matmul_accumulate`, `matmul_accumulate_no_mop` (BH)
- `matmul_pack_to_cb`, `matmul_reload_partials`
- `matmul_reduce_subblock_inplace`
- Init wrappers, `MatmulConfig`, `MatmulMode`
- Migrate the 2 SDPA call sites

### Phase 3: Cleanup
- Remove any Tier 3 function not used by a production call site
- Remove old `matmul_op.h` header
- Remove summary `.md` files from repo root

### Phase 4: Test and verify
- Run full test suites on both WH and BH
- Verify zero regressions across all ~2,700+ tests

---

## Open Questions

### Question 1: Conv partials CB pointer management

`conv_bmm_tilize.cpp` manually saves/restores the `fifo_rd_ptr` and `fifo_wr_ptr` of the partials CB across inner-block iterations when using `PACKER_L1_ACC`. This is because L1 accumulation writes back to the same memory location, but the CB FIFO semantics would normally advance the pointer.

**Options:**
- (a) Add `partials_pointer_stable` template param to `matmul_block` that handles pointer save/restore internally
- (b) Handle this in `PreKBlockFn` (the functor saves/restores pointers)
- (c) Handle this generically in the L1_ACC spill/reload path within `matmul_block` (the llk5 implementation already handles L1_ACC FIFO advancement — check if the conv pattern is actually the same logic)

**My lean:** (c) — the llk5 `matmul_block` already has L1_ACC FIFO advancement logic (`cb_wait_front/cb_pop_front` on interm between K-blocks). The conv pattern may be doing the same thing differently. Need to verify.

### Question 2: Gathered variant CB alternation

The gathered variant alternates between two in0 CBs per batch. The compile-time CB template param model doesn't support runtime CB switching. Should this be:
- (a) A `PreKBlockFn` that sets the active in0 CB (requires the helper to accept a mutable in0 CB — breaks compile-time model)
- (b) A separate `matmul_block_gathered` helper
- (c) Leave the gathered variant using Tier 3 building blocks

**My lean:** (b) or (c). The gathered variant is one kernel file.

### Question 3: SKIP_COMPUTE path in fused_bias kernel

The `SKIP_COMPUTE` path in `bmm_large_block_zm_fused_bias_activation.cpp` skips matmul and only does reload→pack. It's ~100 lines of manual DST/CB code. Should we:
- (a) Create a separate `reload_and_pack` helper
- (b) Leave it inline (it's an edge case)

**My lean:** (a) — it's a clean, self-contained pattern.

### Question 4: Naming — `matmul_tile` collision

Both Tier 1 and Tier 3 have a function conceptually named "matmul for one tile":
- Tier 1 (llk5): `matmul_tile<in0_cb, in1_cb, out_cb>(Mt, Nt, Kt)` — full loop
- Tier 3 (matmul_op_integ_verf): `matmul_tile<mode>(cfg, in0_idx, in1_idx, dst_idx)` — single LLK dispatch

Should we rename Tier 3 to `matmul_single` to avoid confusion?

**My lean:** Yes. The `matmul_op_integ_verf` implementation already has `detail::matmul_single` internally.

---

## What We Take From Each Branch

### From `llk5`:
- Tier 1 implementations (`matmul_block`, `matmul_tile`, `add_bias_bcast_rows`) — proven, tested
- 4-phase DST convention
- Compile-time CB template params for Tier 1
- `PostComputeFn` / `PreKBlockFn` functor pattern
- `PACKER_L1_ACC` / `PACK_RELU` / `pack_last_to_interm` as compile-time template params
- Compile-time `static_assert` validation
- Runtime `ASSERT` validation (DST capacity, CB capacity)
- Separate file organization per helper type

### From `matmul_op_integ_verf`:
- `MatmulConfig` struct with factory methods (for Tier 3 / SDPA)
- `MatmulMode` enum with `TILE`/`BLOCK` (for Tier 3 / SDPA)
- Tier 3 building blocks: `matmul_single`, `matmul_accumulate`, `matmul_pack_to_cb`, `matmul_reload_partials`, `matmul_reduce_subblock_inplace`
- `#ifndef ALWI` guard pattern
- Comprehensive call site migration experience (all ~28 sites analyzed and tested)
- Init wrapper patterns (`matmul_init_short_with_dt`, etc.)
