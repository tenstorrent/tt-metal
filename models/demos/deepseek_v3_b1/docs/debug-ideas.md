# Half-DEST Iter-PCC — Two Debug Methods

> Companion to [`half-dest-workload.md`](./half-dest-workload.md). Two concrete experiments to localize / fix the alternating iter-0 vs iter-1 PCC (0.99318986 vs 0.99344836) that Austin's iter-top `llk_math_pack_sync_init` + `llk_pack_dest_init(0)` papers over at `decoder_block_kernel.cpp:3055-3058`.

---

## Current ZEROACC sequence (the thing we're poking at)

`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h:37-59` — half-DEST path:

```cpp
TTI_STALLWAIT(STALL_MATH, PACK);                            // stalls PACK until pack finishes
TT_ZEROACC(CLR_HALF, fp32, 0, ADDR_MOD_1, dest_offset_id%2);// clears bank PACK just consumed; runs on MATH pipe
_llk_packer_set_math_semaphore_<NONE>();                    // increments MATH_PACK sem
flip_packer_dest_offset_id();                                // PACK flips its own pointer
select_packer_dest_registers<Dst>();
```

And `_llk_math_wait_for_dest_available_` (`llk_math_common.h:78-84`) is just `math_dest_wait()` — a SEMWAIT, nothing else. MATH's own bank flip already happened at the previous `tile_regs_commit()` (`_llk_math_dest_section_done_` → `dest_section_flip()`), so by the time MATH wakes from acquire, `MATH_Offset` already points at the bank that PACK was supposed to have ZEROACC'd. There's no STALLWAIT between PACK's ZEROACC (which is issued onto the MATH/SFPU pipe!) and the SEMINC that releases MATH. That's the gap Method 2 targets.

---

## Method 1 — extra acquire/commit/release pair in the critical region (parity-bisect)

**What it tests.** Without Austin's workaround, every iter does `N` half-bank flips, where `N` is odd (otherwise iter-1 would already start on bank 0 and there'd be no bug). Injecting one extra "dummy" flip at point P makes the total per-iter flips `N+1` (even) → iter-1 again starts on bank 0 → bug goes away **regardless of where P is**. So a single extra pair is parity-equivalent to Austin's fix and tells you nothing about location.

The useful variant is a **conditional** extra flip: insert a dummy acquire/commit/release at point P that runs **only on iter 0**. Then iter-0 sees `N+1` flips and iter-1 sees `N` flips, so the iter-0/iter-1 parity at every operation downstream of P is now inverted relative to baseline. By sweeping P through the critical region and watching which iter produces 0.99318986 vs 0.99344836, you bisect:

- Move P forward until the PCC pair **swaps** (iter-0 now gets the iter-1 number and vice versa). Everything to the left of that boundary is bank-parity-insensitive. Everything to the right is.
- The leftmost P at which the swap happens identifies the first DEST consumer whose output depends on physical-bank identity.

Concrete dummy pair, droppable anywhere TRISC has no pending DEST work:

```cpp
#if defined(COMPILE_FOR_TRISC)
if (iteration == 0) {
    tile_regs_acquire();
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
}
#endif
```

Coarse sweep first — put it (a) immediately before `flash_mla`'s `tile_regs_acquire` at `flash_mla.hpp:747`, (b) immediately after its `tile_regs_release` at `:791`, (c) just after the post-SDPA tail at `:813`, (d) before `moe_body()` at `decoder_block_kernel.cpp:3093`. Whichever neighbouring pair brackets the swap is the suspect region. Then bisect inside.

Caveats:
- Acquire blocks on `MATH_PACK` semaphore being at max — make sure no pending PACK work could deadlock you (i.e. all previous `cb_push_back`s have settled). Placement just inside the iteration loop (around the existing `mla_body`/`moe_body` boundaries) is safe; inside `compute_sdpa_chunk`'s nested SFPU phases is not.
- An acquire/release pair on TRISC affects **both** MATH and PACK parity. If the bug is asymmetric (e.g. PACK side stays misaligned but MATH flips back), this experiment will mask it. If the swap point seems weirdly diffuse, that's a hint to investigate per-thread imbalance.

---

## Method 2 — move ZEROACC from PACK release to MATH acquire

**What it tests.** The hypothesis: PACK's `TTI_STALLWAIT(STALL_MATH, PACK)` only gates PACK on prior PACK ops; the subsequent `TT_ZEROACC` is dispatched onto the MATH/SFPU pipe and is **not** ordered with the immediately-following `_llk_packer_set_math_semaphore_` (which is a PACK-pipe op). So PACK can post the MATH_PACK semaphore before ZEROACC has retired. MATH wakes from `math_dest_wait()` and starts issuing FPU writes into a bank that's still being cleared → race-condition-style stale residues that depend on which physical bank you're on (because the two banks have different relative timing for their first MATH op after wake — e.g. SRCB-reuse paths in `sdpa_custom_mm_reuse_dest_srcb_block` may schedule differently).

Moving ZEROACC into `_llk_math_wait_for_dest_available_` puts it in the **same thread** as the FPU writes that follow, so program-order on TRISC1 is sufficient — no cross-thread sync needed.

Proposed edit (need to add the `is_fp32_dest_acc_en` template param to the acquire LLK, which currently only takes `Dst`):

```cpp
// llk_pack_common.h — keep stall and flip, remove ZEROACC
inline void _llk_pack_dest_section_done_() {
    TTI_STALLWAIT(STALL_MATH, PACK);
    // ZEROACC removed — now happens on MATH at acquire
    _llk_packer_set_math_semaphore_<NONE>();
    if constexpr (Dst == DstSync::SyncHalf) {
        flip_packer_dest_offset_id();
        select_packer_dest_registers<Dst>();
    }
}

// llk_math_common.h — clear the bank we're about to use
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_wait_for_dest_available_() {
    math_dest_wait();
    if constexpr (Dst == DstSync::SyncHalf) {
        TT_ZEROACC(CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
    } else {
        TTI_ZEROACC(CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0);
    }
}
```

Why the bank index is correct after the move: MATH already executed `dest_section_flip()` at the previous `tile_regs_commit` (`llk_math_common.h:88-93`), so MATH-thread's `dest_offset_id` already names the new bank when acquire returns. That's the same physical bank PACK just packed from (and was clearing under the old scheme). Good.

Gotchas / things to verify before declaring victory:

1. **Tile-regs-acquire callers everywhere.** This LLK is used by every compute kernel, not just `flash_mla`. The bank you wake into is supposed to already be clean from the *previous* `tile_regs_release`; under the old scheme PACK clears it eagerly, under the new scheme MATH clears it lazily right before use. Anything that reads DEST between release and the next acquire (rare, but `dest_srcb_reuse` paths in MM2 and the SFPU helpers do touch DEST off the critical path) could now observe non-zero stale tiles. Audit any `dst_reg`/`dest_srcb` access that doesn't go through the standard acquire→write→commit→release flow.
2. **First iteration.** The very first `tile_regs_acquire` of a kernel was previously seeing a bank that `_llk_pack_dest_init_` had left in some state (init doesn't ZEROACC). Now the first acquire will ZEROACC unconditionally on entry — fine, but worth confirming nothing depends on "uninitialized but specific" DEST contents at boot. The reset workaround at `decoder_block_kernel.cpp:3056-3057` already calls `llk_math_pack_sync_init` + `llk_pack_dest_init` every iter, so that path is robust.
3. **DST_ACCUM_MODE wiring.** `_llk_math_wait_for_dest_available_` is currently called via the `tile_regs_acquire` macro chain without `is_fp32_dest_acc_en`. The macro currently routes through `acquire_dst()` which calls `MATH((llk_math_wait_for_dest_available()))` — you'll need to thread `DST_ACCUM_MODE` through that wrapper or hardcode it from the existing `DST_ACCUM_MODE` define used in compute kernels.
4. **Side hypothesis worth checking cheaply first**: add a single `TTI_STALLWAIT(STALL_MATH, MATH|SFPU0|SFPU1)` after the existing ZEROACC in `_llk_pack_dest_section_done_` but before `_llk_packer_set_math_semaphore_`. If that alone makes iter-1 PCC match iter-0, you've confirmed the in-flight-ZEROACC race without restructuring the LLK. That's the surgical version of Method 2 — keep the location, just close the gap.

---

## Recommended order

1. **Drop in the stallwait-after-ZEROACC patch first** (gotcha #4). Cheapest experiment, no API changes, directly tests the race hypothesis. If iter PCC equalizes, Method 2 is the right structural fix.
2. If #1 doesn't move PCC, run **Method 1 with the iter-0-only conditional flip** sweep. The boundary tells you which DEST consumer is bank-sensitive, which is information neither Method 2 nor the workaround gives you.
3. Only then commit to the full LLK refactor in Method 2 — by that point you'll know whether the issue is timing (#1 fixes it), bank-identity (Method 1 localizes it to a specific consumer that needs its own fix), or both.
