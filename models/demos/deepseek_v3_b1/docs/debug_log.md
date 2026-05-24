# Half-DEST Iter-PCC Debug Log — #43563

> Running record of experiments attempted to localize / fix the iter-alternating MLP PCC in `test_decoder_mlp`. Companion to [`debug-ideas.md`](./debug-ideas.md), [`half-dest-workload.md`](./half-dest-workload.md), and [`half-dest-conversation-summary.md`](./half-dest-conversation-summary.md).

## Environment & baseline

- Host: `bh-lb-26` (Blackhole P150, 8 devices, 12×10 worker grid in fast dispatch → 13×10 with `TT_METAL_SLOW_DISPATCH_MODE=1`).
- Test: `pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_decoder_block.py::test_decoder_mlp` with `position_id=8190`, `num_internal_iterations ∈ {1, 2}`, mesh `4×2`, half-DEST mode.
- Run command:
  ```bash
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
    python_env/bin/pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_decoder_block.py::test_decoder_mlp \
    -x --timeout=1500 -s
  ```
- Markers/edits enabling the run on this host:
  - `@pytest.mark.requires_grid_size((13, 10))` commented at `test_decoder_block.py:892` (slow dispatch gives 13×10, but autouse fixture races device open).
  - Austin's iter-top workaround commented at `decoder_block_kernel.cpp:3055-3058` (so the bug is observable).

### Reproduced baseline (no patches)

| `num_internal_iterations` | MLP PCC (vs torch golden) |
|---|---|
| 1 | **0.9931898601668998** |
| 2 | **0.9934483676630709** |

Bit-identical to the values recorded in `half-dest-conversation-summary.md` §1. The bug is reproducible and stable.

Both PCCs pass the 0.975 threshold, so failures are not visible to CI; the harm is the parity-dependent drift itself.

---

## Attempt 1 — Patch #4: STALLWAIT between ZEROACC and MATH_PACK SEMPOST

**What.** In `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h:52`, change `_llk_packer_set_math_semaphore_<p_stall::NONE>()` to `_llk_packer_set_math_semaphore_<p_stall::MATH | p_stall::WAIT_SFPU>()` inside `_llk_pack_dest_section_done_`. Per `ckernel.h:281-290`, this injects `TTI_STALLWAIT(STALL_SYNC, MATH | WAIT_SFPU)` immediately before the SEMPOST that releases MATH.

**Motivation.** Gotcha #4 from `debug-ideas.md`: PACK's existing `TTI_STALLWAIT(STALL_MATH, PACK)` orders MATH on PACK, but the subsequent `TT_ZEROACC` is dispatched onto the MATH/SFPU pipe while `_llk_packer_set_math_semaphore_<NONE>` issues `TTI_SEMPOST` onto the SYNC pipe — no ordering, so PACK can post MATH_PACK before ZEROACC retires. MATH wakes from `math_dest_wait()` and starts FPU writes into a bank still being cleared. Tightest single-instruction fix.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| Patch #4 | 0.9931898601668998 | 0.9934483676630709 |

Bit-identical to baseline. JIT cache hit rate 80.2% (~20% TRISC-compute kernels rebuilt with the new header), TRISC2 ELFs were re-emitted — the patch reached the JIT.

**Conclusion.** Null effect → timing-race hypothesis falsified. Also consistent with the conversation-summary ablation table: "Bilateral barriers only (no SW/HW resets) diverges" — barriers strictly stronger than this fence already failed, so this fence couldn't have succeeded. The bug is a **persistent state asymmetry**, not a pipe-ordering race.

**Next steps.** Move to Method 1 parity-bisect to localize *which* DEST consumer is bank-sensitive.

**State.** Reverted.

---

## Attempt 2 — Method 1 sweep at point (d): dummy before `moe_body`

**What.** Insert iter-0-only dummy `tile_regs_acquire/commit/wait/release` at `decoder_block_kernel.cpp:3093`, between `MOE_CB_RECONFIG` and `moe_body()`.

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

**Motivation.** Method 1 from `debug-ideas.md`. iter-0-only dummy adds +1 bank flip in iter-0 only, so iter-0 ends in the opposite bank parity vs baseline (and iter-1 inherits opposite starting parity). Effect on PCC localizes the bank-sensitive consumer: changed iter-0 PCC means consumer is downstream of the dummy. Point (d) brackets `moe_body` vs everything else.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| Dummy @ (d) | **0.9931898601668998 (unchanged)** | 0.9934946587532015 (new, +4.6e-5 vs baseline) |

**Conclusion.** iter-0 PCC unchanged → `moe_body` is bank-INSENSITIVE for its own MLP output (the bank parity at which moe_body's compute runs does not affect the final write to the MLP output buffer). iter-2 PCC changes (new third value, not a clean swap) because iter-0's end-state bank propagates into iter-1's starting bank, which does shift iter-1's downstream output via the KV-cache-write coupling.

**Next steps.** Move the dummy earlier — to top-of-iter — to test whether anything upstream of `moe_body` is bank-sensitive.

**State.** Removed.

---

## Attempt 3 — Method 1: dummy at TOP of iter (before `mla_body`)

**What.** Move the iter-0-only dummy to `decoder_block_kernel.cpp:3072`, just before `mla_body()`.

**Motivation.** This places every compute consumer downstream of the dummy. If iter-0 PCC changes, the bank-sensitive consumer is somewhere inside `mla_body` (pre-SDPA / `flash_mla` / post-SDPA) — since `moe_body` was just shown to be insensitive.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| Dummy @ TOP | **0.9935509776934355 (CHANGED, +3.6e-4)** | 0.9934946587532015 |

**Conclusion.** iter-0 PCC shifts → there is at least one bank-sensitive consumer inside `mla_body`. The bug is therefore in `mla_body` (RMSNorm/MCAST/MATMUL stack → `flash_mla` → post-SDPA / matmul4 / matmul5). iter-2 PCC matches Attempt 2 — same iter-1 starting bank inheritance, location of dummy in iter-0 doesn't matter for iter-1's value.

**Next steps.** Drop the dummy immediately before `flash_mla`'s `tile_regs_acquire` (`flash_mla.hpp:747`) — the first natural split inside `mla_body`. This separates pre-SDPA (matmuls/RMSNorms) from `flash_mla` itself.

**State.** Removed.

---

## Attempt 4 — Method 1: dummy at `flash_mla.hpp:747` (immediately before main acquire)

**What.** Inject a static-counter dummy in `flash_mla.hpp` right before `tile_regs_acquire()` at line 747. Uses a TRISC-local `static uint32_t debug_flash_mla_call_count` (the kernel-level `iteration` variable is not visible inside the templated `FlashMLADecode::Op` body); first call only fires the dummy.

**Motivation.** Brackets the entire pre-SDPA stack vs `flash_mla` itself. Same outcome as TOP-of-iter (still shifted) → entire pre-SDPA stack is bank-insensitive. Different outcome (back to baseline) → consumer is in pre-SDPA.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| Dummy @ pre-flash_mla (:747) | **0.9935509776934355 (same as TOP)** | 0.9934946587532015 |

**Conclusion.** Bit-identical to Attempt 3 → the **entire pre-SDPA stack is bank-insensitive**: RMSNorm, MCAST, the q_a / q_b / kv_a matmul chain, RMSNorm2, MATMUL2/3, QNOPE/QROPE projection, CreateQHeads, DKV matmul / gather, KV-RMSNorm, K-ROPE, and the KV-cache write are all bank-parity-clean. The bug is at `flash_mla.hpp:747` or downstream.

**Next steps.** Inject after `flash_mla`'s main `tile_regs_release` at `flash_mla.hpp:791` to separate `flash_mla` itself from the post-SDPA / matmul4-5 / reduce tail.

**State.** Removed.

---

## Attempt 5 — Method 1: dummy at `flash_mla.hpp:791` (immediately after main release, point (b))

**What.** Inject a static-counter dummy in `flash_mla.hpp` right after `tile_regs_release()` and the trailing `MATH(t6_semaphore_wait_on_max<...>(FPU_SFPU))` at line 793. This is between `flash_mla`'s main chunk-loop block and the tree-reduction tail at `:799-813`.

**Motivation.** Combined with Attempt 4, this brackets `flash_mla`'s main `acquire→release` region (chunk loop + per-chunk SFPU phases + tail + final pack) vs everything downstream (tree-reduction tail + post-SDPA + matmul4/5 + reduce + moe_body).

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| Dummy @ post-flash_mla (:791) | **0.9931898601668998 (BASELINE)** | 0.9934946587532015 |

**Conclusion.** iter-0 PCC returns to baseline → the bank-sensitive consumer is **inside `flash_mla.hpp:747-791`**, i.e. the `acquire→release` block that wraps the K-chunk loop (`compute_sdpa_chunk` × `num_chunks`), the SFPU recip / tail-pack, and the final `pack_block_contiguous(mm2)` loop. Every region downstream of `:791` (tree-reduction tail, post-SDPA reduce worker/forwarder, MATMUL4, GATHER2, MCAST3, MATMUL5, GATHER3, CCL all-reduce, MOE_CB_RECONFIG, moe_body) is bank-insensitive.

**Localization summary after Attempts 1–5:**

| Region | Bank-sensitive? |
|---|---|
| Pre-SDPA stack (RMSNorm/MCAST/MATMUL/QNOPE/QROPE/KV cache write) | No |
| **`flash_mla.hpp:747-791` (chunk loop + tail + final pack)** | **YES — bug zone** |
| Tree-reduction tail (`flash_mla.hpp:799-813`) | No |
| Post-SDPA / MATMUL4 / GATHER2 / MCAST3 / MATMUL5 / GATHER3 / CCL all-reduce | No |
| MOE_CB_RECONFIG, `moe_body` | No |

The bug zone is ~40 lines of TRISC code containing: QK^T (MM1), per-row max via SFPU reduce-row replay, broadcast subtract, exp_mul_prev correction, fast_approx_exp via SFPU per-tile, softmax·V (MM2 via `sdpa_custom_mm_reuse_dest_srcb_block` with granularity-paced semaphore posts), per-row sum via SFPU reduce-row replay, optional recip+rescale tail, and the final `pack_block_contiguous(mm2)`.

Method 1 cannot bisect further without modifying `flash_mla` itself — the doc explicitly warns nested `tile_regs_acquire/release` inside the existing block is unsafe (would deadlock or corrupt state). To localize further, switch to targeted reset-injection or to candidate-driven fixes.

**Next steps.** Try fixes from the open-candidate list in `half-dest-conversation-summary.md` §4, in order of cheapness:
- **2a:** initialize `DEST_OFFSET_LO+1..3` / `DEST_OFFSET_HI+1..3` in `_llk_init_packer_dest_offset_registers_` (currently only `+0` is set). Tests the "PACK section GPRs beyond PACK_SEC0..3 carry stale bank-parity-dependent state" hypothesis.
- **2b:** re-init the SFPU reduce-row replay buffers at the top of each `compute_sdpa_chunk` instead of once per kernel.
- **2c:** reset UNPACK DEST mirror in `sdpa_custom_mm_reuse_dest_srcb_block` (MM2) explicitly per chunk.

**State.** Removed.

---

## Attempt 6 — Patch 2a: initialize `DEST_OFFSET_LO/HI +1..+3` GPRs in `_llk_init_packer_dest_offset_registers_`

**What.** In `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h:69-71`, add six more `TTI_SETDMAREG` instructions to initialize `p_gpr_pack::DEST_OFFSET_LO + {1,2,3}` to `0` and `p_gpr_pack::DEST_OFFSET_HI + {1,2,3}` to `DEST_REGISTER_HALF_SIZE`. Previously only the `+0` slots were initialized.

```cpp
TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
```

**Motivation.** Top open-candidate from `half-dest-conversation-summary.md` §4 (#1). `select_packer_dest_registers<SyncHalf>` emits `WRCFG_128b` which writes PACK_SEC0..3 in one shot from four consecutive GPRs starting at `DEST_OFFSET_LO+0` or `DEST_OFFSET_HI+0` (see `cpack_common.h:659-668`). Currently `_llk_init_packer_dest_offset_registers_` only sets the `+0` slots — `+1..+3` carry whatever they happened to hold at kernel boot, which could differ between iterations or program-cache hits.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| Patch 2a | **0.9931898601668998 (bit-identical baseline)** | **0.9934483676630709 (bit-identical baseline)** |

**Conclusion.** Null effect → stale `PACK_SEC1..3` GPR contents are NOT the bug. Either `flash_mla` only uses packer 0 (so `PACK_SEC1..3` content is unread), or the stale values in those GPRs happen to be benign on this kernel layout. Candidate eliminated.

**Next steps.** Move to the next open candidate. Two reasonable options:
- **2b:** re-init the SFPU reduce-row replay buffers (`_init_sdpa_reduce_max_row_8x32_replay_buffers_` / `_init_sdpa_reduce_sum_row_8x32_replay_buffers_`) at every `compute_sdpa_chunk` invocation. Currently they're set up once per chunk too, but the *contents* persist across chunks and may encode a bank-parity-dependent address that gets reused.
- **2c:** explicitly reset UNPACK DEST mirror in `sdpa_custom_mm_reuse_dest_srcb_block` per chunk. MM2 reads DEST through SRCB; the SRCB-reuse base pointer is set via its own `TT_SETC16` (`llk_math_sdpa_custom_mm.h:101`, `llk_math_sdpa_custom_mm_reuse_dest_srcb.h:122,130`) which might not be aware of the bank flip.

**State.** Reverted.

---

## Attempt 7 — Patch 2c (variant): explicit `MATH_Offset` reset on both TRISC1 and TRISC2 at top of every `compute_sdpa_chunk`

**What.** In `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h:263-265`, add two explicit `TT_SETC16` calls at the top of every `compute_sdpa_chunk` invocation — one on MATH (TRISC1) and one on PACK (TRISC2) — that re-write each thread's `DEST_TARGET_REG_CFG_MATH_Offset` to the current bank base.

```cpp
MATH((TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base())));
PACK((TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset())));
```

**Motivation.** Note that the doc's literal "reset UNPACK DEST mirror" framing for 2c is misleading: UNPACK does not touch DEST. The MM2 op (`sdpa_custom_mm_reuse_dest_srcb_block`) reads DEST through SRCB via MOVD2B on the *MATH* side (`llk_math_sdpa_custom_mm_reuse_dest_srcb.h:119-145`), and that path is bank-aware via `get_dest_buffer_base()` captured at line 119. So the literal 2c is already correct. Adjacent test: force a known-good `MATH_Offset` on both threads at chunk entry, mirroring the explicit `matmul.hpp:167` idiom for TRISC2 alignment. If any consumer were operating from a stale `MATH_Offset` at chunk entry (despite the per-op `TT_SETC16` calls in the SDPA helpers), this would patch over it.

**Result.** First patch with a measurable effect — not a null.

| Case | iter=1 PCC | iter=2 PCC | iter alternation gap |
|---|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 | +2.6e-4 |
| Patch 2c (MATH+PACK MATH_Offset reset per chunk) | **0.9935941662184071** | **0.9935820920940954** | **−1.2e-5** |

Both PCCs shifted upward (+4.0e-4 on iter=1, +1.3e-4 on iter=2). The iter-alternation gap collapsed from +2.6e-4 to −1.2e-5 — roughly **20× smaller**, and the sign actually flipped (iter=2 is now marginally *below* iter=1 instead of above). JIT cache hit rate was 0.9% (essentially cold rebuild), so the patch absolutely went through.

**Conclusion.** 2c is the first patch that meaningfully attacks the bug. The iter-alternation is dramatically reduced but not eliminated. Two things this tells us:

1. **A non-trivial component of the iter-divergence does live in chunk-entry MATH_Offset state** — even though the per-op SDPA helpers each set `MATH_Offset = src + get_dest_buffer_base()` before their dispatch, the redundant pre-chunk reset is still doing something. Most likely candidates for *why* a redundant set helps: it provides a synchronization point on the CFG pipe before the per-op SETC16s land, OR there's a HW ordering effect between the per-op writes and immediately-following SFPU reads that benefits from an extra "settled" write earlier.

2. **A residual asymmetry of ~1e-5 remains** — much smaller than baseline, but still non-zero and sign-flipped. So 2c is not a complete fix; some bank-asymmetric state survives even with both per-chunk MATH_Offsets pinned.

Combined with Patch #4 (timing fence) and Patch 2a (PACK_SEC GPRs 1..3) being null:
- ❌ in-flight ZEROACC race
- ❌ stale PACK_SEC1..3 GPR contents
- ⚠️ **MATH_Offset chunk-entry state — partially the culprit; redundant per-chunk reset halves the bug**

**Next steps.** With the localization down to chunk-entry CFG-pipe state, two reasonable follow-ups:

- **2c-narrow:** find out which one of the two `TT_SETC16` calls (MATH or PACK) carries the effect. Run with just the MATH-side call, then just the PACK-side. Tells us which thread's chunk-entry MATH_Offset matters.
- **2c-deeper:** investigate the residual ~1e-5 gap. Candidates: tail path (`compute_sdpa_recip` + final `pack_block_contiguous(mm2)`, which is after the chunk loop and doesn't get the per-chunk reset), or per-chunk state in `init_fast_approx_exp_constants` (SFPCONFIG writes to LREG[12,13,14] inside each chunk), or replay-buffer interactions.

**State.** Reverted.

---

## Attempt 8 — Patch 2c + 2c-tail combined: same MATH+PACK MATH_Offset reset before the post-chunk-loop tail

**What.** Keep Attempt 7's per-chunk reset in `compute_sdpa_chunk`, and additionally inject the same MATH+PACK `TT_SETC16` pair in `flash_mla.hpp` right before the post-chunk tail (immediately before line 775 — the `if (!sdpa_output_is_final)` branch and the final `pack_block_contiguous(mm2)` loop).

```cpp
// In flash_mla.hpp, just before the tail at line ~775:
MATH((TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::get_dest_buffer_base())));
PACK((TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset())));
```

**Motivation.** The 2c chunk-entry reset collapses the iter-PCC gap ~20× but leaves a residual ~1e-5. The tail (recip + final pack of mm2, OR the tree-reduce-worker pack of max) runs *after* the chunk loop and never sees the per-chunk reset. If the residual asymmetry lives in chunk-exit / tail state, this extra reset should close the gap.

**Result.** Direct opposite of expected — both PCCs revert to **bit-identical baseline**:

| Case | iter=1 PCC | iter=2 PCC | gap |
|---|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 | +2.6e-4 |
| 2c alone (Attempt 7) | 0.9935941662184071 | 0.9935820920940954 | −1.2e-5 |
| **2c + 2c-tail** | **0.9931898601668998** | **0.9934483676630709** | **+2.6e-4 (=baseline)** |

JIT cache hit rate 0.9% — both patches genuinely compiled in. Adding 2c-tail completely **canceled** the 2c effect.

**Conclusion.** This is a counterintuitive but informative null. Two takeaways:

1. **2c's benefit is not from "putting `MATH_Offset` in a known-good state at chunk entry"** — if it were, adding the *same* known-good state at one more location (chunk exit) couldn't possibly hurt. The fact that it does hurt rules that interpretation out.

2. **The effect of 2c depends on inter-flash_mla state propagation.** The pre-tail reset changes the state that `flash_mla` leaves on TRISC1/TRISC2 at exit (specifically, the `MATH_Offset` registers and any associated CFG-pipe ordering). That exit state propagates into the *next* flash_mla invocation's starting state — apparently in a way that exactly cancels whatever 2c-at-chunk-entry was correcting. Result: baseline.

So 2c's mechanism is more subtle than "stale MATH_Offset at chunk entry." It's more like: the previous flash_mla leaves a specific MATH_Offset value behind, the next flash_mla's compute_sdpa_chunk inherits that as the starting MATH_Offset, and the chunk-entry reset overrides it with `get_dest_buffer_base()` *which happens to be the same value most SFPU helpers would set anyway*. Yet this re-write is doing something — possibly a CFG-pipe synchronization side effect, or a hidden interaction with how the per-op `TT_SETC16` calls are pipelined / committed.

**Next steps.** The 2c result is genuine but the mechanism is unclear; 2c-tail rules out "simple staleness". To make progress:

- **2c-narrow** (still planned): isolate MATH-only vs PACK-only side of 2c. If only one side carries the effect, we'd know whether it's a TRISC1 or TRISC2 CFG-pipe phenomenon. ~25 min total.
- **Inspect chunk-exit MATH_Offset state.** Add a single `MATH((TT_SETC16))` *only at chunk entry* (no PACK side), see if the asymmetric effect persists. Then try only the PACK side. Compare to 2c.
- **Capture which DEST tile is responsible.** Instrument the SDPA helpers to dump (via DPRINT / device_zone telemetry) which physical DEST index gets touched in iter-0 vs iter-1, look for a divergence beyond the bank-base offset.

**State.** Reverted (both 2c and 2c-tail patches gone).

---

## Attempt 9 — 2c-narrow MATH-only: chunk-entry MATH side only

**What.** Same as Attempt 7 but PACK-side `TT_SETC16` removed; only MATH-side fires at chunk entry.

**Motivation.** Isolate which CFG pipe carries the 2c effect — TRISC1 (MATH) only?

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| 2c-narrow MATH-only | **0.9931898601668998 (baseline)** | **0.9934483676630709 (baseline)** |

**Conclusion.** Null. MATH side alone is insufficient.

**State.** Reverted.

---

## Attempt 10 — 2c-narrow PACK-only: chunk-entry PACK side only

**What.** Same as Attempt 7 but MATH-side `TT_SETC16` removed; only PACK-side fires at chunk entry.

**Motivation.** Isolate which CFG pipe carries the 2c effect — TRISC2 (PACK) only?

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| 2c-narrow PACK-only | **0.9931898601668998 (baseline)** | **0.9934483676630709 (baseline)** |

**Conclusion.** Also null. Combined with MATH-only being null, **2c requires BOTH MATH and PACK writes together** — it's a cross-thread CFG-pipe interaction, not a single-thread effect.

**State.** Reverted.

---

## Attempt 11 — 2c-move pre-MM2: MATH+PACK pair just before `sdpa_custom_mm_reuse_dest_srcb_block`

**What.** Keep both MATH and PACK `TT_SETC16` writes, but move them from chunk entry (top of `compute_sdpa_chunk`) to right before MM2 (line 313 in `sdpa.h`).

**Motivation.** MM2 is the DEST-via-SRCB op — the most direct suspect for bank-asymmetric reads. If the pair only matters because of MM2, moving it right next to MM2 should preserve the 20× collapse.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| 2c-move pre-MM2 | **0.9931898601668998 (baseline)** | **0.9934483676630709 (baseline)** |

**Conclusion.** Null. Position before MM2 (mid-chunk, after MM1 + max-reduce + sub + exp) gives no effect. So MM2 itself is not what benefits from the cross-thread CFG writes.

**State.** Reverted.

---

## Attempt 12 — 2c-move chunk-end: MATH+PACK pair at end of `compute_sdpa_chunk`

**What.** Move the pair from chunk entry to chunk exit (after `cb_pop_front(cb_k, ...)`, before the closing brace of `compute_sdpa_chunk`).

**Motivation.** Functionally adjacent to "chunk entry of next chunk" — only difference is the very first chunk (no preceding chunk-end) and the very last chunk (chunk-end fires but no following chunk). Tests whether per-chunk-frequency matters or specifically chunk-entry-of-chunk-0.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| 2c-move chunk-end | **0.9931898601668998 (baseline)** | **0.9934483676630709 (baseline)** |

**Conclusion.** Null. So the difference between "chunk-entry-of-chunk-i" and "chunk-end-of-chunk-(i-1)" matters — the pair must fire BEFORE chunk 0's first work, not after the last chunk's last work. Either the placement before chunk 0 is unique, or there's an inter-chunk subtlety that only the literal "chunk-entry" position satisfies.

**State.** Reverted.

---

## Attempt 13 — 2c-move flash_mla-entry: MATH+PACK pair ONCE before chunk loop

**What.** Single MATH+PACK reset placed in `flash_mla.hpp` at line ~746, right before `tile_regs_acquire` and the chunk loop. Fires once per flash_mla call (not per chunk).

**Motivation.** If "before chunk 0" is the magic, then doing it ONCE in that position should work. If null, then per-chunk frequency is essential.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| 2c-move flash_mla-entry | **0.9931898601668998 (baseline)** | **0.9934483676630709 (baseline)** |

**Conclusion.** Null. So neither "before chunk 0 only" nor "before each chunk anywhere in compute_sdpa_chunk" is sufficient on its own. The unique-working recipe is:
- **Both** MATH and PACK `TT_SETC16` writes,
- Fired **every chunk**,
- Placed **at the very top of `compute_sdpa_chunk`** (immediately before `_init_sdpa_reduce_max_row_8x32_replay_buffers_`),
- **Not** also repeated at the tail.

Any deviation (MATH-only, PACK-only, mid-chunk, chunk-end, once-per-flash_mla, also-at-tail) gives baseline.

**State.** Reverted.

---

## Attempt 14 — NOP test: 2c's position with semantic-inert writes

**What.** Same chunk-entry position as Attempt 7, but the two `TT_SETC16` writes replaced with semantic NOPs emitted through the same `ckernel::instrn_buffer[0]` write mechanism that `TT_SETC16` uses internally.

```cpp
MATH((ckernel::instrn_buffer[0] = TT_OP_NOP));
PACK((ckernel::instrn_buffer[0] = TT_OP_NOP));
```

This compiles to the same instruction-stream shape (host CPU writes a 32-bit value into the Tensix instruction buffer for that thread; thread dispatches it) but the dispatched instruction is a NOP, not a CFG write. Same JIT pattern, different semantics.

**Motivation.** Discriminate (A) real CFG-pipe semantic effect vs (B) code-gen / instruction-layout side effect. If iter-PCC shifts ~20× with NOPs, the position itself matters and 2c is a code-gen artifact. If it stays at baseline, the actual `SETC16` writes carry the effect.

**Result.**

| Case | iter=1 PCC | iter=2 PCC |
|---|---|---|
| Baseline | 0.9931898601668998 | 0.9934483676630709 |
| 2c (real `SETC16` writes) | 0.9935941662184071 | 0.9935820920940954 |
| NOP test | **0.9931898601668998 (baseline)** | **0.9934483676630709 (baseline)** |

**Conclusion.** NOPs at the same position → bit-identical baseline. **Interpretation A is correct: the `SETC16` writes themselves carry the effect, not the code-gen side effect.** 2c is a credible lead — the cross-thread CFG writes really do change HW state in a way that mitigates the bug.

**State.** Reverted.

---

## Where to go next (handoff)

### What we know

The Method 1 bisect localized the bug to `flash_mla.hpp:747-791`. Patch sweep:

| Patch | Result |
|---|---|
| #4 (STALLWAIT before MATH_PACK SEMPOST) | null |
| 2a (init PACK_SEC1..3 GPRs) | null |
| **2c (per-chunk MATH+PACK MATH_Offset reset at chunk entry)** | **20× alternation collapse, residual ~1e-5** |
| 2c-tail (same reset additionally before flash_mla tail) | canceled 2c effect (baseline) |
| 2c-narrow MATH-only | null |
| 2c-narrow PACK-only | null |
| 2c-move pre-MM2 | null |
| 2c-move chunk-end | null |
| 2c-move flash_mla-entry (once before chunk loop) | null |
| NOP test (Attempt 14) | null — confirms 2c is semantic, not code-gen |

The unique working recipe is *cross-thread MATH+PACK pair, per-chunk, at chunk entry (before replay-buf init), NOT also at tail*. Any single variation breaks it. The NOP test confirms it's the real `SETC16` semantics doing the work.

### What we don't know

- **Why per-chunk frequency matters.** Once-per-flash_mla is null; per-chunk is 20×. The cumulative writes do something the single write can't.
- **Why both threads are required.** Each thread's `dest_offset_id` is independent and the per-op SDPA helpers already set `MATH_Offset` per dispatch — so a single redundant write per thread shouldn't do anything new, yet only the cross-thread pair works.
- **Why position-before-replay-buf-init is the unique-good spot.** Pre-MM2, chunk-end, flash_mla-entry are all null. There's something about *writes pre-pending the `_init_sdpa_reduce_max_row_8x32_replay_buffers_` call* in each chunk.
- **Why adding more (2c-tail) cancels.** If `SETC16` writes are doing semantic good, doubling down should help, not hurt. The cancellation hints at a parity-style effect on some HW counter.
- **What the residual ~1e-5 gap represents.** Some bank-asymmetric effect persists even with 2c.

### Hypotheses to smoke out the bug in the ~40 lines

The bug zone is `flash_mla.hpp:747-791`: `tile_regs_acquire` → chunk loop (`compute_sdpa_chunk` × N) → tail (`compute_sdpa_recip` or pack-max) → final `pack_block_contiguous(mm2)` loop → `tile_regs_commit/wait/release`. ~40 lines of TRISC1/MATH + TRISC2/PACK activity.

Hypotheses, roughly cheap → expensive:

1. **The `SETC16` writes are draining a CFG pipe that needs draining per chunk.** Triggers a barrier the existing code lacks. Test: replace the pair with a single `TTI_STALLWAIT(STALL_CFG, THCON)` (or similar) on each thread at the same position. If that also gives ~20× collapse, the effect is the CFG-pipe drain, not the address write — and we can simplify the fix.

2. **`load_replay_buf` records a bank-sensitive `MATH_Offset` snapshot at the time of `lltt::record`.** SFPLOAD instructions in the replay buffer use `Imm10 + MATH_Offset_at_replay_time`, but maybe the recording itself captures something about CFG state. Test: bracket the `_init_sdpa_reduce_max_row_8x32_replay_buffers_` call with `TTI_STALLWAIT` before AND after, on PACK only. See whether that alone reproduces 2c.

3. **Parity in some hidden counter.** 2c writes 16 times per flash_mla (2 threads × 8 chunks). 2c+2c-tail writes 18 times. If a HW counter is sensitive to count-mod-2 or count-mod-some-N, that explains the cancellation. Test: vary the count by adding 1 extra MATH+PACK pair to one specific chunk (say chunk 0 fires the pair twice). If iter-PCC moves toward baseline at every odd extra, parity hypothesis confirmed.

4. **Reduce to 1 chunk via small `position_id`.** The test currently uses `position_id=8190` → ~8 chunks per device. The parametrize comment-out includes `127`, `511`, `1023`. With `position_id=127` (1 chunk per device), the per-chunk-frequency story collapses. If iter-alternation still appears at 1 chunk, the bug is inside `compute_sdpa_chunk` itself (or its tail). If it disappears at 1 chunk, the bug is cross-chunk state.

5. **The bug lives in the tail (`compute_sdpa_recip` / final pack), not the chunk loop.** Method 1 only bracketed `:747-791` as a unit — we couldn't bisect inside because nested `tile_regs_acquire/release` would deadlock. But we *can* test by short-circuiting the tail: replace `compute_sdpa_recip` with a hardcoded no-op (or a different code path) only on iter 0, see if iter-PCC alternation moves.

6. **Asymmetric SFPU helper.** Each SDPA SFPU helper (`fast_approx_exp`, `non_approx_exp_mul_prev`, `recip_sum`, `reduce_max_row`, `reduce_sum_row`) does its own `TT_SETC16(MATH_Offset, src + base)`. If one of them has a bug where it reads/writes the wrong DEST tile (e.g., off-by-one), it would be bank-asymmetric. Audit each: print the actual computed `src + base` value vs the intended tile offset, look for a discrepancy.

7. **`init_fast_approx_exp_constants` is called per chunk and writes to LREG[12..14] via SFPCONFIG. If the SFPCONFIG state isn't fully drained / sequenced before the next op, LREG content could carry across chunks.** Test: remove `init_fast_approx_exp_constants` from the chunk and inline the constant loads. Or call it ONCE at flash_mla top instead of per-chunk.

8. **DEST-bank readback microbench.** Skip the whole investigation — write a minimal kernel: FPU + SFPU pass into bank 0, dump to L1, repeat into bank 1, dump. Diff. Confirms / refutes any HW asymmetry.

My pick for next experiment: **hypothesis 4 (small position_id)**. Cheapest information-per-time-spent — single parameter change, no code edits, ~12 min run. Decides whether the bug needs cross-chunk state at all.

---
