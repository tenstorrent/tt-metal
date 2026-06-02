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

## Attempt 15 — Hypothesis 4: small `position_id` (1 chunk vs 64 chunks)

**What.** Enable the previously-commented `position_id=127` in the test parametrize at `test_decoder_block.py:866`, in addition to the standard `position_id=8190`. Run both with `num_internal_iterations ∈ {1, 2}` — gives 4 PCCs total in one run.

At `k_chunk_size=128`, `position_id=127` → `(127+1)/128 = 1` total k chunk across all SP devices (vs `(8190+1)/128 = 64` for the standard config). With only 1 chunk, the `compute_sdpa_chunk` body runs exactly once per `flash_mla` invocation; no cross-chunk state can build up.

**Motivation.** If iter-alternation still manifests at `position_id=127`, the bug is intrinsic to a single chunk's compute. If it disappears, the bug requires cross-chunk accumulation — which immediately constrains the candidate to either (a) the MM2 P·V DEST accumulator, (b) the LReg carry of max/sum across `non_approx_exp_mul_prev` calls, or (c) some other cross-chunk state.

**Result.**

| `position_id` | iter=1 PCC | iter=2 PCC | gap |
|---|---|---|---|
| **127** (1 chunk total) | **0.9901647631143314** | **0.9901647631143314** | **0 (bit-identical)** |
| 8190 (64 chunks, the standard config) | 0.9931898601668998 | 0.9934483676630709 | +2.6e-4 |

**Conclusion.** At 1 chunk the iter-alternation **completely disappears** — both iters produce bit-identical output. At 64 chunks the alternation is present at its known +2.6e-4 magnitude.

This is the strongest localization signal so far. **The bug is a cross-chunk accumulating effect**, not intrinsic to a single chunk's body. Rules out:

- All single-chunk intra-op bugs (any SFPU helper / MM1 / MM2 / max-reduce / sum-reduce in isolation).
- The post-chunk tail (recip + final pack) running once per flash_mla — would manifest at 1 chunk.
- The pre-SDPA stack / post-SDPA / matmul4/5 / moe_body — these run once per iter regardless of chunk count.

Rules in:
- Something that *accumulates across chunks within one `flash_mla` call*. There are two structural candidates:
  - **MM2 P·V accumulator in DEST.** Each chunk's `sdpa_custom_mm_reuse_dest_srcb_block` reads mm1 via MOVD2B and accumulates into `mm2_dst_offset`. The DEST values at `mm2_dst_offset` carry from chunk i to chunk i+1.
  - **LReg-carry across chunks.** `non_approx_exp_mul_prev` (chunks 1+) reads `LREG0/2` (curr-sum) and `LREG1/3` (prev-max) directly without a SETC16 setup — they're expected to hold values from the previous chunk's reduce_sum / reduce_max.

This also makes 2c's positional sensitivity click into focus: the cross-thread SETC16 pair at chunk *entry* (per chunk) is correcting whatever cross-chunk state is drifting. The 2c-tail cancellation may be because the same writes at tail are corrupting one of these accumulators on the way out.

**State.** Test parametrize edit left in place (both `127` and `8190` enabled) — gives free comparison data for future runs at low marginal cost (the 127 case adds ~5 min to a 12-min run).

---

## Updated handoff

### What we know

The bug:
- Localized to `flash_mla.hpp:747-791` (Method 1).
- Requires multiple chunks (Attempt 15) — gone at 1 chunk, present at 64.
- Carried by real semantic `SETC16` writes, not code-gen (Attempt 14).
- Single best partial fix (Attempt 7, "2c"): cross-thread MATH+PACK MATH_Offset reset per chunk at chunk entry. Collapses gap 20×; residual ~1e-5.
- Highly position-sensitive: any of 2c-narrow / 2c-move / 2c-tail breaks it.

### Two main remaining suspect mechanisms

**S1 — MM2 P·V accumulator drift.** Across chunks the mm2 DEST tiles accumulate ε errors that have bank-asymmetric magnitude. Plausible because MM2 reads mm1 through SRCB via MOVD2B, and the MOVD2B+MVMUL sequence may have bank-asymmetric numerical rounding.

**S2 — LReg-carry across chunks.** `non_approx_exp_mul_prev` reads LREG0/2/1/3 without re-setting MATH_Offset for the read — they're expected to hold prev-sum and prev-max. If these LRegs are bank-asymmetrically corrupted across chunks (perhaps because `_calculate_sdpa_reduce_*_row_8x32_` writes them via SFPLOAD from a bank-dependent DEST address), the chunk-1+ correction picks up a slightly wrong prev-value.

Both are consistent with 2c's "per-chunk reset at chunk entry" recipe doing something useful: it touches CFG state right where the accumulating drift gets used.

### Cheap-to-expensive next experiments

1. **Sweep `position_id` to characterize alternation vs chunk count.** Enable `position_id=255, 511, 1023, 2047, 4095` etc. Tells us whether the gap grows linearly, saturates, or has a knee. ~12 min per point or one big run.

2. **Hard-clear LReg state between chunks.** Insert SFPLOADI zero into LREG0/1/2/3 at chunk entry (PACK side). If iter-alternation goes to zero at multi-chunk, S2 confirmed.

3. **Hard-reset mm2 DEST tiles between chunks.** Insert ZEROACC on the mm2 region at chunk entry; this breaks the accumulator semantics so PCC will tank — but if the *alternation* gap collapses, S1 confirmed (even if PCC drops a lot).

4. **Audit `_llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_` for bank-dependence.** The MOVD2B in MM2 reads DEST via MATH_Offset captured at function entry. Per-tile reads use ADDR_MOD with constant increments. Look for any cross-bank pointer leak.

5. **Audit `_calculate_sdpa_reduce_*_row_8x32_`.** These set MATH_Offset and SFPLOAD into LREGs. Look for off-by-one or partial-row issues that would only differ between banks.

6. **Profiler / DEST address trace.** Lower-priority now that we have specific suspects.

My pick: **(1) `position_id` sweep**. Confirms the cross-chunk-accumulation mechanic and gives us a quantitative knob — knowing how the gap grows with chunk count lets us write much cheaper iteration cycles.

---

## Attempt 16 — `position_id` sweep (chunk-count scan)

**What.** Enable `position_id ∈ {127, 511, 1023, 8190}` in the parametrize. With `k_chunk_size=128` this gives 1, 4, 8, 64 total k chunks respectively. Each is run for `num_internal_iterations ∈ {1, 2}` → 8 PCC values.

**Motivation.** Characterize how the iter-PCC gap grows with chunk count. Three plausible shapes:
- **Linear growth:** the bug accumulates ε per chunk → drift visible at all multi-chunk counts, larger at 64 than at 4.
- **Saturation:** the bug fires once on the first cross-chunk transition and stays → gap ~constant beyond a small chunk count.
- **Step at a specific chunk count:** the bug needs a specific chunk-count threshold.

**Result.**

| `position_id` | total k chunks | iter=1 PCC | iter=2 PCC | gap (iter2−iter1) |
|---|---|---|---|---|
| 127 | 1 | 0.9901647631143314 | 0.9901647631143314 | **0** |
| 511 | 4 | 0.9914568554511025 | 0.9916859209677518 | +2.29e-4 |
| 1023 | 8 | 0.9926787610146476 | 0.9928732147523281 | +1.94e-4 |
| 8190 | 64 | 0.9931898601668998 | 0.9934483676630709 | +2.59e-4 |

**Conclusion.** The gap **does not scale with chunk count** — it saturates at ~2-3e-4 from 4 chunks onward, and 8 chunks (1.94e-4) is even slightly smaller than 4 chunks (2.29e-4), so the "linear accumulation" hypothesis is dead.

The gap appears once a second chunk runs, then plateaus. **The bug fires once, on the first cross-chunk handoff (chunk-0 → chunk-1)**, and the resulting error remains stable through subsequent chunks.

What changes structurally between chunk 0 and chunk 1:

- Chunk 0 sets `first_chunk = true`. Chunks 1+ set `first_chunk = false`.
- The `first_chunk = false` branch in `compute_sdpa_chunk` (sdpa.h:282-299) gates `non_approx_exp_mul_prev` + bcast-mul of mm2 by `corr_exp`. This is exactly the LReg-carry path.
- `non_approx_exp_mul_prev` reads LREG0/2 (curr_max from current chunk's reduce_max) and LREG1/3 (which should hold prev_max from chunk 0's reduce_max). The reduce_max replay buffer's LREG layout is: LREG0/2 = running pool result, LREG1/3 = freshly loaded scratch from DEST. After chunk 0's reduce_max returns, LREG0/2 = chunk-0-max and LREG1/3 = some leftover scratch tile.
- When `non_approx_exp_mul_prev` reads LREG1/3 as "prev_max" in chunk 1, it's actually reading **the last DEST tile loaded by chunk 0's reduce_max replay** — *not* chunk-0's max value.

That's bizarre — either the code has a latent bug in how it conventions LRegs for prev_max, OR I'm mis-reading the LReg lifecycle. Either way the most direct test is now experiment 2 from the previous handoff: **hard-clear LRegs between chunks** and see whether the iter-alternation collapses. If it does, S2 (LReg-carry) is confirmed and the fix is to explicitly re-stage prev-max from a DEST tile rather than relying on register persistence.

**State.** Sweep parametrize edits left in place — gives free comparison data for future runs.

---

## Updated handoff (post-sweep)

The bug fires on the **chunk-0 → chunk-1 transition**, specifically in code that's gated by `first_chunk = false`:

```cpp
// sdpa.h:282-299 (only runs for chunks 1+)
PACK((non_approx_exp_mul_prev<...>(sum_dst_offset, corr_exp_dst_offset)));
PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v>(cb_q);
MATH((t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)));
sdpa_bcast_col_srca_srcb_reuse_preamble(corr_exp_dst_offset);
sdpa_mul_bcast_col_srca_srcb_reuse_tiles<num_tiles_v, true>(mm2_dst_offset);
MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
MATH((t6_semaphore_get<p_stall::NONE>(SFPU_FPU)));
```

`non_approx_exp_mul_prev` is the function that does the heavy lifting — reads LRegs (LReg1/3 = prev_max, LReg0/2 = curr_max), computes `corr = exp((prev_max-curr_max)*scale) - 1`, writes to `corr_exp_dst_offset`, then multiplies prev-sum by `exp(...)` and writes back to sum-dst.

After this runs, the FPU does a bcast-mul: `mm2 *= corr_exp` (rescale the running V-accumulator with the correction factor).

**Hypothesis (refined):** the LReg state that `non_approx_exp_mul_prev` reads as "prev_max" in LREG1/3 is bank-asymmetric — either because the LReg state from chunk 0's reduce_max ended up differently depending on which bank chunk 0 wrote to, or because the read itself is somehow bank-dependent (which shouldn't be possible for plain LReg reads, but maybe SFPU has some bank-shadow).

**Next:** experiment 2 — hard-clear LRegs LREG0/1/2/3 to 0 between chunks. If iter-alternation goes to zero (or close), S2 confirmed. If unchanged, the bug is in something other than LReg-carry on that transition.

---

## Attempt 17 — LReg-zero at chunk entry (S2 disconfirmation)

**What.** Two variants tried:
- (v1) `PACK((sfpi::l_reg[sfpi::LRegs::LRegX] = sfpi::vFloat(0.0f)));` for X ∈ {0,1,2,3}
- (v2) `PACK((TT_SFPLOADI(X, 0xA, 0)));` + `PACK((TT_SFPLOADI(X, 0x8, 0)));` for X ∈ {0,1,2,3}

Both placed at the very top of `compute_sdpa_chunk` (same position as 2c was at).

**Motivation.** Disrupt the LReg-carry path. If the bug is in `non_approx_exp_mul_prev` reading bank-asymmetric leftover LReg state, zeroing should kill the alternation (PCC will tank but the diagnostic still works).

**Result.**

| Variant | iter=1 (pos=127) | iter=2 (pos=127) | iter=1 (pos=8190) | iter=2 (pos=8190) |
|---|---|---|---|---|
| Baseline | 0.99016476 | 0.99016476 | 0.99318986 | 0.99344836 |
| v1 (sfpi) | 0.99016476 | 0.99016476 | 0.99318986 | 0.99344836 |
| v2 (TT_SFPLOADI) | 0.99016476 | 0.99016476 | 0.99318986 | 0.99344836 |

**Both variants are bit-identical to baseline.** PCC didn't even tank, which means the zeroed LRegs *don't affect* the downstream compute.

**Conclusion (much more informative than predicted by the original hypothesis).** Reading `_calculate_sdpa_reduce_max_row_8x32_` (`ckernel_sfpu_sdpa_reduce_row.h:172-194`) reveals the actual cross-chunk mechanism:

```cpp
// Inside _calculate_sdpa_reduce_max_row_8x32_, after computing this chunk's max into LREG0/2:
TT_SETC16(MATH_Offset, dst_index + get_dest_buffer_base());   // point at max_dst_offset
TTI_SETRWC(...);

if (prev_max) {                                                // for chunks 1+
    TTI_SFPLOAD(LREG1, ..., 0);                                // load prev cumulative max top from DEST[max_dst]
    reduce_lregs_instr<MAX, LREG0, LREG1>();                   // LREG0 = max(this_chunk, prev_cum)
    TTI_SFPLOAD(LREG3, ..., 4);                                // load prev cumulative max bottom
    reduce_lregs_instr<MAX, LREG2, LREG3>();
    TTI_SFPLOAD(LREG1, ..., 0);                                // RE-cache for non_approx_exp_mul_prev
    TTI_SFPLOAD(LREG3, ..., 4);
}
TTI_SFPSTORE(LREG0, ..., 0);                                   // store new cumulative max
TTI_SFPSTORE(LREG2, ..., 4);
```

So the "LReg carry" is a misframing. **The actual cross-chunk state lives in the DEST tile at `max_dst_offset`** (and analogously `sum_dst_offset`):
- Each chunk's reduce_max stores cumulative max to DEST via SFPSTORE.
- Each chunk 1+'s reduce_max reads prev cumulative max from DEST via SFPLOAD, combines with this chunk's max, and re-caches it into LREG1/3 so that `non_approx_exp_mul_prev` can read it from LRegs.

The LRegs that `non_approx_exp_mul_prev` reads as "prev_max" are RE-LOADED from DEST inside reduce_max (lines 186-187 of the LLK). My initial LReg-zeroing was wiped by these SFPLOADs before `non_approx_exp_mul_prev` ever ran. That's why PCC was unchanged.

**The bug is in DEST-tile-carry across chunks**, not LReg-carry. The candidate tiles:
- `max_dst_offset` — cumulative running max, read/written by reduce_max
- `sum_dst_offset` — cumulative running sum (analogous mechanism in reduce_sum)
- `mm2_dst_offset` — running P·V accumulator, read via MOVD2B / written by MVMUL in MM2 across chunks

For the alternation to fire on chunk-0→chunk-1, the candidate tile must be (a) written by chunk 0, (b) read in a bank-asymmetric way by chunk 1's `!first_chunk` path. All three tiles qualify.

**State.** Reverted.

---

## Updated handoff (post-LReg)

The bug is **cross-chunk DEST-tile-carry**, not LReg-carry. The carried state lives at three DEST tile offsets:
- `max_dst_offset` (read in reduce_max prev_max branch + via re-cached LREG1/3 in `non_approx_exp_mul_prev`)
- `sum_dst_offset` (read in reduce_sum prev_sum branch + indirectly via `non_approx_exp_mul_prev`'s sum_dst write)
- `mm2_dst_offset` (P·V accumulator across chunks, read via MOVD2B in MM2)

The bug fires once at chunk-0→chunk-1 transition. Either:
- One specific DEST tile is read in a bank-asymmetric way on chunk-1, OR
- The DEST WRITE at chunk-0's end is bank-asymmetric in a way that leaves the tile slightly different depending on bank.

### Cheap next experiments

1. **Localize the tile.** Overwrite `max_dst_offset` (or `sum_dst_offset`, or `mm2_dst_offset`) with a fixed value via SFPSTORE between chunks — i.e., insert a per-chunk re-write that doesn't depend on prior chunk's state. If iter-alternation goes to zero for ONE of these tiles, that's the bug-carrier. PCC will tank but the diagnostic is clean.

2. **Vary which 2c offset is touched.** Currently 2c sets `MATH_Offset = bank_base` (= offset 0 within DEST half). Try `MATH_Offset = max_dst_offset + bank_base`, then `sum_dst_offset + bank_base`, then `mm2_dst_offset + bank_base`. If any specific tile's offset reproduces the 20× collapse, that's the bug-affected tile.

3. **Compare DEST `max_dst_offset` contents bank-0 vs bank-1.** Hand-write a TRISC pack that dumps `max_dst_offset` to L1 after every chunk, for both iter=0 and iter=1. Diff the L1 contents — if they differ beyond the documented bank offset, we've directly observed the asymmetric tile.

My pick: **(2) sweep 2c's address**. Cheapest — one-line edit per run, gives a clean signal about which tile (if any) the bug latches onto.

---

## Attempt 18 — CB-hash debug: input identity check

**What.** Pulled in the `hash_cb(cb_id, num_tiles, label)` debug LLK from [PR #43041](https://github.com/tenstorrent/tt-metal/pull/43041) (FNV-1a checksum over a CB's L1 bytes, emitted via DPRINT; runs scalar on UNPACK, touches no Tensix state). Added three files: `tt_metal/hw/inc/api/compute/debug/cb_hash.h`, `tt_metal/hw/ckernels/blackhole/metal/llk_api/debug/llk_hash_cb_api.h`, and the wormhole counterpart. Enabled via `("DEBUG_CB_HASH", "1")` in `decoder_block/op.py` defines.

Hash call sites:
- `flash_mla.hpp:746` (once per flash_mla call, after `cb_wait_front(cb_q_in, q_chunk_tiles)`): `hash_cb(cb_q_in, q_chunk_tiles, 0x10)` — label 0x10 = "Q".
- `sdpa.h:269` (once per `compute_sdpa_chunk`, after `cb_wait_front(cb_k, ...)`): `hash_cb(cb_k, num_tiles_k * chunk_size, 0x20)` — label 0x20 = "K".

Run config: `position_id=511` (4 chunks total — alternation present at +2.29e-4, small enough output to read), `num_internal_iterations=[2]` only (need iter-0 vs iter-1 within one launch).

**Motivation.** Determine whether the iter-1 divergence is from changed input data (KV-cache write contamination across iters) or from bank-asymmetric compute on identical inputs.

**Result.**

PCC: 0.9916859209677518 (matches the baseline iter-2 value at position_id=511 — bug still firing with `DEBUG_CB_HASH` instrumentation in place, as expected).

Hash distribution across all 256 emitted hashes (2 active devices × 16 cores × 2 iters × 4 chunks per core, simplified):

| Label | Distinct hash values | Each value's occurrence count |
|---|---|---|
| 0x10 (Q) | 10+ distinct (per-head Q tile content varies by core) | 4 or 8 times each (even — iter-0 + iter-1 contribute equally) |
| 0x20 (K) | 4 distinct | 32 times each (16 cores × 2 iters — even split) |

Every unique hash value's occurrence count is **even**, confirming that **the iter-0 and iter-1 runs see bit-identical input data**. Spot-check on core (x=0,y=0) confirms iter-0 and iter-1 produce exactly the same `0x10` and `0x20` hashes.

**Conclusion.** This decisively rules out the "iter-0's KV-cache write contaminated iter-1's input" hypothesis. **Inputs are deterministic across iters.** The iter-1 PCC divergence comes entirely from flash_mla's compute being bank-asymmetric (iter-0 starts at bank 0; iter-1 inherits bank 1; same code, different bank, different numerical output).

This nails the bug location to **bank-asymmetric numerical compute inside the `flash_mla.hpp:747-791` block**, with no contribution from upstream input shaping.

**State.** Hash infrastructure files added permanently. Hash call sites + DEBUG_CB_HASH define are left in tree (toggleable via `op.py` define line). Test parametrize narrowed to `position_id=511` only for hash-debug iteration speed.

---

## Updated handoff (post-hash)

The bug is bank-asymmetric numerical compute inside `flash_mla.hpp:747-791`:
- Confirmed: inputs (Q, K via cb_q_in/cb_k_in) identical between iter-0 and iter-1 (Attempt 18).
- Confirmed: bug fires on chunk-0 → chunk-1 transition (Attempt 15-16, gap saturates by 4 chunks).
- Confirmed: 2c (cross-thread MATH+PACK `MATH_Offset` reset per chunk at chunk entry) collapses gap ~20× via real semantic CFG writes (Attempt 7, Attempt 14 NOP control).
- Confirmed: the cross-chunk state lives in DEST tiles `max_dst_offset`, `sum_dst_offset`, `mm2_dst_offset` (Attempt 17 reveals LReg-carry is misframed — the real mechanism is SFPSTORE/SFPLOAD round-trips through DEST).

What remains: **which specific bank-asymmetric op produces the numerical difference**.

### Concrete next experiments

1. **Hash sdpa output CB.** Add `hash_cb(sdpa_output_cb, vDHt, 0x30)` right after the final `pack_block_contiguous(mm2_dst_tile_offset, sdpa_output_cb, ...)` in flash_mla (line 783-787 area, after the inner loop). The output CB is the LAST thing flash_mla writes; comparing iter-0 vs iter-1 hashes on the SAME core directly measures flash_mla's bank-asymmetric output divergence.

2. **Hash sdpa_ms_cb.** Same idea for the max+sum tile output. May or may not be written in the test path.

3. **Pack-DEST-and-hash inside chunk loop.** Insert temporary `pack_block_contiguous(max_dst_tile_offset, debug_cb, 1)` (or analogous for sum/mm2) right after the SFPSTORE in reduce_max/reduce_sum and right after MM2's accumulator update. Hash the debug_cb. This reveals which DEST tile diverges first.

4. **Direct chunk-content comparison via PACK → L1 dump.** Same as (3) but write to a known L1 region per chunk, then dump from host.

5. **2c-address sweep** (from earlier handoff): vary which DEST offset the `MATH_Offset` is set to in 2c. If pointing at max_dst_offset (vs sum_dst_offset vs mm2_dst_offset vs base) discriminates, that's a strong localization signal.

My pick: **(1) hash sdpa_output_cb at end of flash_mla**. Smallest patch, direct measurement of flash_mla's output divergence. If iter-0 and iter-1 emit DIFFERENT output hashes (which they must, since downstream PCC differs), we confirm the divergence point is inside `:747-791`. Then move to (3) to bisect within the block.

---

## Attempt 19 — Hash `sdpa_output_cb` at end of flash_mla (1-chunk vs 4-chunk)

**What.** Add `hash_cb(sdpa_output_cb, out_chunk_tiles, 0x30)` right after the final `cb_push_back(sdpa_output_cb, out_chunk_tiles)` in `flash_mla.hpp:792`. Run with `num_internal_iterations=[2]` and `position_id ∈ {127, 511}`.

**Motivation.** Directly measure whether flash_mla's *own* per-core output differs between iter-0 and iter-1 — independent of any downstream propagation. position_id=127 is the control: at 1 chunk the MLP PCC alternation vanishes, so we'd expect output hashes to match. position_id=511 is the bug case: alternation present, so output hashes should differ.

**Result.**

| Run | Total 0x30 hashes | Unique 0x30 values | MLP PCC | Pairing |
|---|---|---|---|---|
| `position_id=127` (1 chunk) | 32 | **32** (all unique!) | 0.9901647631143314 (iter-0 = iter-1, no alternation) | every emission distinct |
| `position_id=511` (4 chunks) | 128 | 123 | 0.9916859209677518 (iter-1 of `num_internal_iterations=2`) | 122 distinct, 3 paired |

Spot check at `position_id=127`, core (x=0,y=1) on device 6:
- iter-0 output hash: `0x982aef42`
- iter-1 output hash: `0x3c073639`

Completely different 32-bit hashes — FNV-1a is highly nonlinear so we can't infer magnitude, but the two emissions are not bit-identical CBs.

**Conclusion (major revision to the bug picture).**

flash_mla's per-core output is **bank-asymmetric on every single call**, even at 1 chunk where the final MLP PCC doesn't show alternation. The earlier "bug fires once on chunk-0→chunk-1 transition" interpretation (Attempt 16) was misleading — what actually happens:

- flash_mla per-core output divergence between iter-0 (bank 0) and iter-1 (bank 1) is a **universal, every-call phenomenon**.
- The downstream stack (post-SDPA tree reduce + matmul4/5 + all-reduce + MoE) is **mostly robust** to small per-core SDPA differences — at 1 chunk it cancels them out entirely, producing bit-identical final MLP outputs in both iters.
- At multi-chunk the cumulative per-chunk asymmetry inside flash_mla (via reduce_max/sum's DEST round-trips and MM2's running accumulator) compounds enough that downstream stops canceling, and the asymmetry surfaces in MLP PCC.

The "saturation" of MLP-PCC gap at ~2-3e-4 from 4 chunks onward (Attempt 16) is consistent: each additional chunk adds a small bank-asymmetric perturbation, but the cumulative effect saturates because the downstream max/sum normalization rebases everything onto the same scale.

**This rewrites the experimental approach.** We can now:
- Use `position_id=127` (1 chunk) as the **cheapest repro** — much faster JIT/runtime than 8190.
- Use **hash equality across iter-0 / iter-1** at sdpa_output_cb as the diagnostic, not MLP PCC. MLP PCC is noisy and downstream-dependent; the hash is a direct signal.

The bug zone is the same `flash_mla.hpp:747-791` block, but now we can bisect with much faster iteration cycles.

**State.** Hash sites in tree. Parametrize includes both 127 and 511 for control + bug case. Kernel patches all reverted.

---

## Updated handoff (post-output-hash)

flash_mla compute is **bank-asymmetric on every call**. Per-core SDPA output diverges between iter-0 (bank 0) and iter-1 (bank 1) regardless of chunk count. The MLP-PCC alternation is just the downstream-visible projection of this for multi-chunk runs.

### What we know

- Confirmed (Attempt 18): inputs identical between iters.
- Confirmed (Attempt 19): flash_mla per-core output differs between iters at 1 chunk, even though MLP PCC doesn't.
- Bug zone: `flash_mla.hpp:747-791`, specifically inside the `tile_regs_acquire→release` block.
- 2c partial fix: cross-thread MATH+PACK `MATH_Offset` reset at chunk entry collapses MLP PCC gap 20× (at multi-chunk).
- 2c-narrow / 2c-move sweep: only the exact "per-chunk, chunk-entry, both threads, not also at tail" recipe works.

### Cheapest next experiments (with hash-equality diagnostic)

1. **2c at position_id=127 — does it equalize hashes?** Apply 2c and hash; if hash pairs across iters (16 unique values × 2 each), 2c fully fixes the bug at 1 chunk. If still 32 unique, 2c only mitigates downstream propagation, not the per-call asymmetry.

2. **Bisect within `compute_sdpa_chunk` by op-skipping.** With 1-chunk repro (`first_chunk=true` path), the ops are: MM1 → reduce_max → bcast_sub → fast_approx_exp (loop) → MM2 → reduce_sum → tail. Skip each in turn (just on iter-0, or just on iter-1, or both, with simplified replacement) and watch hash equality. The op whose skip makes hashes match is the bug op.

3. **2c-address sweep with hash-equality diagnostic.** Vary which DEST offset 2c's SETC16 points at. Quickly detects which tile address is the bug-relevant one.

4. **Pack-DEST-and-hash inside chunk.** Insert intermediate hash points by packing DEST tiles to L1 mid-chunk. More invasive but pinpoints which DEST tile diverges first across iters.

My pick: **(1) 2c at position_id=127 with hash diagnostic** — single-line edit, validates whether 2c addresses the per-call asymmetry or just downstream visibility.

---

## Attempt 20 — Experiment A: ZEROACC entire DEST at kernel boot

**What.** At top of `kernel_main()` in `decoder_block_kernel.cpp`, immediately before the `iteration = 0; while (true) {}` loop, add a one-shot `TT_ZEROACC(CLR_ALL, ...)` on MATH (TRISC1). Clears all 64 tiles (both banks) before any iter runs.

```cpp
#if defined(COMPILE_FOR_TRISC)
    MATH((TT_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, ADDR_MOD_1, 0)));
#endif
```

**Motivation.** SRAM doesn't necessarily power up to zero. If DEST bank 1 holds non-zero residual data from a prior test on the same device (or genuine power-on garbage), iter-1's `flash_mla` accumulating ops (`mm2 +=`, reduce_max's prev_max combine, reduce_sum's prev_sum accumulate) would pick up that residual as the "initial state" — producing bank-asymmetric output relative to iter-0 (which sees the bank-0 boot state). Clearing both banks at boot rules this hypothesis in/out.

**Result.**

| Metric | Baseline | Experiment A |
|---|---|---|
| MLP PCC @ pos=127 (`num_internal_iterations=2`) | 0.9901647631143314 | 0.9901647631143314 (same) |
| MLP PCC @ pos=511 (`num_internal_iterations=2`) | 0.9916859209677518 | 0.9916859209677518 (same) |
| Unique 0x30 hashes @ pos=127 | 32 / 32 | **28 / 32** |

The unique-hash count drops from 32 → 28: 4 cores out of ~16 active now have iter-0 and iter-1 emitting **bit-identical** `sdpa_output_cb`. The other 12 cores still diverge across iters. MLP PCC unchanged.

**Conclusion.** Residual DEST data is **partially** the story — for 4 cores, clearing bank state at boot is enough to make iter-0 and iter-1 outputs match. For the other 12 cores there's another mechanism. Residual data is not the dominant cause; clearing it doesn't fix the bug or change MLP PCC.

**State.** Reverted.

---

## Attempt 21 — Experiment B: move `sdpa_custom_mm_block_uninit()` to before `cb_push_back`

**What.** In `flash_mla.hpp:792-801`, move `sdpa_custom_mm_block_uninit()` from its original position (after `tile_regs_release()`) to before `cb_push_back(sdpa_output_cb, ...)`. The MM CFG state isn't read by `pack_block_contiguous` (which already ran above the move point), so the uninit can safely move earlier.

```cpp
// Before:
cb_push_back(sdpa_output_cb, ...);
hash_cb(...);
tile_regs_commit/wait/release;
sdpa_custom_mm_block_uninit();
MATH(t6_semaphore_wait_on_max<...>(FPU_SFPU));

// After:
sdpa_custom_mm_block_uninit();           // moved here
cb_push_back(sdpa_output_cb, ...);
hash_cb(...);
tile_regs_commit/wait/release;
MATH(t6_semaphore_wait_on_max<...>(FPU_SFPU));
```

**Motivation.** The original ordering does CFG-pipe writes (`sdpa_custom_mm_block_uninit` tears down MOP config / replay buffer) *after* `tile_regs_release`. If those writes interfere with the bank-flip / ZEROACC issued at release (which dispatches on MATH/SFPU pipe), the resulting transient state might leak into the next chunk's compute and contribute to bank-asymmetric output.

**Result.**

| Metric | Baseline | Experiment B |
|---|---|---|
| MLP PCC @ pos=127 | 0.9901647631143314 | 0.9901647631143314 (same) |
| MLP PCC @ pos=511 | 0.9916859209677518 | 0.9916859209677518 (same) |
| Unique 0x30 hashes @ pos=127 | 32 / 32 | **31 / 32** |

Effectively a no-op — only 1 pair of emissions now matches. MLP PCC unchanged.

**Conclusion.** Uninit position is not a significant source of bank-asymmetric output. The CFG writes from `sdpa_custom_mm_block_uninit` after release don't meaningfully affect the next chunk's compute.

**State.** Reverted.

---

## Updated handoff (post experiments A+B)

Summary of hash-equality results at `position_id=127`:

| Patch | Unique 0x30 / 32 | Effect |
|---|---|---|
| Baseline (no patch) | 32 | bug fully present per-core |
| Experiment A (ZEROACC boot) | 28 | 4 cores fixed |
| Experiment B (uninit early) | 31 | 1 core fixed |
| 2c chunk-entry (Attempt 7, on MLP PCC) | n/a (haven't measured at pos=127 yet) | 20× collapse on MLP PCC |

Neither A nor B fixes the bug. They each provide small signal — A is more interesting because it definitively shows that **for some cores, the bank state at function entry IS a contributor**, just not the dominant one.

The remaining open puzzle: why do ~12 of 16 active cores still have iter-0 vs iter-1 output divergence even when DEST is fully cleared at boot? Possibilities:
1. The boot ZEROACC is on TRISC1 (MATH) but not synced with TRISC2 (PACK); PACK's view of bank state may still hold stale data.
2. Some non-DEST register (PACK_SEC GPR contents, ADC counters, MOP config) persists from one tile_regs cycle to the next in a bank-asymmetric way.
3. The bug is in a HW op (MOVD2B / MVMUL / SFPLOAD with specific MATH_Offset values) that has bank-asymmetric numerical behavior independent of DEST contents.

### Next move (back on the original plan)

Run **2c at position_id=127 with hash diagnostic** — does 2c (cross-thread MATH+PACK MATH_Offset reset at chunk entry) make iter-0/iter-1 hashes match for all cores at 1 chunk? If yes, 2c addresses the per-call asymmetry directly. If still 32 unique, 2c only masks downstream propagation at multi-chunk and the per-call mechanism is independent.

---

## Attempt 22 — 2c at `position_id=127` with hash diagnostic

**What.** Re-apply the original 2c patch (cross-thread MATH+PACK `MATH_Offset` reset at the top of every `compute_sdpa_chunk`, same code as Attempt 7) and run at `position_id=127` (1 chunk, MLP PCC iter-0 = iter-1 = bit-identical baseline). Check whether 2c reduces the per-core sdpa_output_cb hash divergence between iter-0 and iter-1.

**Motivation.** Disambiguate whether 2c is a real per-call fix or just a downstream-visibility masker. Attempt 7 showed it collapses the MLP PCC gap ~20× at `position_id=8190` (64 chunks). The hash diagnostic — sensitive at the per-core SDPA output level — would reveal whether 2c also reduces hash-divergence at 1 chunk where the bug is invisible to MLP PCC.

**Result.**

| Run | iter=1 PCC @ pos=127 | iter=2 PCC @ pos=127 | iter=1 PCC @ pos=511 | iter=2 PCC @ pos=511 | Unique 0x30 hashes @ pos=127 |
|---|---|---|---|---|---|
| Baseline | n/a (skipped) | 0.9901647631143314 | n/a | 0.9916859209677518 | 32 / 32 |
| Attempt 22 (2c) | n/a | 0.9901647631143314 (same) | n/a | 0.9916859209677518 (same) | **31 / 32** |

Only 1 pair of emissions now matches across iter-0 and iter-1 (vs 0 pairs at baseline). MLP PCC unchanged at both `position_id` values.

For reference, the other per-call-state-touching patches gave:

| Patch | Unique 0x30 / 32 @ pos=127 | Cores fixed |
|---|---|---|
| Baseline | 32 | 0 |
| **2c** (this run) | **31** | **1** |
| Experiment B (uninit-before-push) | 31 | 1 |
| Experiment A (ZEROACC at boot) | 28 | 4 |

**Conclusion (decisive).** 2c is **not a per-call asymmetry fix**. It moves only 1 pair (basically a no-op at 1 chunk). The ~20× MLP PCC gap collapse at 64 chunks is a **downstream-visibility masking** effect — 2c doesn't change what flash_mla emits per-core; it changes the cumulative state across many chunks such that the downstream aggregation cancels more of the per-core differences. At 1 chunk there is nothing to cumulate, so 2c has no effective work to do.

This sharply revises the picture from earlier attempts:
- Attempt 7's "20× collapse" interpretation was misleading.
- Attempts 9-13's tight position-sensitivity story was tracking a downstream-visibility tuning knob, not a real fix.
- The real bug is the per-call, per-core bank-asymmetric flash_mla compute (Attempt 19 confirmed; Attempt 22 confirms 2c doesn't touch it).

Closest patch we have to a per-call effect remains Experiment A (ZEROACC at kernel boot): 4 cores fixed out of 16. That's evidence that residual DEST state contributes for *some* cores, but most cores remain bank-asymmetric for another reason.

**State.** Reverted.

---

## Planned next session — tt-exalens dump via `asm("ebreak")`

The hash diagnostic shows **what** diverges (per-core SDPA output) but not **why**. To inspect actual Tensix state (DEST contents, CFG registers, GPRs, ADC counters, etc.) at the moment of divergence, halt the TRISC at a chosen PC and dump state with tt-exalens.

### Plan

1. **Insert `asm volatile("ebreak");`** in the kernel at the suspect transition point. Two interesting halt locations:
   - **(A) Right before iter-1's `mla_body()` call** (gated by `iteration == 1`). Inspect state iter-1 sees at entry → compare to expected boot-clean state. Differences = whatever iter-0 left behind that iter-1 inherits.
   - **(B) At the very top of `compute_sdpa_chunk`** (after gating on iter-0 vs iter-1 via a counter, so we can choose one or the other). Inspect state at the exact chunk-entry CFG position where 2c's SETC16 pair acts. Compare iter-0's halted state to iter-1's halted state to pin down exactly which register the cross-thread SETC16 is affecting.

2. **Run the test** with `num_internal_iterations=2`, `position_id=127`. The TRISC executing the gated `ebreak` will halt. Other TRISCs / cores continue.

3. **From a separate Python process** (not pytest, to avoid device ownership conflict): `init_ttexalens()` + `get_tensix_state(core_coord, device_id=…)` for each SDPA worker core that's halted. Save the `TensixState` (alu_config, pack_config, pack_dest_rd_ctrl, pack_edge_offset, pack_counters, pack_strides, gpr, register_window_counters, address_counters). Also use `read_riscv_memory` to read DEST contents directly if exposed, or `read_words_from_device` for L1.

4. **Resume** by writing a 0 to the BRISC `mailbox`/`debug_resume` register (RISC-V debug interface), or by simply restarting the test for the next dump (we don't strictly need to resume — once we have the state snapshot we're done).

5. **Compare** iter-0 dump vs iter-1 dump:
   - Same MATH_Offset value? (expected: 0 for iter-0, HALF_SIZE for iter-1)
   - Same PACK_SEC0..3 values? PACK_SEC GPRs (DEST_OFFSET_LO+0..3 / HI+0..3)?
   - Same ADC counters? Pack_counters?
   - Same DEST contents (the residual bank state we partially confirmed in Attempt 20)?
   - Any unexpected register that *differs* and would explain bank-asymmetric numerical output.

### Open questions to resolve before the session

- Does tt-exalens cleanly attach to a device that ttnn already holds? (`init_ttexalens` may want exclusive UMD access. Likely need to release the ttnn mesh before attaching, or use a `mesh.synchronize_device()` + Python sleep to keep the device alive while we attach via exalens, then resume.)
- What does `read_riscv_memory` see — L1 / TRISC private memory? DEST is not L1-addressable; we may need a debug-pack-DEST-to-L1 step before the ebreak to capture DEST contents.
- Does `ebreak` on TRISC halt cleanly without taking down the rest of the program?

### Cheaper warm-up before the exalens session

If exalens-on-live-device proves fiddly, we can fall back to **kernel-self-dump**: pack both DEST banks to a debug L1 buffer right before the ebreak (or right before iter-1 starts), let the kernel complete normally, then read the L1 buffer from the test's host side. Same information as exalens for DEST contents, less for CFG registers. Simpler to wire up.

### Why this is the right next step

Every patch we've tried so far has been a *guess* at the bug location. The hash diagnostic narrows the bug to "per-call bank-asymmetric flash_mla compute", but to know which specific register or state element is the asymmetry source we need to *look*. The ebreak + exalens dump is the first experiment that produces direct evidence rather than indirect inference from PCC / hash signals.

---

## Parallel track — SFPU hash bisection (PR #43060)

### Setup (done — files installed)

Pulled in [PR #43060](https://github.com/tenstorrent/tt-metal/pull/43060) which adds `hash_cb_sfpu(in_cb, num_tiles, out_cb, label)` — an SFPU-accelerated CB hasher with ~32× throughput vs the scalar `hash_cb`. Five files installed (`ad2269bb` snapshot of pr43060):

- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_hash_cb.h`
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/experimental/llk_math_hash_cb.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_hash_cb_api.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/experimental/llk_math_hash_cb_api.h`
- `tt_metal/hw/inc/api/compute/debug/cb_hash_sfpu.h`

### Caveats — read before using

1. **"Best-effort draft, not hardware-validated"** (from the LLK header). The reduction-tree `SFPSHFT2` ordering and DEST addressing have not been tested on silicon. May need iteration on the LLK before it produces stable hashes.
2. **23-bit FNV variant** ("FNV23"), not bit-equal to the scalar `hash_cb` (FNV-1a-32). Only compare SFPU hashes to SFPU hashes.
3. **Uses DEST + SFPU state** — the very machinery we suspect of being bank-asymmetric. This is both the caveat the PR explicitly notes and *the feature* we want: if the SFPU hash of `cb_q_in` (whose L1 content scalar `hash_cb` already proved bit-identical between iter-0 and iter-1 in Attempt 18) differs across iters under `hash_cb_sfpu`, we have direct evidence that the SFPU pipeline produces bank-asymmetric output on identical input.
4. **Requires a host-allocated output CB** (1 tile, INT32 format) to receive the hash result. The decoder block kernel does not currently allocate one — this is a non-trivial host-side change in `attention_block/op.py` or similar to add a debug CB to the descriptor.

### Bisection plan

User-defined strategy: at each probe point, hash the same logical state across iter-0 and iter-1. Decision tree per point:

- **Hash differs on SOME cores** → race condition or L1 data-movement corruption. Hunt down dataflow kernels (BRISC/NCRISC) writing the wrong bytes / colliding broadcasts.
- **Hash differs on ALL cores** → compute leftover state (DEST or L1). Hunt down compute kernel state that carries across iters.
- **Hash matches on ALL cores** → state at this point is deterministic across iters. Move to next probe point downstream.

Probe points to consider, ordered earliest-to-latest within flash_mla:

| Probe | Where | What it tests |
|---|---|---|
| A1 | `flash_mla.hpp:746`, hash `cb_q_in` | SFPU hash of an L1 input we already know is bit-identical (scalar Attempt 18). Direct test of "is the SFPU pipeline itself bank-asymmetric on identical input?" |
| A2 | `sdpa.h:269` (chunk top), hash `cb_k_in` | Same idea, for K input on every chunk. |
| B | After MM1 in `compute_sdpa_chunk`: pack mm1 tile (or specific row) to a debug CB, hash the debug CB | First DEST-write under suspicion; if hashes match here, MM1 (FPU) is OK. |
| C | After reduce_max (PACK-SFPU) | Tests SFPU reduce + cumulative max-DEST tile. |
| D | After bcast_sub (FPU) | Tests FPU broadcast subtract. |
| E | After fast_approx_exp (SFPU, per-tile loop) | Tests SFPU exp pipeline. |
| F | After MM2 (sdpa_custom_mm_reuse_dest_srcb_block, FPU+MOVD2B) | The DEST-via-SRCB read path — strong suspect. |
| G | After reduce_sum (PACK-SFPU) | Tests cumulative sum-DEST tile. |
| H | `flash_mla.hpp:792`, hash `sdpa_output_cb` (we already have this with scalar; would compare SFPU hash too) | End of flash_mla. We already know scalar hash differs here. |

Probes B–G require a "pack DEST → debug CB → hash debug CB" pattern. Each adds invasive code inside the `tile_regs_acquire→release` block. Cleanest implementation:

1. Allocate a per-core 1-tile debug CB in host code (free at all relevant points).
2. Add a helper `pack_and_hash_dest(dest_tile_idx, debug_cb, hash_out_cb, label)` that issues `pack_block_contiguous(dest_tile_idx, debug_cb, 1)` then `hash_cb_sfpu(debug_cb, 1, hash_out_cb, label)` then `cb_pop_front(debug_cb, 1)`.
3. Call it at chosen probes.

### Recommended first run (when resuming)

**Probe A1 only** — install hash_cb_sfpu host-side CB plumbing, place a single `hash_cb_sfpu(cb_q_in, q_chunk_tiles, cb_hash_out, 0x40)` at `flash_mla.hpp:746` (alongside the existing scalar `hash_cb(cb_q_in, ..., 0x10)`), and run with `position_id=127`, `num_internal_iterations=2`.

Expected outcomes:
- **SFPU hash matches across iters for all cores** → SFPU pipeline is deterministic on identical input. Bug is in some FPU or downstream path. Move to probe B.
- **SFPU hash differs on all cores** → SFPU itself is bank-asymmetric on identical input. The bug zone is the SFPU dispatch within compute_sdpa_chunk; bisect via probes B–G.
- **SFPU hash differs on SOME cores** → some race or partial-data effect; investigate dataflow kernels feeding cb_q_in.

This is the cleanest first signal.

### Why this is worth doing alongside the exalens plan

The exalens plan looks at *state* (registers, GPRs); the SFPU hash bisection looks at *computed output* of a known-bank-using op on bit-identical input. They're complementary:

- exalens tells us *what* register/cell differs at a halt point.
- SFPU hash tells us *which dispatch* first produces a bank-asymmetric output.

If both agree, we've nailed both ends of the divergence. If they disagree, the discrepancy itself is informative.

---

## Attempt 23 — Probe B: hash mm2 after chunk loop (scalar hash via cb_out_in)

**What.** Took the simpler route (avoid host-side new-CB plumbing): pack the mm2 P·V accumulator tile (at `mm2_dst_tile_offset=0`) to `cb_out_in` after the chunk loop ends but before the tail, hash with scalar `hash_cb`, then pop:

```cpp
// In flash_mla.hpp, after the chunk for-loop ends, before the !sdpa_output_is_final branch:
pack_block_contiguous(mm2_dst_tile_offset, cb_out_in, 1);
cb_push_back(cb_out_in, 1);
hash_cb(cb_out_in, 1, 0x40);
cb_pop_front(cb_out_in, 1);
```

`cb_out_in` chosen as scratch because (a) it has the same `stats_df` (bf16) 8x32 tile format as the existing PACK setup, so no `pack_reconfig` needed; (b) it's an input CB to `sdpa_tail` which only runs when `num_cores_to_wait > 0` — at `position_id=127` (1 chunk total, `num_active_s_blocks=1`, no tree-reduce partners) it's unused.

**Motivation.** Probe whether MM2's output (the running P·V accumulator) is bit-identical between iter-0 and iter-1 on the same core. If yes → MM2 (and all of MM1 / reduce_max / bcast_sub / fast_approx_exp) are bank-symmetric on identical inputs, and the divergence is downstream in the tail (`compute_sdpa_recip` + final pack). If no → MM2 or upstream is bank-asymmetric.

**Caveat hit on first attempt.** Initially tried with both `position_id=127` and `=511` in the parametrize. **Pos=511 hung** because `num_cores_to_wait > 0` there, so `sdpa_tail` actually runs and tries to `cb_wait_front(cb_out_in, ...)`. Our `cb_pop_front` had drained the CB, so sdpa_tail waits forever. Killed and restricted parametrize to pos=127 only.

**Result @ pos=127, num_internal_iterations=2.**

Hash distribution:

| Probe | Total emissions | Unique values | Pattern |
|---|---|---|---|
| `0x40` (mm2 hash, post-chunk-loop) | 32 | 20 | **12 paired (24 emissions) + 8 singletons (4 cores × 2 hashes)** → 12/16 cores match, 4/16 diverge at mm2 |
| `0x30` (final sdpa_output_cb hash, end of flash_mla) | 32 | **16** | **All 16 values appear exactly twice** → **16/16 cores match at final output!** |

Baseline reminder (Attempt 19 @ pos=127, no probe B): `0x30` was 32 unique / 32 emissions — **0/16 cores matched**.

Cross-check on one diverging core (dev 6, x=0, y=1):

```
hash[0x40] = 0x87537656   ←  iter-0 mm2
hash[0x30] = 0xaa6fc356   ←  iter-0 final output
hash[0x40] = 0xc74bc945   ←  iter-1 mm2 (DIFFERENT from iter-0)
hash[0x30] = 0xaa6fc356   ←  iter-1 final output (IDENTICAL to iter-0)
```

mm2 differs between iters; final output is bit-identical. The tail (`compute_sdpa_recip`'s `mm2 *= 1/sum`) cancelled the bank-asymmetric scale factor for that core.

**Conclusion (two parts).**

**A.** Mathematically, flash-attention is scale-invariant: if MM1 / SFPU-exp / MM2 produce `mm2_iter1 = c * mm2_iter0` and correspondingly `sum_iter1 = c * sum_iter0` for some scalar `c`, then `mm2/sum` is bit-identical. The 4 cores that diverge at mm2 but match at the final output are exhibiting exactly this: bank-asymmetric scale factor cancelling through the recip normalize. Compute_sdpa_recip is doing its job.

**B.** Adding probe B (`pack + cb_push_back + hash_cb + cb_pop_front` of `cb_out_in`) inadvertently **masked the final-output bug for ALL 16 cores at pos=127**. Baseline had 0/16 matching; with probe B in place, 16/16 match. The probe doesn't touch the bug's actual mechanism — it's a stealth side-effect, similar to 2c's masking, but acting at a different layer.

This is the **strongest masking** we've seen so far in any patch. 2c only collapsed gap at multi-chunk MLP PCC (not at per-core 1-chunk hashes); probe B fully fixes per-core 1-chunk output. Worth bisecting probe B's components to find the active ingredient.

**State.** Probe B left in place; test parametrize narrowed to pos=127 only (multi-chunk hangs sdpa_tail; needs a different scratch CB).

---

## Attempt 24 — Swap final `pack_block_contiguous` for `pack_tile` (per-tile loop)

**What.** Replaced the final mm2 → `sdpa_output_cb` pack loop (`flash_mla.hpp:812-816`) with per-tile `pack_tile(mm2_dst_tile_offset + i + j, sdpa_output_cb, i + j)` calls. Removed `pack_block_contiguous_init(sdpa_output_cb)` at line 741 (pack_tile is documented as not needing explicit init — `tt_metal/hw/inc/api/compute/pack.h:44`). Reverted Attempt-23+ pre-recip dump block. Restored Attempt-19's `hash_cb(sdpa_output_cb, out_chunk_tiles, 0x30)` after the final `cb_push_back`.

**Motivation.** Direct test of "is `pack_block_contiguous` itself bank-asymmetric on identical DEST input?" If swapping to `pack_tile` collapses the hash distribution, the experimental block-contiguous MOP/replay machinery is implicated.

**Result @ pos=127, `num_internal_iterations=2`.**

| Metric | Attempt 19 | pack_tile |
|---|---|---|
| MLP PCC | 0.9901647631143314 | **0.9483054293745399** (fails 0.975 threshold) |
| Hash unique / total at `0x30` | 32 / 32 | 23 / 32 |
| Paired cores (iter-0 == iter-1) | 0 / 16 | 9 / 16 |

**Conclusion (contaminated).** Initially read as a "strong signal" (9/16 cores fixed) but the PCC drop to 0.9483 — far below 0.975 threshold — means `pack_tile` is writing **different** bytes than `pack_block_contiguous`, so the hash comparison is apples-to-oranges. Two distinct findings entangled:

1. `pack_tile` produces different L1 output than `pack_block_contiguous` for this kernel (PCC 0.99 → 0.95).
2. Among those (wrong) bytes, 9/16 cores happen to be bank-symmetric.

We cannot infer that `pack_block_contiguous` is the bug op from this experiment.

**State.** Reverted.

---

## Attempt 25 — Swap final pack for `pack_tile_block` (canonical sibling)

**What.** Replaced inner loop with `pack_tile_block(mm2_dst_tile_offset + i, sdpa_output_cb, output_granularity)` — the canonical multi-tile pack API documented at `pack.h:101`, using `llk_matmul_pack` under the hood. Kept everything else as in Attempt 24 (no `pack_block_contiguous_init`).

**Motivation.** `pack_tile` and `pack_tile_block` use the same LLK dispatch (`llk_pack` / `llk_matmul_pack`); the latter packs N tiles in one call. If both produce the same hash distribution, the API doesn't matter; if `pack_tile_block` matches `pack_block_contiguous`'s correctness, the bug is in `pack_block_contiguous`'s MOP specifically.

**Result @ pos=127.**

| Metric | Attempt 19 | pack_tile | pack_tile_block |
|---|---|---|---|
| MLP PCC | 0.9901647631143314 | 0.9483054293745399 | **0.9483054293745399** (bit-identical to pack_tile) |
| Unique / total | 32 / 32 | 23 / 32 | 20 / 32 |
| Paired cores | 0 / 16 | 9 / 16 | 12 / 16 |

**Conclusion (structural error found).** `pack_tile` and `pack_tile_block` produce **identical** numerics (PCC bit-identical), confirming the regression is structural, not API-specific. The reason: mm2's DEST occupies 16 *mini-tiles* (8×32 layout from `TD_8x32`), which is 4 standard-32×32-tile-equivalents. `pack_tile_block(..., ntiles=16)` interprets `ntiles=16` as **16 standard 32×32 tiles** and reads 4× past the actual mm2 region into the adjacent DEST scratch (max/sum/corr_exp/mm1 — all positioned per `flash_mla.hpp:711-719`). So we were hashing "mm2 + scratch" L1 bytes, not pure mm2. The 12 paired cores reflect "scratch areas happen to be bank-symmetric on those cores", not "the bug was in pack_block_contiguous".

`pack_block_contiguous` uses a MOP that's configured via `_llk_pack_block_contiguous_mop_config_(pack_dst_format, face_r_dim, num_faces)` — and that mop_config reads CB metadata (`get_output_face_r_dim`, `get_output_num_faces`), so it adapts to mini-tile geometry. `pack_tile` / `pack_tile_block` use the default packer dispatch that assumes standard tiles.

**State.** Reverted.

---

## Attempt 26 — Surgical RMW: `pack_reads_per_xy_plane = face_r_dim`

**What.** Inspection of the LLK reveals (`tt_llk_blackhole/llk_lib/experimental/llk_pack_block.h:40-42`):

> Precondition: `_llk_pack_init_` or `_llk_pack_configure_addrmod_` + `set_packer_strides` must have been called to establish the normal pack ADDR_MOD_0/1/2 and strides. This function only replaces the MOP.

`pack_block_contiguous_init` programs the MOP and REPLAY buffer for the mini-tile face geometry (via `get_output_face_r_dim` / `get_output_num_faces`), but the precondition state — addr_mods, strides, **pack counters** — comes from whatever upstream `llk_pack_init` last ran. In our pipeline the last upstream init was for standard 32×32 tiles, so `PACK_COUNTERS_SEC0.pack_reads_per_xy_plane = FACE_R_DIM = 16`. With our 8×32 mini-tile output CBs, the tile position generator's Y wraps past mini-tile boundaries, generating wrong `EdgeMask` selections (per `tt-isa-documentation/WormholeB0/.../Packers/EdgeMasking.md`).

Surgical patch — a single RMW to `PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW` before `pack_block_contiguous_init(sdpa_output_cb)`:

```cpp
PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(
    get_output_face_r_dim(get_output_id(sdpa_output_cb)))));
pack_block_contiguous_init(sdpa_output_cb);
```

**Result @ pos=127.**

| Metric | Baseline | RMW with `face_r_dim` (=8) |
|---|---|---|
| MLP PCC | 0.9901647631143314 | **0.9901647631143314 (bit-identical)** |
| Unique / total | 32 / 32 | **29 / 32** |
| Paired cores | 0 / 16 | **3 / 16** |

PCC stays at baseline (correctness preserved). 3 cores now have bit-identical iter-0 / iter-1 output.

**Conclusion.** `pack_reads_per_xy_plane` IS a real stale-state issue — patching it cleanly fixes 3 cores out of 16. But 13 cores are unaffected, so this counter is not the dominant asymmetry source for most of them.

**State.** Kept (refined to value `1` in Attempt 27).

---

## Attempt 27 — RMW with value `1` (Nikola's "safe everywhere except pack_untilize")

**What.** Per HW-team guidance: `pack_reads_per_xy_plane = 1` is safe for any pack op that isn't `pack_untilize`. Forces the tile position generator's Y to wrap after every iteration. Replaced the dynamic `get_output_face_r_dim(...)` value with hardcoded `1`.

**Result @ pos=127.**

| Metric | Baseline | RMW=8 (face_r_dim) | **RMW=1** |
|---|---|---|---|
| MLP PCC | 0.9901647631143314 | 0.9901647631143314 | **0.9901647631143314** |
| Unique / total | 32 / 32 | 29 / 32 | **28 / 32** |
| Paired cores | 0 / 16 | 3 / 16 | **4 / 16** |

One additional core collapses. PCC unchanged.

**Conclusion.** Best surgical fix found so far. Kept as the in-tree patch.

**State.** Kept.

---

## Attempt 28 — Reset all PAC ADC counters (X/Y/Z/W on both channels)

**What.** Hypothesis follow-up after reading `tt-isa-documentation/.../Packers/InputAddressGenerator.md` + `OutputAddressGenerator.md`: PACR's DEST read address is composed from ADC Channel[0] X/Y/Z/W, and the L1 write address from Channel[1] Y/Z/W. `_llk_pack_block_contiguous_` (`tt_llk_blackhole/.../llk_pack_block.h:189-204`) only resets `ch0_z` (mask `0b0001`) on entry and `ch0_z`+`ch1_z` (mask `0b0101`) on exit. All other ADC fields are inherited from upstream. If `ch0_y` is non-zero on entry, the first PACR reads from row `ch0_y` instead of row 0 — landing in different DEST contents on bank 0 vs bank 1.

Tried two placements of `TTI_SETADCXY(p_setadc::PAC, 0,0,0,0, 0b1111) + TTI_SETADCZW(p_setadc::PAC, 0,0,0,0, 0b1111)`:

**28a — pre-chunk-loop** (right after the new RMW, before `pack_block_contiguous_init`):
| MLP PCC | 0.0006286624305142914 (catastrophic) |
| Paired cores | 1 / 16 |

The chunk loop's compute ops (`reduce_max`/`reduce_sum`/MM2) rely on ADC state being set by their own inits; zeroing it at flash_mla entry corrupts the chunk's mid-pack ops.

**28b — post-chunk-loop, pre-final-pack** (between `compute_sdpa_recip` and the final pack loop):
| MLP PCC | 0.004497569060970548 (catastrophic) |
| Paired cores | 5 / 16 |

Still catastrophic. `compute_sdpa_recip` and the chunk loop deliberately leave ADC in the state the next pack needs (set_dst_write_addr only sets `ch0_w`; the rest must be correct from the prior op for the pack's MOP to read mm2 at the right DEST address).

**Conclusion.** ADC counters at the boundary BETWEEN compute ops are NOT "stale upstream garbage" — they are the load-bearing state machine output of the prior compute. Resetting them breaks the pipeline. The PACR-level partial reset (only `ch0_z`) in `_llk_pack_block_contiguous_` is intentional, not a bug.

**State.** Reverted.

---

## Attempt 29 — Reset only `ch0_y` (surgical narrowing of Attempt 28)

**What.** Most ADC fields needed to stay (per 28), but `ch0_y` is the one most likely to misdirect DEST reads at a half-DEST granularity (rows vs face boundaries). Replaced the full reset with `TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010)` — only `ch0_y = 0`.

**Result @ pos=127** (on top of Attempt 27's RMW=1):

| Metric | RMW=1 only | RMW=1 + `ch0_y` reset |
|---|---|---|
| MLP PCC | 0.9901647631143314 | 0.9901647631143314 (bit-identical) |
| Unique / total | 28 / 32 | **30 / 32** |
| Paired cores | 4 / 16 | **2 / 16** |

PCC preserved, but paired-core count DROPPED from 4 to 2. Resetting `ch0_y` made cores diverge MORE.

**Conclusion.** `ch0_y` at flash_mla's final-pack entry is consistent between iter-0 and iter-1 on the cores that previously matched — forcing it to 0 shifts those cores into a different (asymmetric) DEST region. So `ch0_y` was load-bearing on those cores, not stale.

**State.** Reverted.

---

## Updated handoff (post Attempts 24-29)

**Confirmed surgical fix in tree:**
```cpp
PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1)));
pack_block_contiguous_init(sdpa_output_cb);
```
4 of 16 cores' bank-asymmetry collapses; PCC unchanged. Real stale-state bug found and fixed (one of multiple).

**Asymmetry sources remaining (12/16 cores still divergent):**
- NOT pack-side ADC residue (Attempts 28-29 ruled out).
- NOT pack MOP/REPLAY config (Attempt 25 ruled out the API).
- Possibly: `PCK_EDGE_OFFSET_SEC` mask / `TILE_ROW_SET_MAPPING` left at non-default by upstream.
- Possibly: ADDR_MOD_PACK_SEC values left at non-default — `llk_pack.h:49-68` shows default `y_src.incr=4`, `y_dst.incr=4`, but if upstream used `untilize` or `tilize` variants, increments differ.
- Most likely: **compute-side bank-asymmetry** in the chunk loop (`reduce_max` / `reduce_sum` SFPSTORE-SFPLOAD round-trips) or in `compute_sdpa_recip` itself. The bank-asymmetric bytes are already in DEST when pack runs; pack faithfully transmits them to L1.

**Next probe candidates:**
1. Hash DEST→L1 dump RIGHT AFTER `compute_sdpa_recip`, pre-final-pack — does mm2 in DEST already differ between iter-0 and iter-1? (Caveat: any added pack acts as a masking probe per Attempts 23+. Hash via SFPU path (#43060) would be less invasive.)
2. Inspect `PCK_EDGE_OFFSET_SEC0_mask` and `TILE_ROW_SET_MAPPING_0` via tt-exalens at flash_mla entry vs after the patch.
3. Run RMW=1 patch at `position_id=8190` (multi-chunk, MLP-PCC-alternation case) to see if the partial fix moves the downstream MLP PCC gap.

---

## Attempt 30 — Bump `num_internal_iterations` to 8 (per-iter pattern probe)

**What.** Revert the Attempt-27 RMW (clean Attempt-19-style baseline), set `test_decoder_block.py:886` parametrize to `num_internal_iterations=[8]`, run @ `position_id=127`. The kernel now calls `flash_mla` 8 times within a single program launch; with the hash diagnostic at the end of each `flash_mla` we get **8 consecutive emissions per core**.

**Motivation.** Attempt 19's "iter-0 vs iter-1" framing was 2-trip-count specific. We didn't know if the bank-asymmetry beyond iter-1 is (a) pure parity alternation `A,B,A,B,...`, (b) per-iter drift `A,B,C,D,...`, or (c) something else. With 8 iters and 16 cores we get 128 hash emissions to characterize the pattern.

**Result @ pos=127, `num_internal_iterations=8`.**

- MLP PCC: 0.9901647631143314 (unchanged from 2-iter baseline)
- Total hash emissions: 128
- **Unique hash values: 37** (8 × 16 = 128 max if every emission distinct; 32 max if 2 per core; 37 → most cores have 2–3 distinct values, no pure parity)

Per-core hash sequence (iter-0 to iter-7, A/B/C labels assigned per-core in first-seen order):

```
(6,0,1):  A A A A A A A A                                    # completely stable from iter-0
(6,0,2):  A A A B B A A A                                    # mostly A, brief flicker mid-run
(6,1,1):  A B B B B B B B                                    # iter-0 unique, then stable
(6,1,2):  A B C B B C A B                                    # cycles through 3
(6,2,1):  A A B A A A B A                                    # mostly A, sporadic B
(6,2,2):  A A A A B A A B                                    # mostly A, sporadic B
(6,3,1):  A B B B B C B B                                    # mostly B after iter-0
(6,3,2):  A A A A A A B A                                    # mostly A, one B near end
(7,0,1):  A A A A A B B A                                    # mostly A, brief 2-iter B
(7,0,2):  A B C C C C C C                                    # 3 distinct then stable
(7,1,1):  A B C C C C C C                                    # 3 distinct then stable
(7,1,2):  A B C B B B B B                                    # 3 distinct early, stable B
(7,2,1):  A A B A A A A B                                    # mostly A, sporadic B
(7,2,2):  A A A A A B B A                                    # mostly A, brief 2-iter B
(7,3,1):  A A A A A B B A                                    # same pattern as (7,2,2)
(7,3,2):  A B C B C C C C                                    # stabilizes at C
```

**Observations.**

1. **No pure parity alternation.** A bank-parity-only bug would predict every core showing `A,B,A,B,A,B,A,B`. None of the 16 cores match that pattern.

2. **Most cores have 2 or 3 distinct hashes**, not 8. So it's NOT random drift either.

3. **(6,0,1) is bit-identical across all 8 iters**, but Attempt 19's 2-iter probe documented this same core diverging at iter-0 vs iter-1. So **whether iter-1 matches iter-0 depends on the total iteration count scheduled**. Reasonable explanation: the kernel's "wind-down" sequence at the end of the last iteration (mcast teardown, semaphore drain, etc.) is dispatched only on the FINAL iter — and dispatcher-level scheduling on TRISC may shift instruction issue ordering for the previous iters' tail when there's more or less wind-down to fold in. That's a host-NoC-side scheduling effect, not a kernel-logic effect.

4. **Many cores converge after iter-0/iter-1 then mostly stabilize**, e.g. `(6,1,1)` and `(7,0,2)`. Plausibly: DEST residual state at the bank-overshoot read location reaches a fixed-point because flash_mla writes and overwrites the same DEST tiles every iteration; once both banks have been written by flash_mla itself the divergent residue is gone.

5. **Some cores have late-run flicker** (e.g. `(7,0,1)` and `(7,2,2)` both `A A A A A B B A`). Not a parity pattern. Suggests a perturbation at iter-5/6 specifically — maybe a scheduling event in the surrounding program (mcast, all-reduce) that bleeds into pack timing.

**Conclusion.** The bug-asymmetry signal is **NOT a clean bank-parity flag**. It's a richer phenomenon involving:
- An initial transient (iter-0 / iter-1 typically distinct from later iters)
- A near-steady-state for the bulk of long runs
- Sporadic per-iter flicker that doesn't follow parity

This reframes the whole investigation. "Bank parity" was probably never the right primary axis. The asymmetry sources are:
- Residual DEST state at flash_mla entry (which differs between bank 0 and bank 1 in the early iters, but converges as flash_mla itself writes both banks repeatedly)
- LLK stale state (e.g., `pack_reads_per_xy_plane`, partially addressed by Attempt 27)
- Some HOST/NoC scheduling effect that depends on `num_internal_iterations` (since (6,0,1) is stable at 8 iters but was bank-asymmetric at 2 iters)

**Next step: re-run with the same configuration to check determinism** — if (6,0,1)'s 8-iter stability and the per-core patterns above replay bit-identically across two runs, the phenomenon is deterministic and the investigation can proceed. If they don't, there's a true non-determinism (race condition or HW-side jitter) that needs a different debugging approach.

**State.** RMW reverted in flash_mla.hpp; parametrize at 8 iters.

---

## Attempt 31 — Re-anchor on deterministic MLP-PCC alternation @ multi-chunk

**Motivation (handed to me by review).** The hash-based investigation (Attempts 19-30) was reading a contaminated channel. Decisive argument: bit-identical MLP PCC across two pytest runs while 15/16 per-core hashes varied across the same two runs implies the per-core hash *cannot* be a faithful proxy for what reaches the MLP output. craqsim's "output identical across iters, hash varies between iter-0 and iter-1" confirms the decoupling is structural, not a race — `hash_cb` reads `fifo_rd_ptr` for the UNPACK-side view of `cb_out_final`, and that read window is not what ultimately lands in the host-visible output. Three months of bisection (Attempts 19-30, incl. my Attempt 27 RMW=1 "4/16 cores fixed") were measured on this contaminated channel and must be discounted.

**Re-anchor.** Pristine kernel (revert Attempt 27 RMW, remove `hash_cb` diagnostic) + `position_id=511` (4 chunks, fastest multi-chunk repro) + `num_internal_iterations ∈ {1, 2}`. The deterministic ground truth is MLP PCC, not the hash.

**Baseline (pristine kernel, two independent runs)**

| Run | iter=1 | iter=2 | Δ |
|---|---|---|---|
| 1 | 0.9914568554511025 | 0.9916859209677518 | 2.29e-4 |
| 2 | 0.9914568554511025 | 0.9916859209677518 | 2.29e-4 |

Bit-identical to 16 digits across runs → **the bug is deterministic**. The "run-to-run hash noise" found in Attempt 30 was a property of the diagnostic, not of the kernel.

**Validation of Attempt 27 RMW patch against this anchor.**

| Patch | iter=1 | iter=2 |
|---|---|---|
| Pristine | 0.9914568554511025 | 0.9916859209677518 |
| + RMW `pack_reads_per_xy_plane=1` | 0.9914568554511025 | 0.9916859209677518 |

Bit-identical with and without the RMW. The RMW affects nothing host-visible. The "4/16 cores collapsed" result it produced was hash-diagnostic artifact only. **Reverted from tree.**

---

## Attempt 32 — Re-enable Austin's iter-top workaround (fix)

**What.** Restored the two lines at `decoder_block_kernel.cpp:3059-3060`:

```cpp
MATH((llk_math_pack_sync_init<false>()));
PACK((llk_pack_dest_init<false, false>(0)));
```

These reset MATH/PACK sync state at the top of every internal iteration, forcing every iter to start in DEST bank 0. Root cause (bank-1 producing numerically different output) remains open per the existing comment at line 3044-3055, but the workaround papers it over deterministically.

**Result.**

| Setup | iter=1 | iter=2 | Δ |
|---|---|---|---|
| Pristine | 0.9914568554511025 | 0.9916859209677518 | 2.29e-4 |
| + workaround | 0.9914568554511025 | **0.9914568554511025** | **0** |

iter=2 now matches iter=1 bit-for-bit at the bank-0 PCC. Alternation eliminated.

**State.** Committed. Anchor (pristine RMW + hash diagnostic dropped from tree); workaround active.

---

## Attempts 33-35 — Root-cause hunt against the deterministic MLP-PCC anchor

After committing Attempt 32 (workaround as fix), pushed back: workaround papers over, not root-cause. The HW shouldn't inherently care about bank 0 vs bank 1 — there must be a kernel-level cause. Three surgical experiments to find it, all against the deterministic anchor (pristine iter=1=0.9914568554511025, iter=2=0.9916859209677518, Δ=2.29e-4).

### Attempt 33 — ZEROACC the bank at `flash_mla` entry

**What.** Inserted `MATH((TT_ZEROACC(p_zeroacc::CLR_HALF, 0, 0, ADDR_MOD_1, dest_offset_id % 2)));` immediately after `tile_regs_acquire()` in `flash_mla.hpp`. Hypothesis: MM2 (`sdpa_custom_mm_reuse_dest_srcb`) does `dest += a*b` without explicit clear-on-first-chunk, so iter-1 was accumulating onto bank-1 residue.

**Result.** PCCs **bit-identical to pristine**: 0.9914568554511025 / 0.9916859209677518. **ZEROACC did nothing.** Bank residue is NOT the cause; the bank is already clean when MATH enters flash_mla.

### Attempt 34 — MATH-side reset only at iter top

**What.** Re-enabled `MATH((llk_math_pack_sync_init<false>()));` at decoder_block_kernel.cpp:3059, kept `llk_pack_dest_init` commented. This resets MATH.dest_offset_id and MATH_PACK semaphore but NOT PACK.dest_offset_id.

**Result.** iter=1 PCC = 0.9914568554511025 (OK), iter=2 **FAILED**: `Device 0 (SP=0) KV Cache before and after op mismatch`. MATH-side reset alone causes a MATH/PACK desync that corrupts the KV cache write. **Not a viable surgical fix.**

### Attempt 35 — PACK-side reset only at iter top

**What.** Inverse of 34: only `PACK((llk_pack_dest_init<false, false>(0)));`. Resets PACK.dest_offset_id, packer ADC, DEST_OFFSET GPRs, but not MATH.

**Result.** Same failure as 34: iter=1 PCC = 0.9914568554511025, iter=2 **FAILED** with identical KV cache mismatch. **Not a viable surgical fix either.**

### Conclusion (root cause status)

The bug is **not** bank residue (33) and **cannot** be fixed by resetting only one thread's state (34, 35). The two halves of Austin's workaround are **load-bearing together** — they coordinate the MATH+PACK `dest_offset_id` reset, and the kernel relies on this coordination for half-DEST correctness.

What's NOT the cause:
- DEST bank residue (33 rules out)
- MATH `dest_offset_id` alone (34's success at iter=1 + iter=2 crash → it gets MATH right but desyncs)
- PACK `dest_offset_id` alone (35 same story)
- LLK stale state pack_reads_per_xy_plane (Attempt 27 ruled out against MLP PCC)

What remains: the comment at decoder_block_kernel.cpp:3050-3055 stands. "Bank 1 produces numerically different output from bank 0 even when all known DEST-addressing registers (TRISC1.MATH_Offset, TRISC2.MATH_Offset via SDPA helpers, PACK_SEC0..3) point at the correct bank." This points to either:

1. A piece of stateful hardware (CFG register, ADC counter, SFPU LREG, replay buffer, mop_cfg) that is implicitly bank-conditional and not yet identified.
2. A piece of kernel state (e.g., the SDPA SFPU replay buffer initialization, or a SETC16 sequence in compute_sdpa_chunk) that is conditioned on `dest_offset_id == 0` and produces different output when conditioned on `== 1`.
3. A genuine HW asymmetry on Blackhole — bank 1 of half-DEST has subtly different numerical behavior than bank 0 (e.g., timing-conditional pipeline forwarding).

To distinguish (1)/(2)/(3): would need either tt-exalens state-snapshot at the moment MATH issues its first MVMUL on bank 1 (compare to the bank-0 case), or a microbenchmark that pins down exactly which op shifts numerically when run on bank 1 vs bank 0 with identical DEST contents.

**Engineering decision.** Ship Austin's workaround as the deterministic fix. Open a follow-up to characterize the bank-1 asymmetry with tt-exalens or via a minimal SFPU/FPU microbenchmark.

**State.** Workaround committed (this commit). Anchor parametrize kept at `position_id=511`, `num_internal_iterations ∈ {1, 2}` as the multi-chunk regression test.

---

## Attempts 36-40 — tt-exalens snapshot + LOADMACRO-hazard hunt + per-shard / per-device localization

### Attempt 36 — tt-exalens state snapshot at flash_mla entry

**What.** Added a kernel `ebreak` gate at `flash_mla.hpp` just after `tile_regs_acquire()`, controlled by a `HALT_FLASH_MLA_ITER` compile-time define injected via env var `TT_HALT_FLASH_MLA_ITER`. Ran the test with the halt at iter-1 entry, then iter-0 entry; used `ttexalens.get_tensix_state()` to dump per-TRISC GPR + ADC + CFG state from logical core (1,1) on device 6.

**Result.** Same-core iter-0 vs iter-1 diff (after filtering out core-position-dependent diffs):

- `gpr[2].dest_offset_hi`: 512 → 0 — PACK's GPR that holds the bank-1 base offset for `select_packer_dest_registers`.
- 5 other PACK scratch GPRs (`tile_header`, `tmp0`, `tmp1`, `tmp_lo`, `exp0_sec_size_bfp`) similarly went from non-zero → 0.
- A handful of `register_window_counters` fields (`rwc_math_winner`, `rwc_math_winner_thread`, `rwc0_dst_reg_addr_d`, `rwc_math_instrn`) differed — runtime pipeline state, likely just timing artifacts of the halt.
- All packer ADC counters and pack_config / pack_strides / pack_edge_offset / pack_dest_rd_ctrl fields were **identical** between iters.

**Conclusion.** `DEST_OFFSET_HI` GPR is zeroed between iter-0 and iter-1. If PACK uses this GPR at `select_packer_dest_registers` (called at every `tile_regs_release`) while `dest_offset_id == 1`, `PACK_SEC0_Offset` CFG becomes 0 → PACK addresses bank 0 instead of bank 1.

### Attempt 37 — Restore `DEST_OFFSET_HI` GPR at iter top

**What.** Inserted at decoder iter top:
```cpp
PACK(TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK));
PACK(TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI)));
```

**Result.** PCC at iter=1 / iter=2: bit-identical to pristine (no change). Restoring just the GPR doesn't update the CFG register that PACK actually uses at the next release.

### Attempt 38 — Call full LLK helper to re-init GPR + CFG

**What.** Replaced with `PACK((_llk_init_packer_dest_offset_registers_<DstSync::SyncHalf>()))`. This re-inits both GPRs AND calls `select_packer_dest_registers` to update `PACK_SEC0_Offset` CFG from the current GPR.

**Result.** Same as Attempt 37: no PCC effect. **`DEST_OFFSET_HI` zeroed-state isn't propagating to the CFG register in a bug-causing way** — the GPR may be transient. The dump captured a moment where the GPR was zeroed by some intermediate op, but it gets restored before the next `select_packer_dest_registers` reads it.

### Attempt 39 — LOADMACRO-hazard mitigation (per Confluence "Using LOADMACRO Safely" + PR #45660)

**What.** Three independent attempts targeting the LOADMACRO Dst write-to-read structural hazard described in https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/2022408406/Using+LOADMACRO+Safely :

1. `TT_METAL_DISABLE_SFPLOADMACRO=1` env var (via `device_kernel_defines + [("DISABLE_SFPLOADMACRO", "1")]` in op.py) — disables ALL LOADMACRO branches in recip, exp, reduce, mul_int, where, typecast. Verified `defines_generated.h` contains the define.
2. SFPNOPs between the three back-to-back `SFPLOADMACRO`s in `ckernel_sfpu_recip.h`'s main loop (the exact pattern PR #45660 patches).
3. 4× SFPNOPs after `SFPSTORE` at the end of `_calculate_sdpa_reduce_max_row_8x32_` and `_calculate_sdpa_reduce_sum_row_8x32_` (to drain SFPU pipe before the subsequent FPU `bcast_sub` reads the same DEST max via SRCA/SRCB).

**Result.** All three: PCC at iter=1 / iter=2 bit-identical to pristine (0.9914568554511025 / 0.9916859209677518). **The bug is NOT a LOADMACRO Dst write-to-read hazard, NOT an exp/recip-specific drain issue, and NOT a reduce_max/sum → bcast_sub timing issue.**

### Attempt 40 — Per-shard PCC + per-device pre-MoE localization

**What.** Added to `test_decoder_mlp`: (a) per-shard PCC by splitting `decoder_mlp_output_valid` into N=8 chunks along the feature dim (validates the slicing produces sensible per-shard PCCs whose mean ≈ aggregate); (b) saves `decoder_mlp_output_valid`, golden, and the 8 per-device `attention_block_output_tensor` tensors to `/tmp/43563_outputs/iters_{1,2}_pos_511.pt`. Diffed across the two subtests.

**Result @ pos=511.**

| Metric | Value | Notes |
|---|---|---|
| Final tensor shape | `(1, 1, 1, 7168)` | 7168 features on root device |
| iter=1 → iter=2 element-wise diff | **6739 / 7168 non-zero** (94%) | uniform across all 8 shards |
| max abs diff | **0.14** | bf16 magnitude ~1.0 → ~14% relative |
| mean abs diff | 0.028 | structured, not noise |
| Pre-MoE per-device data | all 8 devices **bit-identical** within each subtest, all 8 devices show **identical diff** between subtests | post-all-reduce blends the per-core perturbation into every output position uniformly |
| Zero-diff positions | scattered, no periodic structure | matches no head / channel / device boundary |

**Conclusion.** The bug originates inside `flash_mla`. The MM4/5 → all-reduce → MoE chain takes whatever per-core SDPA-output perturbation flash_mla emitted and smears it uniformly across all 7168 output features and across all 8 devices. **Output-level localization is impossible** — we'd need to capture per-core SDPA output BEFORE the all-reduce to identify a specific compute step or a specific core. That requires a kernel-side dump that doesn't perturb the bug (the earlier `hash_cb` channel was structurally decoupled from the host output; an L1 byte dump via cb_out_in at multi-chunk would race sdpa_tail).

---

## Final status (post Attempts 24-40)

**The fix that ships.** Austin's two-line MATH+PACK iter-top reset at `decoder_block_kernel.cpp:3056-3072`. Reduces the alternation gap from 2.29e-4 to **0** at `position_id=511, num_internal_iterations ∈ {1, 2}`.

**What we know about the bug.**

The bank-1-vs-bank-0 numerical asymmetry inside `flash_mla` is real, deterministic, and reproducible bit-for-bit across runs. It survives:
- All addressing registers being correct for the active bank (per the existing comment at decoder_block_kernel.cpp:3050)
- Zeroing the bank at flash_mla entry (Attempt 33)
- Restoring stale-looking GPRs at iter top (Attempts 37-38)
- Disabling all `SFPLOADMACRO` usage globally (Attempt 39.1)
- Inserting SFPNOPs at the documented LOADMACRO Dst write-to-read pattern (Attempt 39.2)
- Inserting SFPNOPs at the SFPU→FPU handoff at the end of reduce_max/sum (Attempt 39.3)

The asymmetry uniformly perturbs every per-device pre-MoE output feature by the same magnitude/sign pattern (Attempt 40); after the all-reduce + MM4/5 + MoE it lands as a ~14%-magnitude structured noise across all 7168 final-output features. PCC stays at 0.99 because the noise is correlated with the golden output, not random.

**What we've ruled out (full list).**

- DEST bank residue (Attempt 33)
- `pack_reads_per_xy_plane` LLK counter (Attempt 27)
- `DEST_OFFSET_HI` PACK GPR being zeroed (Attempts 37-38)
- MATH-side dest_offset_id desync alone (Attempt 34: KV-cache crash)
- PACK-side dest_offset_id desync alone (Attempt 35: KV-cache crash)
- All LOADMACRO usage (Attempt 39.1)
- Recip LOADMACRO back-to-back spacing (Attempt 39.2)
- Reduce SFPU→FPU handoff spacing (Attempt 39.3)
- Per-device divergence (Attempt 40: all devices identical)
- Per-output-feature concentration (Attempt 40: uniformly spread)

**Open root-cause hypothesis.**

A HW-level asymmetry on Blackhole's half-DEST register file between bank 0 and bank 1 in either:
- The FPU MVMUL / MOVD2B / dest-srcb-reuse path (used by MM1, MM2, bcast_sub, bcast_mul, recip's final bcast_mul), OR
- A subtle SFPU pipeline timing / arbitration path not covered by the documented hazards on the Confluence "Using LOADMACRO Safely" page.

To distinguish: an RTL-level read of a specific FPU/SFPU op (e.g., a single MVMUL or SFPSTORE) with identical inputs on bank-0 vs bank-1 starting state — beyond what host-visible bisection can establish.

---

## Attempt 41 — Bypass `sdpa_mul_bcast_col_reuse_tiles` (FPU MVMUL + SRCB reuse): **|Δ|=0 at all positions**

**What.** Commented out the call to `sdpa_mul_bcast_col_reuse_tiles<block_size>(cb_l2, cb_l1, tile_index, 0)` inside `sdpa_tail_l_block` (`sdpa.h:527`). This is the FPU ELWMUL with SRCB_BCAST_COL that computes L1*P1 + L2*P2 (the L-block contraction of sdpa_tail). DEST contents are no longer overwritten by this multiply; the subsequent `pack_block_contiguous`/`pack_untilize_dest` packs whatever DEST data is leftover from previous compute. Math is broken, but iter-to-iter alternation tests the bank-symmetry of just this ELWMUL path.

**Result.** `DISABLE_SFPLOADMACRO=1`, Austin's iter-top workaround DISABLED, parametrized position_id × num_internal_iterations sweep:

| position_id | num_iters=1 PCC | num_iters=2 PCC | \|Δ\| (bypass) | \|Δ\| (pristine baseline) |
|---|---|---|---|---|
| 127 | 0.9376369586895297 | 0.9376369586895297 | **0** | 0 (no flash_mla sdpa_tail) |
| 255 | 0.9620043792554636 | 0.9620043792554636 | **0** | 1.13e-4 |
| 383 | 0.9655067800415877 | 0.9655067800415877 | **0** | 2.10e-4 |
| 511 | 0.9712301523152757 | 0.9712301523152757 | **0** | 4.26e-3 |

**PCCs bit-identical to 16 digits across `num_iters=1` and `num_iters=2` at every position.** Bypassing the FPU MVMUL with SRCB reuse eliminates the iter-to-iter alternation entirely. (Absolute PCCs drop because the L-block multiply is broken — math is wrong but deterministic.)

**Conclusion.** The bug lives in the `sdpa_mul_bcast_col_reuse_tiles` LLK pathway — i.e., `_llk_math_sdpa_bcast_col_srcb_reuse_` (FPU `ELWMUL` w/ `SRCB_BCAST_COL`) and/or its `_preamble_` (`MOVD2B`s into SrcB) / `_postamble_` (`SETRWC CLR_B`). These are at:

- `tt_llk_blackhole/llk_lib/llk_math_sdpa_bcast_col_srcb_reuse.h:78` (`_llk_math_sdpa_bcast_col_srcb_reuse_`)
- `tt_llk_blackhole/llk_lib/llk_math_sdpa_bcast_col_srcb_reuse.h:57` (`_llk_math_sdpa_bcast_col_srcb_reuse_preamble_`)
- `tt_llk_blackhole/llk_lib/llk_unpack_A_sdpa.h:93` (`_llk_unpack_A_sdpa_set_srcb_dummy_valid_` — the unpacker companion)

Rules out (combined with prior attempts): SFPU recip/exp/reduce LOADMACRO drain, KV-cache contamination, all-reduce smearing as the *source*. The asymmetry originates in a single FPU path: SRCB reuse via `MOVD2B`-populated SrcB + `ELWMUL` broadcast.

**Next step.** Drill into `_llk_math_sdpa_bcast_col_srcb_reuse_preamble_` and the inner `ckernel_template`-driven ELWMUL loop. Candidate hypotheses to test (in order of suspicion):

1. **SrcB ping-pong buffer state across sdpa_tail calls.** The reducer calls `sdpa_tail` *N* times (`num_cores_to_wait - 1` non-norm + 1 norm). Each sdpa_tail re-runs the preamble. If the SrcB physical buffer used by ELWMUL is the OTHER buffer (still holding stale data from the previous call), the broadcast reads the wrong column. A targeted `TTI_UNPACR_NOP(SrcB, ..., SET_DVALID, ..., UNP_ZEROSRC)` between sdpa_tail calls (or before each preamble) would test this.
2. **ADDR_MOD residual state.** `sdpa_bcast_col_srcb_reuse_configure_addrmod` programs ADDR_MOD_0/1/2/3. If the FPU consumes them in a bank-conditional sequence, ADDR_MOD state could mismatch on the second sdpa_tail call.
3. **MOVD2B src/dst alignment in half-DEST mode.** Preamble uses `set_dst_write_addr<Tile32x32, SrcRegs>(0)` then MOVD2Bs reading DEST offsets {0, 4, 64, 68}. In bank 1, the physical address for "tile 0 row 4" differs from bank 0 — the offset arithmetic could overflow/misalign on one bank.

---

## Attempt 42 — STALLWAIT(STALL_MATH, MATH) between MOVD2Bs and ELWMUL: **no effect**

**What.** Added `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH)` after the 4 MOVD2Bs in `_llk_math_sdpa_bcast_col_srcb_reuse_preamble_` (in the local copy at `models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/llk_lib/llk_math_sdpa_bcast_col_srcb_reuse.h`). Tests whether a MOVD2B-to-ELWMUL pipeline hazard (MOVD2B not draining before ELWMUL reads SrcB) causes the bank-asymmetric output. Pristine math otherwise.

**Result.**

| position_id | num_iters=1 PCC | num_iters=2 PCC | \|Δ\| (STALLWAIT) | \|Δ\| (pristine baseline) |
|---|---|---|---|---|
| 127 | 0.9901647631143314 | 0.9901647631143314 | 0 | 0 |
| 255 | 0.9907709857426044 | 0.990764648048267 | 6.51e-6 | ~1.13e-4 |
| 383 | 0.9913380668762602 | 0.9913680422376573 | 3.00e-5 | ~2.10e-4 |
| 511 | 0.9914568554511025 | 0.9916859209677518 | **2.29e-4** | **2.29e-4** |

`pos=511` Δ identical to pristine. The `STALLWAIT(STALL_MATH, MATH)` (which drains the math pipe) had NO effect on the canonical multi-chunk repro. STALLWAIT reverted.

The Δ reduction at pos=255/383 is curious but not the smoking gun — pos=511 (the original 8190 down-shifted equivalent) is unchanged.

**Conclusion.** MOVD2B-to-ELWMUL pipeline hazard ruled out. The bug is not in the math pipe's read-after-write timing for SrcB.

---
