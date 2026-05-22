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
