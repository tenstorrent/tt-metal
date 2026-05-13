# Half-DEST Iter-PCC Investigation — Conversation Summary

> Trail of the investigation into the alternating MLP PCC observed in `test_decoder_mlp` on Blackhole half-DEST mode. This document is the narrative companion to [`half-dest-workload.md`](half-dest-workload.md), which is the static, code-grounded reference.

---

## 1. The bug

`test_decoder_mlp` on `aho/sdpa-ops` @ `cb192340ac`, 4×2 Blackhole mesh, position_id=8190, half-DEST mode (`dst_full_sync_en=False`):

| Run | MLP PCC vs torch golden |
|---|---|
| `num_internal_iterations=1` (iter 0 only) | **0.9931898601668998** |
| `num_internal_iterations=2` (iter 0 + iter 1) | **0.9934483676630709** |

Δ ≈ 2.6e-4, reproducible. The PCC threshold in the test (0.975) passes either way, but the parity-dependent drift is real and tracked in [tt-metal#43563](https://github.com/tenstorrent/tt-metal/issues/43563).

Austin Ho's working empirical fix is to re-issue
```cpp
MATH((llk_math_pack_sync_init<false>()));
PACK((llk_pack_dest_init<false, false>(0)));
```
at the top of every iteration of the `while (true)` loop in `decoder_block_kernel.cpp` (line 3055-3058). With it, both 1- and 2-iter runs produce bit-identical 0.9931898601668998. Without it, the parity-dependent split returns.

The investigation set out to understand *why* this works and *what the actual root cause is*, with a stretch goal of replacing it with a surgical fix.

---

## 2. Ablation campaign

After resolving JIT-cache cold-compile timeouts (the kernel takes ~5+ min to JIT-compile when the cache is cold; the default `pytest-timeout=300` in `pytest.ini` kills runs prematurely — use `pytest --timeout=900` for ablations), the following subsets of the Austin workaround were tested. Each ablation replaces the bilateral lockstep call; column "iter 0 / iter 1" shows MLP PCC vs golden (or KV-cache PCC when a downstream KV check fails first).

| Ablation | iter 0 / iter 1 | Verdict |
|---|---|---|
| **No fix (baseline)** | 0.99318986 / 0.99344836 | diverges |
| **PACK-side `llk_pack_dest_init` only** (drop MATH side) | hang (deadlock) | bilateral coord needed |
| **MATH-side `llk_math_pack_sync_init` only** (drop PACK side) | hang (deadlock) | bilateral coord needed |
| **Bilateral barriers only** — `tensix_sync` both threads + MATH_PACK drain + `TTI_SEMINIT(2,0,SEMA_1)`, NO `reset_dest_offset_id`, NO HW CFG writes | 0.99318986 / 0.99344836 | sync alone does NOT fix |
| **Bilateral barriers + SW-only** `reset_dest_offset_id()` on both threads (no `set_dest_section_base`, no PACK HW CFG writes) | 0.99318986 / **KV PCC 0.855** | SW alone CORRUPTS KV cache |
| **Bilateral barriers + HW writes only**, NO SW reset (MATH `set_dest_section_base<StartZero>`, PACK STALLWAITs + SETDMAREG×2 + `select_packer_dest_registers`) | 0.99318986 / 0.99344836 | HW alone preserves divergence |
| **Bilateral barriers + SW + HW resets**, dropping `packer_addr_counter_init` + `pack_sync_tile_dst_ptr=0` from the LLK | 0.99318986 / 0.99318986 | works |
| **Full bilateral lockstep call** (Austin's fix) | 0.99318986 / 0.99318986 | works |

### What the ablation table tells us

1. **Synchronization barriers alone are NOT what fixes the divergence.** The bilateral-barriers-only ablation reproduces the baseline divergence bit-for-bit. The bug is in DEST bank parity, not in cross-thread sync.

2. **SW reset alone is unsafe.** Setting `reset_dest_offset_id()` on both threads without touching HW CFG state desyncs the SW `dest_offset_id` from the HW `DEST_TARGET_REG_CFG_MATH_Offset` and `DEST_TARGET_REG_CFG_PACK_SEC0..3`. The next `dest_section_flip` writes HW based on flipped SW, but the underlying HW state from previous iter is no longer consistent with the now-reset SW value. Subsequent DEST writes go to the wrong bank, corrupting KV cache writes (PCC collapses from 0.992 to 0.855).

3. **HW writes alone don't fix it either.** `select_packer_dest_registers<SyncHalf>` uses `get_packer_dest_offset_index() = (dest_offset_id ? DEST_OFFSET_HI : DEST_OFFSET_LO)`, so the GPR it copies into PACK_SEC0..3 depends on SW `dest_offset_id`. Without SW reset on PACK, the wrong GPR (HALF_SIZE) gets selected and PACR ends up reading from bank 1 while FPU is writing bank 0 → output diverges.

4. **BOTH SW and HW resets on BOTH threads are required.** They are mutually load-bearing.

5. **`packer_addr_counter_init()` (TTI_SETADCXY + TTI_SETADCZW) and `pack_sync_tile_dst_ptr = 0` are NOT load-bearing for this parity fix** — dropping them from `_llk_pack_dest_init_` still produces bit-identical 0.99318986 / 0.99318986.

6. **Calling either side of `tile_regs_*` alone deadlocks** because each LLK includes paired `tensix_sync` / STALLWAITs that wait for the partner thread.

---

## 3. Root-cause hunt — what the disparity actually is

### 3.1 First architectural hypothesis: stale TRISC2 `MATH_Offset`

When the standard `tile_regs_*` cycle flips bank parity, `dest_section_flip` (called from `_llk_math_dest_section_done_` at `tile_regs_commit`) writes **TRISC1's** copy of the per-thread CFG register `DEST_TARGET_REG_CFG_MATH_Offset`. Per the BH ISA (`SFPLOAD.md:84`):

```c
uint10_t Addr = Imm10 + ThreadConfig[CurrentThread].DEST_TARGET_REG_CFG_MATH_Offset;
```

`SFPLOAD` uses the per-thread `MATH_Offset` of whichever thread issues it. Since the LLK API never writes TRISC2's copy, the hypothesis was: when `flash_mla` dispatches SFPU from TRISC2 (the `PACK((...sfpu...))` macro pattern, used five times in `compute_sdpa_chunk` and once in `compute_sdpa_recip`), the SFPU reads from a stale bank that never flips.

### 3.2 Validation by codebase pattern

Greping for who else uses the `PACK(())` SFPU pattern surfaced an established codebase idiom for managing TRISC2 `MATH_Offset`. At `models/demos/deepseek_v3_b1/unified_kernels/matmul.hpp:167`, the fused-silu matmul does:

```cpp
PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
```

right before every PACK-frontend SFPU dispatch. So the matmul authors had already recognized this exact issue and were aligning TRISC2's `MATH_Offset` to the packer's current bank before each dispatch. The hypothesis crystallized as: `flash_mla` simply forgot this idiom.

### 3.3 Surgical fix attempt — FAILED

Two-line patch:
- Remove Austin's iter-top reinit at `decoder_block_kernel.cpp:3055-3058`.
- Add `PACK((TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset())))` at `flash_mla.hpp:748` right after `tile_regs_acquire()`.

**Empirical result: 1-iter=0.9931898601668998, 2-iter=0.9934483676630709 — baseline divergence reproduced bit-for-bit. Zero effect.**

### 3.4 Why the surgical fix was a no-op

A grep over the SDPA helpers shows the SFPU functions in `flash_mla`'s call graph **already** set TRISC2's `MATH_Offset` themselves, on every dispatch, using TRISC2's SW `dest_offset_id`:

- `models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sdpa_reduce_row.h:144,177,202`:
  ```cpp
  TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, src_index + get_dest_buffer_base());
  TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
  ```
- `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h:169` (`fast_approx_exp`), `:180,:200` (`non_approx_exp_mul_prev`), `:226` (`recip_sum`): same pattern.

`get_dest_buffer_base()` reads TRISC2's SW `dest_offset_id`, which IS correctly flipped by `flip_packer_dest_offset_id` inside `_llk_pack_dest_section_done_`. In iter 1 it returns `HALF_SIZE` and the SFPU helpers correctly point `MATH_Offset` at bank 1 — exactly where the FPU and PACR are.

So my one `TT_SETC16` at `tile_regs_acquire` was immediately overwritten by the very next SFPU call. The "matmul.hpp pattern" idiom is real, but it's already being applied — at finer granularity, *inside* each SFPU helper itself.

### 3.5 Reframed picture of the bug

With the surgical theory falsified, what we know is:

- Every named DEST-addressing register IS being correctly tracked across iter boundaries:
  - `TRISC1.MATH_Offset` ← flipped by `dest_section_flip` on `tile_regs_commit`
  - `TRISC2.MATH_Offset` ← re-pointed by the SDPA SFPU helpers themselves (`TT_SETC16(..., src + get_dest_buffer_base())`) before every PACK-frontend SFPLOAD
  - `PACK_SEC0..3` ← flipped by `select_packer_dest_registers` on `tile_regs_release`
  - `TRISC2.SW.dest_offset_id` ← flipped by `flip_packer_dest_offset_id` on `tile_regs_release`
- All four point at the correct bank in iter 1.
- Yet operating in bank 1 produces a numerically different output from operating in bank 0 (0.99344836 vs 0.99318986).

The DEST sub-banks are physically symmetric SRAM halves on the silicon. Any bank-asymmetric numerical behaviour must therefore come from a register or piece of state that:
1. Depends on which sub-bank is active, AND
2. Is NOT routinely re-pointed before each use by the in-tree LLK / SFPU helpers.

This is the root cause that we have not yet identified.

---

## 4. Open candidates for the actual asymmetry

Not investigated in this conversation, listed in rough order of plausibility:

1. **PACK section / DEST-offset registers beyond `PACK_SEC0..PACK_SEC3`.** Specifically, any of the `DEST_OFFSET_LO+0..3` / `DEST_OFFSET_HI+0..3` GPRs (`ckernel_gpr_map.h:85-86`) that `select_packer_dest_registers` reads via `WRCFG_128b`. We confirmed `DEST_OFFSET_LO+0` (= 0) and `DEST_OFFSET_HI+0` (= HALF_SIZE) are set by `_llk_init_packer_dest_offset_registers_`, but `DEST_OFFSET_LO+1..3` / `DEST_OFFSET_HI+1..3` (the per-packer offsets for packers 1-3) are not visibly initialized.

2. **DEST replay-buffer state.** `compute_sdpa_chunk` calls `_init_sdpa_reduce_max_row_8x32_replay_buffers_` (`sdpa.h:265`) and `_init_sdpa_reduce_sum_row_8x32_replay_buffers_` (`sdpa.h:328`) once per chunk. If the replay buffers internally cache a DEST address that gets bank-shifted at first use, priming them in bank 0 in iter 0 and consuming them in bank 1 in iter 1 could leak iter-0 state into iter-1 results.

3. **UNPACK-side DEST mirroring in `dest_srcb_reuse` paths.** `sdpa_custom_mm_reuse_dest_srcb_block` (used for MM2 = softmax(QK) · V) reads DEST through SRCB. The SRCB-reuse base pointer is set via its own `TT_SETC16(...)` (see `llk_math_sdpa_custom_mm.h:101`, `llk_math_sdpa_custom_mm_reuse_dest_srcb.h:122,130`). If that base pointer's bank parity is tracked via a different mechanism than the rest of the system, MM2 could read from a half-flipped bank.

4. **A semaphore-handshake race between threads** that leaves stale `MATH_Offset` from a previous tail's `sdpa_tail_l_block` packing if the order in which threads cross the bank boundary differs between iter 0 and iter 1. The `flash_mla.hpp:793` `MATH(t6_semaphore_wait_on_max<STALL_SFPU>(FPU_SFPU))` after `tile_regs_release` is the visible synchronization point, but nothing guarantees the prior SFPU side has finished updating all its CFG writes by then.

5. **Bank-half-specific HW silicon behaviour.** Either bank rotates a SerDes / shuffle pattern internally, or there's a (presumably documented) feature of FP16/BF16 unshuffle in `Dst16b[...]` accesses that depends on which half is addressed. This is what one would expect to be ruled out by silicon validation, but worth eliminating via direct read-back of bank-0 vs bank-1 contents after identical FPU writes.

---

## 5. State of the tree at end of conversation

- **`models/demos/deepseek_v3_b1/fused_ops/decoder_block/kernels/decoder_block_kernel.cpp:3043-3071`** — Austin's iter-top reinit is in place with an updated comment block summarizing what we've ruled out:
  > "the SDPA SFPU helpers already TT_SETC16 the per-thread MATH_Offset to `src_index + get_dest_buffer_base()` themselves before every PACK-frontend SFPLOAD, so a write to MATH_Offset at iter top is overwritten before any SFPU read. The divergence is therefore not in per-thread MATH_Offset management — bank 1 produces numerically different output from bank 0 even when all known DEST-addressing registers (TRISC1.MATH_Offset, TRISC2.MATH_Offset via SDPA helpers, PACK_SEC0..3) point at the correct bank."

- **`models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`** — unchanged from `cb192340ac` baseline (surgical fix reverted after empirical disproof).

- **`half-dest-workload.md`** — 639-line code-grounded breakdown of what the test runs (test params, derived constants, per-iter timeline, flash_mla detail, CB table, DEST-offset table, parity discussion).

- **`half-dest-conversation-summary.md`** (this doc) — investigation narrative.

---

## 6. Operational notes

- `pytest-timeout` is set to 300s in `tt-metal/pytest.ini`. Cold-cache JIT compile of `decoder_block_kernel` can take ~5+ minutes; the default timeout kills runs that would otherwise succeed. Use `pytest --timeout=900` (or higher) when iterating.
- `tt-smi -r --use_luwen` (or `-r all`) is the canonical device reset between runs. Without it, kernels left in deadlocked state from prior failed runs persist on the chip.
- Killing pytest mid-run with `pkill -9` can leave fabric in an inconsistent state. Wait for graceful termination if possible; if the run is genuinely hung, follow `pkill -9` with `tt-smi -r all`.
- The kernel is JIT-compiled. Editing source files (`decoder_block_kernel.cpp`, `flash_mla.hpp`, `sdpa.h`, etc.) is sufficient to test changes; **no host rebuild required**. The JIT cache lives at `/home/ncvetkovic/.cache/tt-metal-cache/`.
- The cache hash is content-addressed over the source text, so even comment-only changes invalidate the cache for affected kernels.

---

## 7. References

- GitHub issue: [tenstorrent/tt-metal#43563](https://github.com/tenstorrent/tt-metal/issues/43563) — "Deepseek Blitz Alternating PCC", filed by tt-aho.
- Workload doc: [`half-dest-workload.md`](half-dest-workload.md) — code-grounded reference for `test_decoder_mlp`.
- BH ISA documentation: `/localdev/ncvetkovic/work/tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md` (lines 84-95 specify the `Imm10 + ThreadConfig[CurrentThread].DEST_TARGET_REG_CFG_MATH_Offset + DstOffset + (SpOffset & 3)` SFPLOAD address composition).
- Reference SDPA workload breakdown by C. Glagovich: https://gist.github.com/cglagovichTT/87b6fd389061f58497c875b6afa4039b (style template for `half-dest-workload.md`).
