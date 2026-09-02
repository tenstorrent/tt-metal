# Wormhole B0 Packer Kernel Audit — Improvement Plan

Static read of the Wormhole B0 packer LLKs, in the same shape as
[blackhole_pack_kernel_audit.md](blackhole_pack_kernel_audit.md). Read that one first if you have
not: the method, the invariants section and several findings are shared, and this document is
written as the delta.

> **Status: nothing here is measured.** Same caveat as the Blackhole packer document — every figure
> is quoted from another file's comment or is an unmeasured estimate. Contrast with
> [blackhole_fpu_math_kernel_measurements.md](blackhole_fpu_math_kernel_measurements.md), where
> every number is a hardware A/B.

## Headline

The two architectures are further apart on the pack thread than on the math thread, and the
difference runs **both ways**:

- **Wormhole is already ahead on address programming.** It has a working blocked multi-tile pack
  path that `_llk_pack_` dispatches to automatically, and its per-tile address write needs no fence.
  Blackhole's P3 and P4 are proposals; on Wormhole they are shipped code. The Blackhole document
  should treat `llk_pack.h` here — not just `experimental/llk_pack_fast_tilize.h` — as the reference
  implementation.
- **Wormhole is further behind on config churn.** It has **four packers**, so every un-guarded
  per-section config write costs 4x what the Blackhole equivalent costs. The single worst offender,
  `reconfigure_packer_l1_acc`, is a packer drain plus **four** RMWCIBs with no skip-if-unchanged
  guard, driven from the matmul inner loop.

So: W1 and W2 below are worth more on Wormhole than their Blackhole counterparts; W3 is a
Wormhole-specific hazard; and the Blackhole P3/P4 work should be a port, not a design.

## Structural Differences That Change the Conclusions

| | Wormhole B0 | Blackhole |
|---|---|---|
| Packers | **4** (`NUM_PACKERS = 4`) | 1 |
| L1 dest address write | `REG2FLOP` — ThCon, in program order after the `SETDMAREG` | `WRCFG` — separate Config Unit |
| Fence needed for it | **none**, documented in `cpack_common.h` | `STALLWAIT(STALL_CFG, THCON)` |
| Extra per-call packer op | **`TTI_PACR` pack flush** | none |
| Blocked multi-tile pack | **implemented**, auto-dispatched | MOP supports it; no dispatch, callers must opt in |
| Per-tile Z-counter reset | none | `SETADCZW` (see Blackhole P5) |

The first two rows explain each other. `program_packer_destination` in
[cpack_common.h](../../tt_llk_wormhole_b0/common/inc/cpack_common.h) spells it out:

> No STALLWAIT is needed before this config write (unlike the BH counterpart): REG2FLOP executes on
> ThCon, in program order after the SETDMAREG GPR writes above, so it reads the freshly written
> OUTPUT_ADDR without a fence (BH uses WRCFG on the separate Config Unit, hence its STALL_CFG/THCON
> guard).

**Do not port fences between these two files in either direction without re-reading that comment.**

## Per-Tile Cost Inventory

`_llk_pack_` in [llk_pack.h](../../tt_llk_wormhole_b0/llk_lib/llk_pack.h), single-tile Default mode:

| Step | Instructions |
|---|---|
| `set_dst_write_addr(tile_index)` | 1 x `SETADC` |
| `program_packer_destination(address)` | 2 x `SETDMAREG`, `REG2FLOP`, **`PACR` (flush)**, 1 x `SETDMAREG` (restore) |
| `mop_run(1, 1)` | 1 issued; MOP body is a single closing `PACR` |
| Untilize only | 1 extra `PACR` to close the tile |

Roughly **7 issued instructions per tile**, one of which is a real packer operation rather than
RISC-V bookkeeping. Compared with Blackhole: no `STALLWAIT`, no `DMANOP`, but a `PACR` flush that
Blackhole does not have.

When `configured_num_tiles > 1`, `_llk_pack_` instead calls `pack_multitile`, which drops the flush
and the GPR restore and lets the MOP advance the address — see W3.

## Findings

### W1 — `reconfigure_packer_l1_acc` is the Blackhole P1 finding, times four

[cpack_common.h](../../tt_llk_wormhole_b0/common/inc/cpack_common.h):

```cpp
inline void reconfigure_packer_l1_acc(const std::uint32_t pack_l1_acc)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);   // full packer drain
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(pack_l1_acc);
    cfg_reg_rmw_tensix<THCON_SEC0_REG8_Pack_L1_Acc_RMW>(pack_l1_acc);
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Pack_L1_Acc_RMW>(pack_l1_acc);
    cfg_reg_rmw_tensix<THCON_SEC1_REG8_Pack_L1_Acc_RMW>(pack_l1_acc);
}
```

One drain plus four RMWCIBs, with no tracker on the bit's value — where Blackhole pays one RMWCIB.

The callers are the *same source files*: `ttnn/.../matmul/device/kernels/compute/` is
architecture-independent, so the pattern described in the Blackhole document applies here unchanged —
one branch calls `llk_pack_reconfig_l1_acc(0)` unconditionally inside the block loop, another writes
1 on every block after the first, and a third site correctly guards with `else if (block == 1)`.
Every one of those calls costs 4 RMWCIBs and a packer drain on Wormhole.

**Fix:** the same F2-shaped tracker as proposed for Blackhole — one guard covering all four section
writes, since they always move together.

**Safety.** The tracker's seed is the same trap: `set_packer_config` calls
`reconfigure_packer_l1_acc(0)` during hw-configure, so hw-configure resets the bit and must reseed
the tracker. `reconfig_packer_data_format` does not touch l1_acc, so a format reconfig correctly
leaves both alone. The failure mode is silent data corruption (accumulate when it should overwrite,
or the reverse), not a hang.

**Verification.** `test_pack_dest_bank.py` sweeps `l1_acc = [No, Yes]` and its source
`sources/pack_dest_bank_test.cpp` calls `reconfigure_packer_l1_acc` directly, on both architectures.
As on Blackhole, confirm there is a case that **toggles** the value within one run (0 -> 1 -> 0, the
matmul pattern) — a test that sets it once would pass with a broken guard.

**Prize: unmeasured**, but strictly larger than the Blackhole equivalent: same drain, four times the
RMW work behind it.

### W2 — packer format reconfig is heavier here, and its redundancy is easier to prove

`set_packer_strides` on Wormhole takes **only** the source format:

```cpp
inline void set_packer_strides(const std::uint32_t pack_src_format)
```

Everything it writes derives from `datum_size_in_bytes(pack_src_format)` and compile-time face
constants — there is not even a `tile_c_dim` input as there is on Blackhole. So when the metal-level
guard admits a reconfig **because the dst format changed**
(`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_common_api.h`, which compares
`pack_dst_format[old] != pack_dst_format[new]` and nothing else), the strides it reprograms are
provably identical: 4 x `SETDMAREG` + 2 x `WRCFG` + 2 x `NOP` of guaranteed-redundant work.

And the surrounding function is much bigger than Blackhole's, because everything is per-section:

- 4 x `REG2FLOP` for the format word (SEC0_REG1, SEC0_REG8, SEC1_REG1, SEC1_REG8)
- up to 4 x `REG2FLOP` for BFP exponent section sizes, or 4 more for the Lf8/Int8 zeroing path
- `reconfigure_exp_threshold` writes **4** `cfg_reg_rmw_tensix`, one per section
- `set_packer_l1_offset`, `TILE_HEADER`, the Dstacc RMW, then the strides

Call it 30-40 instructions and 2 stalls, with no early-out when nothing moved.

**Fix, smallest first:** track `(pack_src_format)` for the strides and early-out; then, separately,
early-out of `reconfig_packer_data_format` as a whole when src format, dst format, geometry and
`tile_size` are all unchanged.

The precedent for the first one exists as a manual flag: `_llk_pack_init_` takes
`skip_packer_strides`, and SDPA streaming passes it deliberately, with a comment saying it "saves a
THCON stall per call on the SDPA streaming hot path". A tracker generalises that to every caller.

**Also inherited from the Blackhole document (W2b):** the metal guard compares dst format only, so
two CBs sharing a dst format but differing in src format would skip a reconfig that
`in_data_format` / `PCK_DEST_RD_CTRL` / the strides all need. Whether that pair is reachable is still
an open question; the overload carries the same `TODO NC: Clean up as the part of tt-metal#34499` on
both architectures. A tracker inside the LLK is immune to the question, which is another argument
for fixing it there.

### W3 — `llk_matmul_pack` cannot use the blocked path, and silently breaks if it is configured

Wormhole's blocked pack is well built. `_llk_pack_mop_config_` programs, for `num_tiles > 1`:

```cpp
ckernel::ckernel_template tmp(
    num_tiles - 1, 1,
    TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0),                      // next dest tile
    TT_OP_ADDDMAREG(p_adddmareg::REG_PLUS_REG, OUTPUT_ADDR, OUTPUT_ADDR, OUTPUT_ADDR_OFFSET));
tmp.set_start_op(TT_OP_PACR(ADDR_MOD_1, ..., 1));                   // pack tile 0
tmp.set_end_ops(
    TT_OP_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, OUTPUT_ADDR),
    TT_OP_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0));
```

with the per-tile L1 stride precomputed into `OUTPUT_ADDR_OFFSET` by
`_llk_pack_output_addr_offset_words_`, and `pack_multitile` skipping both the flush `PACR` and the
GPR restore that the single-tile path pays. That is exactly what the Blackhole document proposes
building.

Two problems, both in the metal layer:

1. **The obvious caller does not use it.** `llk_matmul_pack`
   (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_tile_api.h`) loops `ntiles` times
   calling `_llk_pack_` once per tile — it has the block structure in hand and packs one tile at a
   time anyway. Identical to the Blackhole situation, except here the fast path already exists.
2. **The dispatch is a static, and nothing checks it.** `_llk_pack_` branches on
   `llk_pack_internal::configured_num_tiles > 1`, a file-static set by `_llk_pack_mop_config_`. A
   kernel that calls `llk_pack_init(ocb, 4)` and then `llk_matmul_pack(start, ocb, 8)` would pack
   **4 tiles per iteration, 32 in total**, each iteration re-basing the address — silent overwrite,
   no assert, no hang.

**Proposed:** add an `LLK_ASSERT(configured_num_tiles == 1)` to the per-tile loop in
`llk_matmul_pack` first — it is free and closes the footgun. Then, separately, give
`llk_matmul_pack` the blocked path, clamped as W6 describes.

### W4 — the per-tile flush PACR, and the `restore` argument almost nobody passes

`program_packer_destination` issues `TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0)` — described in the
code as "pack flush", which "drains the previous pack's output buffers and arms a fresh start
address" — on **every** single-tile pack. Blackhole has no equivalent; its argument for needing
nothing is that the `Last = 1` PACR ending the previous MOP already drained the packer.

Whether the same argument applies here (the Wormhole single-tile MOP body is also a `Last = 1`
PACR) is not something a static read can settle: the flush also "arms a fresh start address", which
may be doing work the closing PACR does not. **This is a question for someone with the hardware
definition, not a proposed change** — removing a flush that turns out to be load-bearing corrupts
the first row of every tile.

Cheaper and safer in the same function: `program_packer_destination(addr, bool restore = true)`
rewrites the upper half of `OUTPUT_ADDR` to clear bit 31 after the `REG2FLOP`. Only
`llk_pack_fast_tilize.h` passes `restore = false`; `_llk_pack_` and `llk_pack_rows.h` both take the
default and pay an extra `SETDMAREG` per tile. Work out what reads `OUTPUT_ADDR` between calls —
note that the multi-tile MOP's `ADDDMAREG` accumulates into that GPR, which is very likely why the
restore exists at all — and if nothing does on a given path, pass `false` there.

### W5 — an unfenced `SETDMAREG` -> `WRCFG` in `set_packer_strides`, worth understanding before porting

`set_packer_strides` writes GPRs with `SETDMAREG` and then reads them with `WRCFG`, with no
`STALLWAIT` between and two trailing `NOP`s:

```cpp
TT_SETDMAREG(0, LOWER_HALFWORD(xy_stride), 0, LO_16(p_gpr_pack::TMP0));
TT_SETDMAREG(0, UPPER_HALFWORD(xy_stride), 0, HI_16(p_gpr_pack::TMP0));
TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
...
TTI_NOP;
TTI_NOP;
```

That is the exact producer/consumer pair the Blackhole file fences with
`STALLWAIT(STALL_CFG, THCON)`, and whose asymmetry `program_packer_destination` documents as a
`WRCFG`-vs-`REG2FLOP` difference — but here `WRCFG` is used *without* the guard.

**This is a question, not a bug report.** Wormhole is long-validated and this code has shipped for
years, so either the ordering rules differ, the trailing `NOP`s cover it, or the window is not
reachable in practice. But it is precisely the kind of asymmetry that produces a rare,
layout-dependent failure when someone copies a sequence between the two `cpack_common.h` files. Get
it written down before anyone does.

### W6 — no dest-capacity assert on the multi-tile path

Blackhole's `_llk_pack_init_` asserts `num_tiles <= 4` for Float32 and `<= 8` for Float16/Float16_b —
the dest register cannot hold more. Wormhole's `_llk_pack_init_` has no such check; its multi-tile
asserts cover `num_faces == 4`, `!partial_face` and `!narrow_tile` only.

Since the blocked MOP advances the dest tile with `INCADCZW` per tile, an over-large `num_tiles`
walks off the end of the configured dest half rather than failing. Note Wormhole's init takes
`pack_dst_format` where Blackhole's takes `pack_src_format`, so the assert cannot be copied verbatim —
the capacity depends on the dest-register format, which this signature does not receive. Either
thread the src format through or assert against `get_pack_dest_max_tiles` at the metal layer, where
`llk_matmul_pack` already does exactly that for its own range.

Robustness, not performance — but it becomes load-bearing the moment W3 gives `llk_matmul_pack` a
blocked path.

## Where There Is No Room

- **The per-tile address write is already minimal.** No fence, and the `REG2FLOP` ordering argument
  is documented. Blackhole's P4 (three competing mechanisms, mainline using the most expensive) has
  no Wormhole analogue — this file already uses the cheap one.
- **Blocked multi-tile pack exists, is auto-dispatched, and is tested.**
  `sources/pack_dest_bank_test.cpp` passes `num_tiles_in_block` through to `_llk_pack_init_`, and
  `test_pack_dest_bank.py` sweeps it against formats, dest_acc, num_faces and tilize. The path is
  covered; it is the *callers* that do not reach for it.
- **`_llk_pack_relu_config_` is already the RMWCIB form** under `mutex::REG_RMW`, same as Blackhole.
- **`_llk_pack_dest_section_done_`'s `STALLWAIT(STALL_MATH, PACK)` is required** — it gates the
  `ZEROACC` that clears dest against the packer still reading it. Same as Blackhole. Leave it.
- **No per-tile Z-counter reset to question.** Blackhole's P5 has no counterpart: Wormhole's
  `_llk_pack_` issues no `SETADCZW`, only an extra closing `PACR` in Untilize mode.

## Verification Plan

Identical in shape to the Blackhole plan, and the harness is architecture-aware
(`helpers/chip_architecture.py` resolves `wormhole` / `wormhole_b0`), so the same suites run here:

1. Read `mean(PACK_ISOLATE)` at `marker == TILE_LOOP` from the existing runs **before** changing
   anything — `perf_pack.py` runs `ALL_PERF_RUN_TYPES` over the full `PACK_SWEEP`, and
   `perf_reduce.py` / `perf_matmul.py` / `perf_eltwise_binary.py` collect it alongside the math and
   unpack columns.
2. A/B the same commit with and without the change; fresh `RUNNER_TEMP` per variant; wipe
   `perf_data/runs` and `perf_data/latest` between variants.
3. Carry flat controls and quote them.
4. Correctness, minimum set:
   - W1: `test_pack_dest_bank.py`, `test_pack_tiny_tile_block.py` (both sweep `l1_acc`), plus a
     toggling case.
   - W2: `test_pack.py`, `test_experimental_reconfig_escape.py`, `test_sdpa_reinits.py`,
     `test_tilize_transition_reconfig.py`.
   - W3: `test_pack_dest_bank.py` (the one suite that exercises `num_tiles > 1`), `test_matmul.py`.
   - Anything in `cpack_common.h` reaches every pack kernel and deserves the broad sweep.

Note that Wormhole's four packers mean a mistake in a per-section write can corrupt one quarter of
the output rows and leave the rest correct — a partial-tile corruption pattern that is easy to
misread as a math bug. Check full tiles, not sampled datums.

## Invariants

Everything in the Blackhole document's invariants section applies here, with these substitutions:

- Where that document says the address write needs a `STALLWAIT` fence, Wormhole relies on
  **`REG2FLOP` program order instead** — see W5 for the one place that ordering argument does not
  obviously cover.
- **All four packer sections must be written together.** `Pack_L1_Acc`, the format word, the exp
  section sizes and the exp thresholds are each written to SEC0_REG1, SEC0_REG8, SEC1_REG1 and
  SEC1_REG8. Any guard added must skip all four or none; skipping a subset leaves the packers
  disagreeing, which is the partial-tile corruption pattern above.
- **`configured_num_tiles` and `configured_zero_output` are file-statics that outlive a single op.**
  They are set by `_llk_pack_mop_config_` and read by `_llk_pack_` on every call. Any code path that
  changes the MOP without going through `_llk_pack_mop_config_`, or any caller that assumes
  single-tile behaviour, must be reconciled with them (W3).

## Open Questions for Someone With the Hardware Definition

1. Is the flush `PACR` in `program_packer_destination` still required when the previous pack ended
   with a `Last = 1` PACR (W4)? Blackhole argues it is not needed there; the two files disagree and
   only one can be describing the hardware correctly.
2. Does `SETDMAREG` -> `WRCFG` need a fence on Wormhole (W5)? The Blackhole file says the equivalent
   pair does on its architecture and explains why; `set_packer_strides` here does not use one.
3. What reads `OUTPUT_ADDR` between packs, such that `program_packer_destination` restores its upper
   half (W4)? The multi-tile MOP's `ADDDMAREG` accumulator is the obvious candidate — confirming that
   would let the single-tile path pass `restore = false`.
4. Same question as Blackhole: is per-byte `RMWCIB` atomic across threads? It underpins the trackers
   proposed here, the Blackhole ones, and the existing `configure_pack` comments on both.
