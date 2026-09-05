# Blackhole Packer Kernel Audit — Improvement Plan

Static read of the Blackhole packer LLKs, listing where the pack thread does work it does not have
to, and what each change has to preserve to be safe.

> **Status: nothing here is measured.** Every cycle figure in this document is either quoted from
> another file's own comment or is an unmeasured estimate, and is labelled as such. This is the
> opposite of [blackhole_fpu_math_kernel_measurements.md](blackhole_fpu_math_kernel_measurements.md),
> where every number is an A/B on hardware. Treat this as a work queue, not as results.

Companion documents: [blackhole_fpu_math_kernel_audit.md](blackhole_fpu_math_kernel_audit.md) covers
the math thread and defines the method these findings reuse.

## Quick Links

- Pack kernels: [llk_pack.h](../../tt_llk_blackhole/llk_lib/llk_pack.h),
  [llk_pack_common.h](../../tt_llk_blackhole/llk_lib/llk_pack_common.h),
  [llk_pack_untilize.h](../../tt_llk_blackhole/llk_lib/llk_pack_untilize.h),
  [llk_pack_rows.h](../../tt_llk_blackhole/llk_lib/llk_pack_rows.h)
- Shared pack state: [cpack_common.h](../../tt_llk_blackhole/common/inc/cpack_common.h)
- Reference implementation of the cheap idioms:
  [experimental/llk_pack_fast_tilize.h](../../tt_llk_blackhole/llk_lib/experimental/llk_pack_fast_tilize.h)
- Perf harnesses: [perf_pack.py](../../tests/python_tests/perf_pack.py),
  [perf_pack_dest_bank.py](../../tests/python_tests/perf_pack_dest_bank.py),
  [perf_pack_untilize.py](../../tests/python_tests/perf_pack_untilize.py)
- Correctness: [test_pack.py](../../tests/python_tests/test_pack.py),
  [test_pack_dest_bank.py](../../tests/python_tests/test_pack_dest_bank.py),
  [test_pack_tiny_tile_block.py](../../tests/python_tests/test_pack_tiny_tile_block.py)

## Step 0 — Read the data that already exists before changing anything

`PerfRunType.PACK_ISOLATE` is already collected by `perf_reduce.py`, `perf_matmul.py` and
`perf_eltwise_binary.py`, and [perf_pack.py](../../tests/python_tests/perf_pack.py) runs
`ALL_PERF_RUN_TYPES` over the full `PACK_SWEEP` (formats x dest_acc x relu x dest_sync x input
dimensions). The FPU audit's runs therefore produced pack numbers that were never read.

Do this first, because it decides whether any of P3/P4/P5 below is worth doing at all:

1. Pull `mean(PACK_ISOLATE)` at `marker == TILE_LOOP` from the existing
   `perf_data/latest/*/*.post.csv` files, next to the `UNPACK_ISOLATE` and `MATH_ISOLATE` columns
   already quoted in the FPU audit.
2. Compare against that kernel's unpack bound. If pack sits well under both unpack and math, the
   pack thread is not the limiter for that op and the issue-side findings (P3, P4, P5) buy nothing
   end to end — exactly the argument the FPU audit used to close out matmul and eltwise binary.
3. P1 and P2 are worth doing regardless of where pack sits, because they remove a *packer drain*,
   which stalls the packer itself rather than merely occupying RISC-V issue slots.

## Per-Tile Cost Inventory

What [`_llk_pack_`](../../tt_llk_blackhole/llk_lib/llk_pack.h) issues for one tile, Default mode:

| Step | Instructions | Notes |
|---|---|---|
| `set_dst_write_addr(tile_index)` | 1 x `SETADC` (PAC, CH_0, SET_W) | absolute W, cheap |
| `program_packer_destination(address)` | 2 x `SETDMAREG`, `STALLWAIT(STALL_CFG, THCON)`, `WRCFG`, 1 x `SETDMAREG`, `DMANOP` | 6 instructions, absolute L1 address recomputed on the RISC-V |
| `ckernel_template::run()` | 1 issued | MOP body: `num_faces * num_tiles` PACRs |
| Z-counter reset | 1 x `SETADCZW` | see P5 — may be dead |

That is roughly **8 issued instructions of overhead around the MOP per tile**. For comparison, the
math thread's matmul path was solved at 2.82 cycles/tile of fixed overhead for 3 issued instructions.

**The important qualifier:** none of this is a packer drain. The one `STALLWAIT` is
`STALL_CFG, THCON` — a GPR-producer fence ensuring the `SETDMAREG` retires before `WRCFG` reads it —
and [cpack_common.h](../../tt_llk_blackhole/common/inc/cpack_common.h) documents at length why no
`p_stall::PACK` drain is needed here (the packer latches `L1_Dest_addr` at PACR start, and the
`Last=1` PACR that ends each MOP drains it). So this overhead overlaps with the packer still working
on the previous tile, and only costs when pack *issue* is the limiter. Do not model it the way F1's
drains were modelled.

## Findings

Ranked by confidence x prize. Status is "proposed" for all of them; none is implemented.

### P1 — `reconfigure_packer_l1_acc` has no skip-if-unchanged guard

This is [F2](blackhole_fpu_math_kernel_audit.md) again, on the pack thread, and it is the one finding
here that removes a genuine drain.

[`reconfigure_packer_l1_acc`](../../tt_llk_blackhole/common/inc/cpack_common.h) is:

```cpp
TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);   // full packer drain
cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(pack_l1_acc);
```

unconditionally, with no tracker on the bit's physical value.

The callers make it expensive. In
`ttnn/.../matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`:

- the non-`FUSE_BIAS` branch calls `llk_pack_reconfig_l1_acc(0)` **unconditionally inside the block
  loop** — writing 0 over 0 on every block;
- the `FUSE_BIAS` branch writes 1 on every block after the first, so only the block 0 -> 1
  transition actually moves the bit;
- and a third site ~20 lines further down *does* guard it (`else if (block == 1)`).

The same file containing both the guarded and unguarded idiom is good evidence the churn is
accidental. A dozen more kernels call this API (`grep -rl pack_reconfig_l1_acc ttnn/`), so the
durable fix belongs in the LLK where it fixes every caller at once, not in the kernels.

**Proposed fix.** Mirror F2 exactly: a `pack_l1_acc_hw` tracker, an out-of-line `_apply_...`, an
inlined `_configure_...` guard that returns early when the value has not moved, and a
`_seed_pack_l1_acc_state_` for the hw-configure path.

**Why it is safe to cache — and the seed that must not be missed.**

`Pack_L1_Acc` is bit 19 of cfg word 71
(`THCON_SEC0_REG1_Pack_L1_Acc_ADDR32 == 71`, `_SHAMT == 19`). The other live fields in that word are
`Exp_threshold_en` (20), `Unp_LF8_4b_exp` (22) and `Pac_LF8_4b_exp` (23). Bit 22 is written by the
**unpack** thread, and only ever through `cfg_reg_rmw_tensix` — never a whole-word `WRCFG_32b`
(checked: every `Unp_LF8_4b_exp` write in `llk_unpack_common.h`, `llk_unpack_reduce.h` and
`cunpack_common.h` is an RMW). So no other agent moves bit 19, which is the condition a tracker
needs.

Bits 19 and 22 share byte 2, so cross-thread correctness rests on per-byte `RMWCIB` atomicity — the
same assumption `configure_pack` already relies on for the `ALU_FORMAT_SPEC_REG` Dstacc fields, and
the same one F2 relied on. It is pre-existing, not introduced here, but a reviewer with the hardware
definition should confirm it once for both.

The seed is the part that breaks silently if missed: `set_packer_config` calls
`reconfigure_packer_l1_acc(0)` as part of hw-configure, so **hw-configure resets the bit to 0** and
the tracker must be reseeded there. `reconfig_packer_data_format` does *not* touch l1_acc, so a
format reconfig correctly leaves both the bit and the tracker alone.

**Failure mode if the seed is wrong:** a stale tracker skips a write that was needed, and the packer
either accumulates into L1 when it should overwrite (garbage output that grows with block count) or
overwrites when it should accumulate (silently wrong matmul results on all but the last block). Both
are data-corruption, not hangs, so they will not show up as a crash.

**Verification.** `test_pack_dest_bank.py` and `test_pack_tiny_tile_block.py` sweep
`l1_acc=[No, Yes]` across formats, dest_acc, num_faces and tilize, and
`sources/pack_dest_bank_test.cpp` / `pack_tiny_tile_block_test.cpp` /
`pack_tiny_tile_reconfig_test.cpp` call `reconfigure_packer_l1_acc` directly. That is real coverage,
but note the limitation: a test that sets the value **once** exercises only the tracker's write path
and would pass even if the guard were wrong. Before trusting it, add (or confirm) a case that
**toggles** the value within a single run — 0 -> 1 -> 0 across blocks, which is the matmul pattern —
so the skip path is actually taken and then correctly un-skipped.

**Prize: unmeasured.** It is one `STALLWAIT(STALL_CFG, PACK)` per avoided call. The FPU audit
measured the analogous math-side drain at 2.8-10.0 cycles depending on how full the pipe was; the
packer equivalent should be budgeted the same way — by where it sits, not by counting. In the matmul
inner loop it is one per block.

### P2 — packer format reconfig rewrites strides that cannot have changed

[`set_packer_strides`](../../tt_llk_blackhole/common/inc/cpack_common.h) derives everything from
`datum_size_in_bytes(pack_src_format)`, `tile_c_dim` and the pack mode:

```
x_stride = datum_size_in_bytes(pack_src_format)
y_stride = FACE_C_DIM * x_stride
w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * x_stride
z_stride = (pack_mode != Default && tile_c_dim == TILE_C_DIM) ? 2 * FACE_R_DIM * y_stride
                                                              : FACE_R_DIM * y_stride
```

It costs 4 x `SETDMAREG` + `STALLWAIT(STALL_CFG, THCON)` + 2 x `WRCFG` + 2 x `NOP`, and
`reconfig_packer_data_format` calls it unconditionally.

The only guard is at the metal level in
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_common_api.h`, and it compares the **dst**
format only:

```cpp
if ((pack_dst_format[old_output_id] != pack_dst_format[new_output_id]) && ...) {
    llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(new_output);
}
```

So the common "same dest-register format, different L1 format" reconfig rewrites byte-identical
strides. Same driver-coarser-than-value asymmetry F2 fixed. The single-argument overload has no
guard at all.

**Proposed fix, in increasing order of scope:**

1. Track the `(pack_src_format, tile_c_dim, pack_mode)` triple that the strides were last programmed
   from and early-out of `set_packer_strides` when it has not moved. Smallest, safest, fixes both
   overloads and the no-guard path.
2. Additionally early-out of `reconfig_packer_data_format` as a whole when src format, dst format,
   tile geometry and `tile_size` are all unchanged — it is ~20 instructions and 2 stalls.

Note that (1) is already available as a template flag on the init path
(`skip_packer_strides`), which SDPA streaming uses deliberately: `configure_pack_width` passes
`skip_packer_strides = true` and its comment says this "saves a THCON stall per call on the SDPA
streaming hot path". The precedent exists; it is just manual and per-caller today.

**Safety.** The tracker must be seeded by `configure_pack`, which programs strides during
hw-configure. `_llk_pack_init_` also programs them unless `skip_packer_strides` is set — with a
tracker, that call becomes a no-op when nothing moved, which is the intent, but it means the
tracker's state must survive across init calls and be invalidated by hw-configure.

**Failure mode if wrong:** wrong y/z/w strides mean the packer walks dest with the wrong step and
writes transposed or interleaved garbage — loud, and caught by any format sweep.

**Verification.** `test_pack.py` (the `PACK_SWEEP` behind `perf_pack.py`) plus the reconfig suites
already used as F2's evidence: `test_experimental_reconfig_escape.py`,
`test_tilize_transition_reconfig.py`, `test_matmul_and_unary_sfpu.py`, `test_sdpa_reinits.py`. The
sequence that matters is *format A -> format B -> format A within one run*, which is what the escape
and transition suites do.

### P2b — open question, not a claim: the metal guard compares only dst format

The guard above skips the entire reconfig when `pack_dst_format` matches. If two CBs can share a dst
format while differing in `pack_src_format`, the skip leaves `in_data_format` and `PCK_DEST_RD_CTRL`
(the 32b/int8/unsigned/round-10b-mantissa bits, all derived from `pack_src_format`) stale.

Whether that pair is reachable depends on whether a CB's pack src format can move independently of
its dst format, which this static read cannot settle. The overload already carries a
`TODO NC: Clean up as the part of tt-metal#34499`. Worth resolving before layering a second guard on
top of it — a tracker inside the LLK (P2) is immune to this question, which is another reason to
prefer it over tightening the metal-level condition.

### P3 — the generic `pack_tile` loop pays a per-tile address reprogram the multi-tile path avoids

For in-order output, consecutive tiles land at consecutive L1 addresses: `get_output_tile_address`
returns `fifo_wr_ptr + fifo_wr_tile_ptr - 1` and advances `fifo_wr_tile_ptr += fifo_page_size`. The
per-tile absolute recompute in `program_packer_destination` is therefore re-deriving something a
constant stride already describes.

The mechanism to avoid it **already exists and already ships**. `_llk_pack_mop_config_` Default mode
builds `MOP_OUTER_LOOP = num_faces * num_tiles`, so one `_llk_pack_` call can pack `num_tiles` tiles
with a single address program and a single MOP run.

**Wormhole has already built the full version of this** — see
[wormhole_pack_kernel_audit.md](wormhole_pack_kernel_audit.md) W3. There, `_llk_pack_` dispatches to
a `pack_multitile` helper automatically on `configured_num_tiles > 1`, the MOP advances the L1
address per tile with `ADDDMAREG` + `REG2FLOP` in `end_ops`, and the per-tile stride is precomputed
by `_llk_pack_output_addr_offset_words_`. Treat that as the design to port rather than something to
invent here; the Wormhole document also records the two hazards it comes with (a caller that ignores
it, and a static dispatch flag with no guard). SDPA streaming uses exactly this
(`compute_streaming.hpp`), and its threshold is architecture-tuned with a comment that reads as
measured:

```cpp
// BH benefits from blocked pack at width 4; WH keeps the threshold at 8 because
// width-4 blocked-pack reconfiguration costs more than it saves there.
#ifdef ARCH_BLACKHOLE
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 4;
```

**The candidate is `llk_matmul_pack`** in `llk_pack_tile_api.h`: it already loops over `ntiles`
consecutive in-order tiles and calls `_llk_pack_` once per tile, paying the full 8-instruction
sequence each time. It is the clearest case of a caller that has the block structure but does not use
the block mechanism.

**Constraints that bound this and must not be violated:**

- `_llk_pack_init_` asserts `num_tiles <= 4` for Float32 and `<= 8` for Float16/Float16_b — the dest
  register cannot hold more. Any blocked path must clamp to the same limits, and there is no assert
  for the other formats, so the caller carries that responsibility.
- Switching `num_tiles` requires re-running the MOP config. That is the cost the SDPA threshold is
  balancing, and it is why width-4 blocking wins on BH but not on WH. A blocked matmul pack should
  configure once per block shape, not per call.
- Out-of-order output (`out_of_order_output = true`) computes an arbitrary address per tile and
  cannot be blocked this way. Keep the per-tile path for it.

**Prize: unmeasured, and gated on step 0.** If `PACK_ISOLATE` shows the pack thread comfortably below
the unpack bound for matmul, this buys nothing end to end and should not be done.

### P4 — three in-tree mechanisms for "advance the L1 dest address"; mainline uses the most expensive

| Path | Per-advance cost | Where |
|---|---|---|
| mainline `program_packer_destination` | 3 x `SETDMAREG` + `STALLWAIT` + `WRCFG` + `DMANOP` | [cpack_common.h](../../tt_llk_blackhole/common/inc/cpack_common.h) |
| experimental fast-tilize | `ADDDMAREG` + `STALLWAIT` + `WRCFG` + `NOP`, inside a replay buffer, folded into MOP `end_ops` | [llk_pack_fast_tilize.h](../../tt_llk_blackhole/llk_lib/experimental/llk_pack_fast_tilize.h) |
| pack-untilize | **`CFGSHIFTMASK` + `NOP`** — cfg register `+=` scratch slot, no GPR round-trip, no THCON fence | [llk_pack_untilize.h](../../tt_llk_blackhole/llk_lib/llk_pack_untilize.h) |

For contrast, Wormhole's mainline `llk_pack.h` uses `ADDDMAREG` + `REG2FLOP` inside the MOP's
`end_ops` and needs no fence at all, because `REG2FLOP` executes on ThCon in program order after the
GPR write. The fence Blackhole needs here is a consequence of `WRCFG` living on a separate Config
Unit, which is exactly why the `CFGSHIFTMASK` form — which never round-trips through a GPR — is the
right target on this architecture.

The untilize form parks the stride in `SCRATCH_SEC2` once at init and then does
`THCON_SEC0_REG1_L1_Dest_addr += SCRATCH_SEC[CurrentThread]` from inside a replay buffer. Its own
comment claims it "saves ~3 cyc + 1 STALLWAIT per row" against the `ADDDMAREG` form, and notes it
mirrors an unpacker precedent (`llk_unpack_tilize.h`).

Nothing structural stops the Default path from parking `fifo_page_size` in the same scratch slot at
init and advancing the same way for the in-order case. This is the same shape as the FPU audit's F4:
an experimental file demonstrates the technique and mainline keeps the older one.

**Constraints:**

- `SCRATCH_SEC[CurrentThread]` is a single slot per thread. pack-untilize already claims SCRATCH_SEC2
  for its row stride, so a Default-mode user must not be live at the same time as an untilize op, or
  must use a different slot. Check this before assuming the slot is free.
- The relative advance only works for strictly sequential output. Any caller that jumps (out-of-order
  output, or a new CB) must fall back to the absolute program, so the absolute path stays.
- The absolute path also sets bit 31 of the address (`new_l1_addr = (1 << 31) | addr`) before the
  `WRCFG` and then rewrites the GPR without it. Whatever that bit means to the packer, a relative
  `+=` path does not reproduce it — understand it before replacing the write, not after.

### P5 — the per-tile `SETADCZW` may be dead

`_llk_pack_` ends every tile with `TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101)` to reset the Z
counters. In Default mode the tile-closing PACR uses `ADDR_MOD_1`, whose `z_src = {.incr = 0,
.clr = 1}` already clears ch0 Z, and no Default addr_mod ever increments `z_dst`.

The fast-tilize file states exactly this conclusion for its own path, and cites how it was
established:

```
// Counter state after every complete tile (confirmed via ISA docs + ttsim):
//   Z: 0 (ADDR_MOD_1 last PACR: z_src={clr:1})
```

One instruction per tile. Low value, but it is also nearly free to check, and the same reasoning has
already been done once in-tree. **Confirm against the ISA docs and ttsim before removing it** — the
Untilize mode shares this function and clears both `z_src` and `z_dst` via `ADDR_MOD_2`, so the two
modes need checking separately, and a stale Z counter corrupts the *next* tile's face addressing
rather than the current one, which makes it an unpleasant bug to trace.

## Invariants a Packer Change Must Preserve

The pack thread's correctness rests on a small number of properties that are easy to break without
producing an obvious failure. Any change above must leave all of these true.

1. **The `Last=1` PACR closes the tile.** It drains the packer, writes the tile header, and forces
   the next pack to re-sample `L1_Dest_addr`. Every MOP variant sets it via
   `set_last_outer_loop_instr`. A change that reorders or elides it breaks both the header and the
   no-drain-needed argument that `program_packer_destination` relies on.
2. **`L1_Dest_addr` is latched at PACR start.** This is why no packer drain is needed before
   reprogramming the address. Any new path that writes the address must still write it before the
   next PACR *issues*, not merely before it completes.
3. **`_llk_pack_dest_section_done_`'s `STALLWAIT(STALL_MATH, PACK)` is required.** It gates the
   `ZEROACC` that clears dest against the packer still reading it. It looks like the kind of stall
   these findings remove. It is not — leave it.
4. **`mutex_ADC` must match between `_llk_pack_init_` and `_llk_pack_`.** Guarding only one leaves
   unguarded SETADC issues. It is Default-mode only, because the other modes issue ADC instructions
   from inside the MOP where a scope guard cannot reach them.
5. **`_llk_packer_wait_for_math_done_` must name `STALL_SYNC` as well as `STALL_TDMA`.** The comment
   in `llk_pack_common.h` documents a real deadlock: with a TDMA-only mask the Sync-class `ATGETM` of
   a mutexed `_llk_pack_` slips past the unmet wait, takes `mutex::THREAD2_ADC`, and strands the
   `ATRELM` behind a head-of-line block while the packer waits on `MATH_PACK` holding the mutex the
   unpack thread needs. Do not "simplify" this mask.
6. **Inits own `SETADCXX`.** Every pack init programs the packer X counter itself
   (`FACE_C_DIM - 1` on Blackhole), and `reconfig_packer_data_format` also programs it because a
   reconfig is not always followed by an init. Keep both.
7. **Cross-thread cfg words stay byte-granular.** Word 71 is shared with the unpack thread's
   `Unp_LF8_4b_exp`; word 0 (`ALU_FORMAT_SPEC_REG`) Dstacc fields are shared with the math thread.
   Never widen an RMW on these to a whole-word `WRCFG_32b`.
8. **`_llk_pack_reduce_mask_config_` and `_..._clear_` are a pair.** The clear also restores
   `pack_reads_per_xy_plane` to 1, which `reconfig_packer_data_format` asserts on. Leaving the mask
   set poisons the next packer reconfig.

## Verification Plan

For any of the above, the FPU audit's method transfers directly:

1. **A/B the same commit** with and without the change (`git stash push -- <files>`), never against a
   remembered figure.
2. **Fresh `RUNNER_TEMP` per variant**, and wipe `perf_data/runs` and `perf_data/latest` between
   variants. Reusing them serves a stale ELF and reports the previous variant's cycles as if they
   were new.
3. **Carry flat controls.** Quote the pack numbers for every variant that should not have moved; a
   flat control is the evidence that nothing else changed.
4. **Read `mean(PACK_ISOLATE)` at `marker == TILE_LOOP`**, and quote `UNPACK_ISOLATE` beside it so a
   reader can see whether the saving is above or below the bound.
5. Correctness suites, minimum set per finding:
   - P1: `test_pack_dest_bank.py`, `test_pack_tiny_tile_block.py` (both sweep `l1_acc`), plus a case
     that **toggles** l1_acc within one run.
   - P2: `test_pack.py`, `test_experimental_reconfig_escape.py`,
     `test_tilize_transition_reconfig.py`, `test_sdpa_reinits.py`.
   - P3/P4/P5: `test_pack.py`, `test_matmul.py`, `test_pack_untilize.py`,
     `test_matmul_pack_untilize.py`, `test_pack_rows.py`, `test_dense_pack_untilize.py`.
   - Anything touching `cpack_common.h` reaches every pack kernel, so it deserves the same broad
     sweep `cmath_common.h` got in the FPU work.

## Where There Is No Room

Recording this so the effort is not spent twice.

- **No packer drain in the per-tile path.** F1's jackpot — four full drains per tile to flip one
  config bit — has no equivalent here. The single `STALLWAIT` in `program_packer_destination` is a
  THCON producer fence, and the file argues correctly that no `p_stall::PACK` drain is needed.
- **`_llk_pack_relu_config_` has already been through this.** It dropped its THCON wait when it moved
  from a whole-word write to `RMWCIB`, and the comment explains why the wait was no longer needed.
- **The debug probe is not a release cost.** `reconfig_packer_data_format` calls
  `is_pack_reads_per_xy_plane(1)` — a `noinline` helper with a 10-NOP settle — but only inside
  `LLK_ASSERT`, which compiles to `(void)sizeof(condition)` unless `ENABLE_LLK_ASSERT` is set.
- **`set_packer_config` and `reconfig_packer_data_format` agree.** Both write cfg word 70 as a whole
  word with the same fields zeroed, so the reconfig is not silently clearing state the hw-configure
  established. Checked because the pattern looks dangerous; it is consistent.

## Open Questions for Someone With the Hardware Definition

1. Is per-byte `RMWCIB` genuinely atomic across threads? Three separate places in this library
   (`configure_pack`, F2's INT8 tracker, P1 above) rest on it. One confirmation settles all of them.
2. What does bit 31 of the value written to `THCON_SEC0_REG1_L1_Dest_addr` mean, and why is it
   cleared from the GPR immediately after the `WRCFG`? P4 cannot proceed safely without this.
3. Is `SCRATCH_SEC2` free for a Default-mode L1 address stride, or does pack-untilize's claim on it
   make the two mutually exclusive?
4. Can a CB's `pack_src_format` move while its `pack_dst_format` stays fixed (P2b)?
