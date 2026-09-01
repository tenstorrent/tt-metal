# Blackhole FPU Math Kernel Audit

Cycle-level audit of the Blackhole FPU (matrix-unit) math LLKs: where the math thread has headroom,
where it is already at the hardware limit, and what was changed as a result.

Companion document: [blackhole_fpu_math_kernel_measurements.md](blackhole_fpu_math_kernel_measurements.md)
holds the before/after numbers, the correctness evidence and the reproduction commands.

## Quick Links

- Kernels audited: [tt_llk_blackhole/llk_lib/](../../tt_llk_blackhole/llk_lib/)
  - [llk_math_matmul.h](../../tt_llk_blackhole/llk_lib/llk_math_matmul.h)
  - [llk_math_eltwise_binary.h](../../tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h)
  - [llk_math_reduce.h](../../tt_llk_blackhole/llk_lib/llk_math_reduce.h)
  - [llk_math_transpose_dest.h](../../tt_llk_blackhole/llk_lib/llk_math_transpose_dest.h)
  - [llk_math_eltwise_unary_datacopy.h](../../tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h)
  - [llk_math_common.h](../../tt_llk_blackhole/llk_lib/llk_math_common.h)
- Shared math state: [tt_llk_blackhole/common/inc/cmath_common.h](../../tt_llk_blackhole/common/inc/cmath_common.h)
- MOP / replay infrastructure: [tt_llk_blackhole/common/inc/ckernel_template.h](../../tt_llk_blackhole/common/inc/ckernel_template.h)
- Perf harnesses: [tests/python_tests/perf_reduce.py](../../tests/python_tests/perf_reduce.py),
  [perf_matmul.py](../../tests/python_tests/perf_matmul.py),
  [perf_eltwise_binary.py](../../tests/python_tests/perf_eltwise_binary.py)
- Counter interface: [docs/performance_counters/performance_counters.md](../performance_counters/performance_counters.md)

## Scope and Method

Blackhole only, math thread (T1) only. Unpacker and packer are taken as given: they set the bar a
math-side optimization has to beat, they are not themselves targets.

All figures are `mean(MATH_ISOLATE)` at `marker == "TILE_LOOP"` — cycles per tile spent on the math
thread with unpack and pack stubbed out — measured on a **Blackhole p300a**, `Float16_b -> Float16_b`,
`dest_acc` off, speed-of-light builds, `LOOP_FACTOR=64 x TILE_CNT=16`. Baseline commit: `d53cb57e206`.

Each kernel's own `mean(UNPACK_ISOLATE)` is quoted alongside as the bound. Below that bound the math
thread is not the limiter and a math-side saving buys nothing end to end; above it, math is what the
pipeline waits on.

> A p300a is not the p100a used for earlier SFPU work. Do not compare absolute numbers across the two
> boards.

## Measured Baseline

| Kernel | LoFi | HiFi4 | Unpack bound | Math headroom |
|---|---:|---:|---:|---|
| `matmul` 32x32x32 | 19.20 | 68.33 | 33.25 | none — at HW limit |
| `eltwise` add / sub | 16.62 | n/a | 37.91 | none — at datapath limit |
| `eltwise` mul | 16.62 | 82.52 | 37.91 | none — at datapath limit |
| `reduce` col · max | 16.07 | 16.07 | 37.28 | below bound |
| `reduce` col · sum / avg | 16.10 | 83.07 | 37.28 | HiFi only (F5) |
| **`reduce` row · max** | **97.94** | **97.94** | **37.28** | **2.6x over bound (F1)** |
| `reduce` row · sum / avg | 12.08 | 49.07 | 37.28 | near HW limit |
| `reduce` scalar · max | 41.07 | 41.07 | 48.09 | below bound |
| `reduce` scalar · sum / avg | 41.10 | 130.07 | 48.09 | HiFi only (F5) |

MAX pooling is GMPOOL-only and ignores fidelity, so its two columns are the same measurement.

## Per-Instruction Costs Solved From the Sweep

LoFi and HiFi4 differ only in fidelity-phase count, so the pair of measurements for one kernel solves
for instruction cost and fixed per-tile overhead. Every estimate in the findings is built on these.

| Quantity | Value | Derivation |
|---|---:|---|
| MVMUL | 1.02 cyc | matmul `16m + O = 19.20`, `64m + O = 68.33` |
| Matmul fixed per-tile overhead | 2.82 cyc | same solve — SETC16 dest addr + MOP run + SETRWC |
| ELWADD / ELWMUL | 2.08 cyc | 8 instructions per tile at 16.62; 1024 datums / 16.62 = 62 datums/cyc |
| GMPOOL (16x16 face) | 4.0 cyc | reduce col max = 4 GMPOOL = 16.07 |
| GAPOOL (16x16 face) | 4.0 cyc LoFi, 5.2 cyc HiFi4 | reduce col sum = 4 / 16 GAPOOL = 16.10 / 83.07 |
| MVMUL in reduce row sum | 1.51 cyc | 8 / 32 MVMUL = 12.08 / 49.07 |
| One cfg write + pipe drain | **2.8–10.0 cyc** | measured at both ends, see below |

The last row is the load-bearing one and it is **not a constant**. A `STALLWAIT(STALL_CFG, ...)` +
`RMWCIB` pair costs what it takes to drain the math pipe, so it is cheap when the pipe is shallow and
expensive when it is full:

- **9.97 cyc** in reduce row max, where the write sits between two dense math phases
  (39.88 cycles recovered / 4 writes removed).
- **2.76 cyc** in an eltwise-add loop, where the surrounding math is 8 instructions deep.

Budget cfg writes by where they sit, not by counting them.

## Findings

Ranked by cycles recoverable. Status records what was actually done.

### F1 — REDUCE_ROW MAX paid four pipe drains per tile to flip one config bit — **IMPLEMENTED, −39.88 cyc/tile (−40.7%)**

[`reduce_row_perform_transpose()`](../../tt_llk_blackhole/llk_lib/llk_math_reduce.h) opened by asserting
PRESERVE on the Src zero-substitution flag and closed by restoring the operand-driven DEFAULT. The
value therefore alternated `1 -> 0 -> 1 -> 0`, so the skip-if-set guard in
`_configure_src_zero_flag_` never hit: every call reached `_apply_src_zero_flag_`, which is a
`STALLWAIT(STALL_CFG, MATH | WAIT_SFPU)` full math+SFPU drain plus an `RMWCIB`, out-of-line. The
transpose runs once per face row, so a 32x32 tile paid four drains — **40 of the kernel's 98
cycles/tile**.

The flag's readers are the mov phase (`MOVD2B` / `TRNSPSRCB` / `ELWADD`), which genuinely needs
PRESERVE or a datum whose low byte is zero is flushed mid-reduction. GMPOOL and GAPOOL are not among
the readers listed in `cmath_common.h`, so the pool phase does not care which value is live.

**Fix:** assert PRESERVE once in `_llk_math_reduce_init_` for the `REDUCE_ROW` + `MAX` specialization
and hold it for the whole op; every other reduce path keeps the operand-driven DEFAULT. The execute
path now writes the flag zero times. `_llk_math_reduce_uninit_` — previously empty — restores the
baseline, so the value is paired with the init rather than leaked to whatever op runs next; that costs
one cfg write per op, not per tile. (`_llk_math_transpose_dest_` leaks PRESERVE and relies on the next
init to clean up; `reduce_uninit` is a contract-documented part of the Compute API, so pairing is both
available and cheaper to reason about.)

**Result:** 97.94 -> 58.06 cycles/tile. `test_reduce.py` 3528/3528 pass, which also settles the
open question empirically: the low-byte-zero misdetection does not apply to GMPOOL, or a MAX reduce of
ordinary bf16 values such as `0x4400` (768.0) would have been corrupted across the whole sweep.

**Residual exposure:** if the reader list is incomplete and GMPOOL does consume the flag *correctly*
(flushing only true denormals rather than misdetecting), then MAX-pooling a bf16 denormal against zero
now returns the denormal instead of zero. No test can distinguish this; it is called out here for a
reviewer with the HW definition to confirm.

### F2 — every format reconfig spent a drain on a bit that usually does not move — **IMPLEMENTED, −2.76 cyc per reconfig (−39.6%)**

`_llk_math_reconfig_data_format_srca_ / _srcb_ / _` each issued `STALLWAIT(STALL_CFG, MATH)` + an
`RMWCIB` for `ALU_ACC_CTRL_INT8_math_enabled`, then called `_configure_default_zero_flag_state_()`,
which — because the cached operand formats had just changed — could issue a second drain. The INT8 bit
had no skip-if-unchanged guard even though the zero-substitution flag in the same file has had one
since it was refactored to track its physical value.

The asymmetry mattered because the bit's *driver* is coarser than its *value*: callers reconfig on any
operand format change, but the bit only moves across an Int8/UInt8/Int32 boundary. A `bf16 -> fp32`
reconfig rewrote `0 -> 0` and paid a full MATH drain to do it.

**Fix:** mirror the zero-flag pattern — `int8_math_enabled_hw` tracker, out-of-line
`_apply_int8_math_enabled_`, inlined `_configure_int8_math_enabled_` guard, and a
`_seed_int8_math_enabled_state_` for `_llk_math_hw_configure_`, which writes the bit directly (batched
with the other `ALU_ACC_CTRL` fields under one STALLWAIT). Safe to cache: the math thread is the sole
writer — the unpacker's `ALU_FORMAT_SPEC_REG0_SrcA_ADDR32` RMW masks
(`alu_format_mask | alu_stoch_rnd_mask` in `cunpack_common.h`) exclude this bit.

**Result:** measured with a temporary probe (one same-format reconfig per tile in the eltwise-add
MATH_ISOLATE loop): a reconfig cost 6.97 cyc, now costs 4.21 cyc. In a context with a deeper math pipe
the saving is proportionally larger — see the drain-cost note above.

### F6 — matmul recorded half a replay buffer for one legal tiny-tile shape — **IMPLEMENTED (correctness, no perf change)**

In `matmul_configure_mop`, `replay_buf_len` collapsed all four narrow-shape flags into
`partial_face ? 4 : 8`, but the recording lambda emits 8 MVMULs in the `is_in1_32x16` and plain
`is_in1_16x32` branches regardless of `partial_face`. `lltt::record` captures exactly `replay_buf_len`
instructions; anything beyond executes immediately, outside the record window, against sources the MOP
has not validated — and the MOP then replays half the sequence.

`partial_face` is `in0_tile_r_dim < FACE_R_DIM`. Take in0 = `[8, 16]`, in1 = `[16, 32]`: only
`is_in1_16x32` is set, so the length was 4 while 8 instructions were emitted. The shape is
dimensionally valid (in0 cols = in1 rows = 16) and is **not** covered by
`generate_matmul_tiny_tiles_combinations`, which pins `in0_tile_columns = 32` and `in1_tile_rows = 32`.
The `is_in1_32x16 && partial_face` variant needs in0 cols = 16 with in1 rows = 32, so that one is
dimensionally unreachable.

**Fix:** the length expression now mirrors the lambda's branch structure exactly. Values are unchanged
for every shape the suite exercises; `test_matmul.py` 6784/6784 pass.

Its address mods also fall through to the full-32x32 defaults for this shape, which the file's own
`LLK_ASSERT` warns about for the 16x16-by-16x16 case. Not addressed here — flagged for whoever adds
coverage for these shapes.

### F3 — dest-reuse eltwise binary issues ~28 instructions and 4 drains to drive 8 ELWMULs — **NOT IMPLEMENTED (no harness)**

The standard path is 3 issued instructions per tile and measures 16.62 cycles. The dest-reuse path, per
face, runs `move_d2a_fixed_face()` — a `STALLWAIT(STALL_MATH, MATH | SRCA_VLD)` drain plus 4 MOVD2A —
then a `TT_ZEROACC` whose operand comes from a runtime
`get_dest_index_in_faces(local_tile, face_offset + n)`, then a MOP run. Four faces means four drains,
sixteen MOVD2As and four register-form ZEROACCs against a MOP body of two ELWMULs.

Because the ZEROACC address is absolute and computed on the RISC-V, the per-face sequence cannot be
captured into the replay buffer as it stands; rebasing it on the dest counter would make the whole
sequence static and replayable — one issued instruction per face instead of seven.

Also worth confirming rather than assuming: at LoFi the dest-reuse MOP builds its ELWMUL with
`acc_to_dest = 0`, so the multiply overwrites the dest rows the ZEROACC just cleared. If the clear
only exists for the FP32 two-half write path, it is dead work in the 16-bit LoFi case.

**Why not done:** `perf_eltwise_binary.py` has no `binary_reuse_dest` axis, so there is nothing to
measure against. Rewriting an uninstrumented path on an estimate is how regressions ship. Add the axis
first; on the drain costs measured here this is the most likely place for the next large FPU win.

### F5 — REDUCE_COL / REDUCE_SCALAR sum use GAPOOL where the operands are already laid out for MVMUL — **NOT IMPLEMENTED (needs an unpacker change)**

At identical fidelity and tile shape, REDUCE_ROW SUM runs 32 MVMULs in 49.07 cycles while REDUCE_COL
SUM runs 16 GAPOOLs in 83.07. GAPOOL does twice the work per instruction but costs 3.4x as much, so the
MVMUL formulation wins by ~1.7x. The row path already exploits this; the column path does not.

The operand layout is nearly right already. `_llk_unpack_AB_reduce_init_` documents that for REDUCE_COL
and REDUCE_SCALAR, unpacker 0 fills SrcA with the data and unpacker 1 fills SrcB with one row of
scaler. MVMUL computes `D[i,j] = sum_k B[i,k] * A[k,j]`, so a constant scaler row in SrcB row 0 gives
`D[0,j] = s * sum_k A[k,j]` — the scaled column sum, one instruction per face, with MVMUL's native
dest accumulation folding the two face rows together.

**Why not done:** MVMUL writes eight dest rows, using SrcB rows 0-7. Only row 0 is the wanted result;
rows 1-7 would take `sum_k B[i,k] * A[k,j]` for `i = 1..7`, which is garbage unless SrcB rows 1-15 are
zeroed. For REDUCE_COL (non-tiny, non-scalar) `_llk_unpack_AB_reduce_mop_config_` does not emit the
`UNPACR_NOP ... CLR_SRC_0` clear, so those rows hold stale data. Making this correct requires the
unpacker to clear SrcB — outside the math-thread scope of this audit, and writing large garbage
(possibly inf/nan) into dest rows 1-7 where the old path left stale zeros is not a safe unilateral
change. Estimated prize if the unpacker cooperates: col sum 83 -> ~30, scalar sum 130 -> ~75, both at
HiFi3/HiFi4 only. Worth a follow-up with the unpacker owner.

### F4 — matmul's MOP outer loop is hard-coded to 1 while the block loop re-addresses dest per tile — **NOT IMPLEMENTED (low yield, high blast radius)**

`_llk_math_matmul_` calls `set_dst_write_addr` inside the `t_dim x rut_dim` loop, so every output tile
pays a SETC16 with RISC-V-computed operand, a MOP run and a conditional SETRWC — the 2.82 cycles/tile
solved above. Meanwhile `matmul_configure_mop` constructs
`ckernel_template tmp(1 /* outer loop */, inner_loops, ...)`: the hardware's second loop level sits
idle while the reuse dimension is unrolled on the RISC-V.

[`experimental/llk_math_custom_mm.h`](../../tt_llk_blackhole/llk_lib/experimental/llk_math_custom_mm.h)
already demonstrates the fix — one `set_dst_write_addr` for the whole block, dest advanced between
tiles by `ADDR_MOD_2`'s `.dest = {.incr = 32 or 64, .cr = 1}` (the increment field is wide enough; the
same file uses `1024 - 8`).

**Why not done:** 2.82 cycles is 15% of a LoFi tile but LoFi matmul is unpack-bound (19.20 vs 33.25),
so it buys nothing end to end; at HiFi4 it is 4%. `custom_mm` earns it by restricting itself to
`rt_dim = 1`, LoFi, no transpose, no throttling. Generalizing to mainline matmul means re-deriving dest
addressing across four narrow shapes x transpose x four fidelities x five throttle levels x
reuse-A/reuse-B — a large change to the library's most critical op for a few percent on one
configuration.

### F7 — smaller items from the same read

- **`reduce_row_advance_dest` cannot be collapsed.** The four `SETRWC(CR_D, 8)` calls look like an
  obvious fold into one instruction, but `rwc_d` is a 4-bit field in *both* `TT_OP_SETRWC` and
  `TT_OP_INCRWC` (max 15), so a single-instruction `+32` does not exist. The repetition is ISA-forced.
  An addr-mod on the preceding math instruction could carry a wider dest increment, but the advance has
  to happen after the transpose, where no math instruction is pending. Left alone. *(This item was
  wrong in the first draft of this audit; corrected after checking the encodings.)*
- **The 32-bit unpack-to-dest broadcast paths** in `llk_math_eltwise_unary_datacopy.h` (roughly lines
  109-268) issue 40-60 register-form MOVD2B / MOVB2D instructions with a runtime
  `tile_base = dst_index * 64` immediate, interleaved with several `cfg_reg_rmw_tensix` format toggles
  per tile. The absolute addressing is what blocks replay capture; rebasing on the dest counter would
  make the sequence static. No perf harness covers it.
- **`_llk_math_reduce_init_` and `_llk_math_eltwise_binary_standard_init_`** write
  `CLR_DVALID_SrcA_Disable` and never restore it — their uninit functions are empty and documented as
  "no state to restore". State hygiene, not cycles.
- **`transpose_dest_32b`** interleaves `cfg_reg_rmw_tensix` writes with in-flight MOV instructions with
  no STALLWAIT between them. Worth a second look for the same reason F2 exists, though the file itself
  notes the API is not widely used (tt-llk#290).

## Where There Is No Room, and Why

The useful half of the answer: three of the four kernel families are done, and effort spent there is
wasted.

- **Matmul.** MVMUL retires at 1.02 cycles/instruction, and 16 (LoFi) or 64 (HiFi4) MVMULs is the
  algorithmic minimum for a 32x32x32 tile. The MOP and replay structure already reduce a whole tile to
  three issued instructions. HiFi4 is math-bound at 68.33 vs 33.25, but that is fidelity phases doing
  required arithmetic, not overhead.
- **Eltwise binary, standard path.** 8 ELW ops per tile at 2.08 cycles is 1024 datums in 16.62 cycles —
  62 datums/cycle, the FPU's eltwise datapath width. The MOP already folds faces into the outer loop
  and 8-row groups into the inner one. Both LoFi variants sit at less than half the unpack cost.
- **Reduce, everything except row MAX.** At LoFi all five other variants are 12-41 cycles against
  bounds of 37 and 48. Their HiFi4 costs are fidelity phases, addressed by F5 rather than by
  restructuring.
- **Throttled matmul.** `run_throttled_sequence<1..5>` exists to insert NOPs and cap throughput. It is
  slow on purpose; excluded from this audit.

## Instrumentation Gaps

- **Closed:** `sources/reduce_perf.cpp` hard-coded `MathFidelity::HiFi4`, so every number from
  `perf_reduce.py` was HiFi4 and the LoFi picture — which differs by up to 5x for the SUM variants —
  was invisible to CI. It now takes the injected `MATH_FIDELITY` like `sources/reduce_test.cpp` does,
  and `perf_reduce.py` sweeps `[LoFi, HiFi4]` for SUM/AVG on one format pair (MAX ignores fidelity, so
  it stays at one point). Cost: 144 -> 150 collected tests.
- **Open:** `perf_transpose_dest.py` declares `run_types=[PerfRunType.L1_TO_L1]` only, so the math
  thread is never isolated for either transpose_dest path.
- **Open:** no perf coverage for eltwise binary with dest reuse (F3), the datacopy 32-bit broadcast
  paths (F7), or tiny-tile reduce shapes.
