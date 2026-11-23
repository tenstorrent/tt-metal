# HAL edge cases found while porting eltwise binary (Blackhole)

Each item: what happened, how it showed up, where to look, what to do. Measured on the eltwise-binary perf sweep
(1300 configs) and six regression-guard perf tests on Blackhole p300c.

## Correctness

1. **`hal::dst::Zero::address_mode` is applied in Face/Half/All mode too.** The HAL docstring says it only matters for
   SingleRow; the hardware applies the ADDR_MOD in every mode. Leaving it at the default 0 (often a "step" mode)
   moved the Dest counter by 8 per ZEROACC and broke every ELWMUL dest-reuse case. *Always pass the ADDR_MOD
   production passes* (`hal/dst.h`).
2. **Auto TTSync is sticky per thread and breaks code that still uses the two-context semaphore handshake.**
   Enabling it in the shared `_llk_unpack_hw_configure_` hung `test_sdpa_custom_mm_mask_extent_restore`. Enable it
   only in the init of the op that relies on it (`_llk_unpack_AB_init_`) and disable it in that op's uninit. Before
   relying on Auto TTSync, issue a SETC16 of `CFG_STATE_ID` (Blackhole erratum, SETC16.md: until then it tracks the
   wrong config bank; without it 1910/5704 eltwise tests failed).
3. **Mixing production and HAL headers in one thread breaks the build.** Production headers include their siblings
   with quotes, which resolve next to the including file. That is why `tests/hal_llk` mirrors the whole `llk_lib`
   tree: never shadow a subset.
4. **Shared build trees.** Concurrent compiles into `/tmp/tt-llk-build` deleted each other's ELFs ("variant did not
   compile", "cannot open output file"). The chip queue gives every job a private `RUNNER_TEMP`; do the same for
   any manual build.
5. **Known non-port failures** (check `--impl tt` before blaming a port): `test_matmul_custom_compressed` hangs on
   production; `test_hadamard_h128[LoFi, normalize:False, 8 tiles]` fails about 1 run in 3 on production;
   `test_unpack_a_bcast_eltwise` runs no cases on Blackhole.

## Code size and RISC cycles (the HAL's C++ shape changes what the compiler can do)

6. **Redundant masking in runtime thread-CFG writes.** `cfg::write<TensixCfgUnit, F, S>(value)` for a thread field
   masked `value` with `& 0xFFFF` after already asserting its range. The extra `zext.h` also stopped GCC from turning
   the per-tile dest address into an increment: ~3 RISC instructions per tile, +1.7 to +2.4 cycles/tile on
   dest-reuse and HiFi ELWMUL. Fixed in `hal/cfg/access.h` (mask removed). `detail::write_thread_word` still masks;
   use `cfg::write_thread_word` (no mask) for per-tile packed words.
7. **Runtime SETDMAREG words are re-encoded instead of hoisted.** When the compiler fully unrolls a small
   compile-time loop of `_llk_pack_` calls (custom_mm `CT_DIM == 7`), production keeps the 21 encoded SETDMAREG words
   in registers; the HAL `scalar::set_low/set_high` path re-derives the high half and spills each tile (+24% pack
   isolate on those configs; invisible at 1-4 tiles, where everything fits, and at 16+, where nothing is unrolled).
   Compute address halves once; for loop-invariant addresses build the instruction words with the `SetLow/SetHigh`
   `operation()` encoders so the compiler can hoist plain integers.
8. **`cfg::write<Access::MMIO>(set<…>…)` on a partly covered word is a load + store.** An uncached MMIO load costs a
   RISC round trip. Production always stores whole words; compose them and use the array form.
9. **MMIO read-back instead of STALLWAIT TRISC_CFG.** An early port polled the last base-address store until it read
   back, because "the HAL has no stall for pending RISC CFG writes". It has: `StallCondition::RiscvAccessProcessed`
   (0x400) is TRISC_CFG. The poll cost ~6 cycles/tile on a RISC-bound unpack loop.
10. **Over-strict stall conditions.** Waiting for PackerIdle before rewriting the L1 destination (production waits
    only for the GPR write, ScalarIdle: the PACR latches the address at start) serialised tiles (+17% pack). Holding
    Pack|Thcon on the math-done SEMWAIT (production holds Tdma|Sync) blocked the next tile's address setup.
11. **Compile-time folding changes issue timing.** A config write with constant operands becomes an immediate
    instruction instead of an instruction-buffer push. Same instruction, but the configure then finishes earlier;
    in kernels whose tile loop starts right after, ~4-5 cycles of a fixed ~20-cycle per-run cost traced to this one
    RMWCIB. The rest of that cost depends on configure timing in ways no single instruction explains (only the
    complete production configure removes it). Worth ~0.2-0.5% on short kernels; measure the whole kernel (KERNEL
    zone), not just TILE_LOOP, before calling it a regression.
12. **Noinline helpers move with unrelated code size.** `_apply_src_zero_flag_` is `noinline` and lives after
    `run_kernel`; its address follows the tile-loop code size, and init timing moved ±6 cycles with it. Code-shift
    tests (`LLK_CODE_SHIFT=4/8/16`) separate placement effects from real ones.

## Where the gains came from (eltwise binary, end to end -22.9%)

- Unpack per-tile path: two address stores, one counter reset and the MOP per tile, with Auto TTSync ordering the
  stores (UNPACK_ISOLATE -35%). Fusing adjacent Tensix pushes (`SETADCZW` next to STALLWAIT/MOP/SEMGET), a branchless
  context select and hiding the semaphore load latency gave the first -5.6%.
- Dest-reuse math: MOVD2A waits only for SrcA valid (MOVD2A.md; MOVD2B keeps the drain), and the move prologue plays
  from a replay buffer as the MOP start op.
- Rejected: carrying the LoFi source release on the last ELW (some configs +3.7%).
- Code size: whole-word config writes and immediates make unpack 3-6%, pack 1.4-2.5% smaller.

## How to investigate a small regression

Pair builds by `build.h`; diff `run_kernel` with registers masked; split the cost into INIT / TILE_LOOP / outside
(KERNEL - INIT - TILE_LOOP); check per-tile scaling (correlation with tile_cnt × loop_factor); run code-shift
controls; build hybrids (HAL for one thread only, via an include dir that shadows only that thread's headers, plus
the rest of the tree) to find the thread; swap single blocks with production code to find the block.
