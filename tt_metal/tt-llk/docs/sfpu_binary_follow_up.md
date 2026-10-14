# SFPU binary follow-up: four branches

Context for the follow-ups to [PR #57137](https://github.com/tenstorrent/tt-metal/pull/57137). That PR aligned Blackhole/Quasar eltwise binary SFPU functional and perf tests. Two issues were left open. They are addressed by four unstacked branches, each rebased onto `origin/main` at `3c202f9944d` and opened as its own pull request on 14 Oct 2026.

This note is not one of those changes. It only records what each branch does, what was deliberately left alone, and what the checks showed.

## Issues

| Issue | What it asks | What these branches do |
|---|---|---|
| [#58145](https://github.com/tenstorrent/tt-metal/issues/58145) | Compile Blackhole `SfpuDivInt32` and `SfpuDivInt32Floor` under `-DLLK_PROFILER` with the instruction stream Metal ships, then put them back in the perf sweep. | Closed by [#60515](https://github.com/tenstorrent/tt-metal/pull/60515) when that PR merges (`Fixes #58145`). |
| [#58855](https://github.com/tenstorrent/tt-metal/issues/58855) | Close the remaining eltwise-binary SFPU perf-coverage gaps: production mul, bf16 rounding, Float16, shallow depth, quant, `COPY_DEST`, mixed formats, reconfig, a `work_tiles` column, and the 16-bit `dest_acc` block-geometry confound. Also lists aligning `loop_factor`. | Split across [#60516](https://github.com/tenstorrent/tt-metal/pull/60516), [#60517](https://github.com/tenstorrent/tt-metal/pull/60517), and [#60518](https://github.com/tenstorrent/tt-metal/pull/60518). Each says `Relates to #58855` and does not close it. |
| [#58207](https://github.com/tenstorrent/tt-metal/pull/58207) | A kernel-speed change that still needs its own rebase onto `sfpu_binary_test.cpp`. | Not touched. |
| [#59833](https://github.com/tenstorrent/tt-metal/pull/59833) | The test-vs-perf and Blackhole-vs-Quasar comparer (`compare_test_and_perf.py`). | Already on main (`3f98c04b71`). Used to check the coverage branches. Not a parent of any of them. |

[#58855](https://github.com/tenstorrent/tt-metal/issues/58855) stays open after these three coverage PRs. Aligning `loop_factor` (Wormhole/Blackhole 16, Quasar 32) was declined: per-tile cost is not tile-count-invariant, and the comparer already lists `loop_factor` in `MEASUREMENT_AXES` and then ignores it. Three further mismatches are architecture properties already written into that issue, not test bugs:

- Int mul uses different kernels (Quasar `SfpuElwmulInt` to `BinaryOp::MUL`; Wormhole/Blackhole `SfpuMulInt32` to `BinaryOp::MUL_INT32`).
- 16-bit `unpack_to_dest` is on for Quasar and off for Wormhole/Blackhole.
- Quasar has no block-float, so `Bfp8_b` is Wormhole/Blackhole-only.

No new issue was filed for those.

## How the branches sit

Each branch is one commit, cut from the same pre-rebase base (`f0832760bf5`) and then rebased independently onto `3c202f9944d`. They are not stacked. B, C, and D can all edit `sfpu_operations.h`, the binary SFPU perf module, and the `perf_eltwise_binary_sfpu` schema. Each of B, C, and D bumps that schema from v5 to v6 on its own, with different new columns. After one of them merges, the other two need a rebase and another version bump.

`main` moved again after those rebases (this branch is `3fc4b9ddc06`). The four PR branches were not rebased a second time.

The comparer reads `@parametrize` names. It joins `math_op` to `mathop` only when the two sides disagree. It does not read CSV `aliases`. `mathop` and `op` still have to be the same parametrize name. It compares axis value sets, not combinations, so `approx_mode` Yes+No on both sides is identical even when only atan2 uses Yes. `loop_factor`, `iterations`, `run_types`, `is_perf`, and `implied_math_format` print and are then ignored. A Wormhole/Blackhole-only axis should stay one-sided. Do not add a fake Quasar axis to make a report match. Quasar paths must be `quasar/...`. Pairing a `test_` file with a `perf_` file under `--cross-arch` is an error.

Checks used `tt_metal/tt-llk/tests/.venv`, from `tt_metal/tt-llk/tests/python_tests`, with `CHIP_ARCH=blackhole`. This host is Blackhole, so Wormhole and Quasar runs are import and compile checks. Quasar imports need `PYTHONPATH=tt_metal/tt-llk/tools/python`. After #59833 merged, the script is the in-tree `compare_test_and_perf.py`.

## A — int32 division under the profiler

- Branch: `ndivnic/sfpu-div-int32-profiler` at `a250c32b806`
- PR: https://github.com/tenstorrent/tt-metal/pull/60515
- Closes #58145

Blackhole `calculate_div_int32_body` in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_div_int32_floor.h` kept `sign = a_orig ^ b_orig` live from the start of the body through the correction path. The SFPU has 8 LRegs. `-DLLK_PROFILER` (set by the perf harness in `helpers/test_config.py`) made that body fail with "too few lregs". Wormhole and Quasar already reload the sign from Dest at the end, which is why they fit. A profiler-only reload was rejected in #57137 because the measured kernel would not be the one Metal ships.

`sfpi::lreg_pressure _;` at the top of the Blackhole body, matching Wormhole, did compile under the profiler. `objdump -d` of `calculate_div_int32_body<true>` and `<false>` did not match current `main` without the flag (a commutative add swapped operands, and a `bltz`/`addi` pair reordered). That attempt was dropped.

The change that landed deletes the early live `sign` and recomputes it from Dest immediately before `v_if(sign < 0)`, the same way Quasar does. It is not wrapped in `#ifdef LLK_PROFILER`. The shipping stream gains two Dest loads and drops the live XOR. The profiler build matches that new stream inside the SFPU body. Both `SfpuDivInt32` and `SfpuDivInt32Floor` share the body.

`_PERF_EXCLUDED_MATHOPS` in `perf_eltwise_binary_sfpu.py` no longer lists those two ops, and the `#58145` TODO is gone. The pre-commit `fix-cstdint` pass also added `#include <cstdint>` and changed a `uint` to `std::uint32_t`.

Checked on the rebased tree:

- `pytest perf_eltwise_binary_sfpu.py -k DivInt32 --compile-producer`: both ops passed. That build is the profiler build. Two cases, `dest_acc=Yes`.
- `pytest test_eltwise_binary_sfpu.py -k DivInt32` on Blackhole: both functional int-uniform cases passed. Those stimuli are positive only.
- `test_compare_test_and_perf.py` and the perf schema header gate passed (20 tests). This branch does not bump the schema.

## B — cheap coverage

- Branch: `ndivnic/sfpu-perf-cheap-coverage` at `3fed234f694`
- PR: https://github.com/tenstorrent/tt-metal/pull/60516
- Relates to #58855. Does not close it.

Test-only. One schema bump of `perf_eltwise_binary_sfpu` from v5 to v6 covers the new columns (`dst_rounding`, `mul_kernel`) in `helpers/perf/test_schemas.py` and `helpers/perf/wide_schema.py`.

**Gap 1, MUL kernel.** `call_binary_sfpu_operation` used to send `BinaryOp::MUL` through generic `calculate_sfpu_binary`. Production `mul_binary_tile` uses `calculate_sfpu_binary_mul` on Wormhole/Blackhole and the generic kernel on Quasar. DIV was already split that way. A perf-only axis `SfpuMulImpl {Generic, Production}` defaults to `Generic`, so today's `SfpuElwmul` rows and the Wormhole/Blackhole to Quasar join stay on the generic kernel. A second Wormhole/Blackhole-only value, `Production`, calls `calculate_sfpu_binary_mul`. Quasar emits only `Generic` because it has no `_mul`. The functional golden path stays generic: `_mul` forces `0 * x = 0` on a bf16 dest. The new perf family is `test_perf_eltwise_binary_sfpu_mul_production`.

**Gap 2, rounding.** `calculate_sfpu_binary` already takes `DstRoundingMode`. Quasar's `bf16_rne` family already passes `NearestEven`. Wormhole/Blackhole `call_binary_sfpu_operation` did not. The parameter is threaded for ADD/SUB/RSUB only, with a perf axis `{Default, NearestEven}` on Float16_b and `dest_acc=No`. The family is named `bf16_rne` so it pairs with Quasar's family. `dst_rounding` is registered on the Wormhole/Blackhole schema; Quasar already had the column. This measures the mode #58855 asks about. It does not replace #58207's kernel change.

**Float16.** `DataFormat.Float16` is pruned from the Wormhole/Blackhole and Quasar binary SFPU perf sweeps. No ttnn dtype maps to it, Blackhole cannot run it at `dest_acc=No`, and Quasar's variant exists only at `dest_acc=No`. Functional Float16 stays.

**Shallow depth.** The float perf family gains `input_dimensions=[64, 32]` (2 tiles) for Float16_b and Float32, add and mul only, beside the 16/32-tile defaults. Production depth is 1, or 2 when sharded (`binary_ng_program_factory.cpp` around 1057–1080). Family name: `float_shallow`. `loop_factor` stays 16.

`call_binary_sfpu_operation` gained trailing `DstRoundingMode DST_ROUNDING = Default` and `bool PRODUCTION_MUL = false`. `SFPU_DST_ROUNDING_MODE` and `SFPU_MUL_IMPL` are always appended to the `sfpu_binary()` templates. Clang-format also collapsed unrelated `calculate_square` and `_relu_max_` call sites in `sfpu_operations.h`; those reflows are in the commit. The polygamma fix on main (`7d20b96553`) also edits that header; the rebase onto `3c202f9944d` applied cleanly.

Checked:

- Schema header gate passed.
- `--compile-producer` for `mul_production`, `bf16_rne`, and `float_shallow`: 17 passed, 3 skipped. The skips are the existing Float32 `dest_acc=No` rule.
- Comparer, all exit 0: `--arch blackhole`, `--arch wormhole`, `--arch quasar`, and `--cross-arch blackhole quasar` for both the perf pair and the functional pair.
- `mul_production` and `float_shallow` are unmatched on Quasar. `bf16_rne` pairs by family name. On the cross-arch report the Quasar side still names the op through `binary_op` (`'ADD'`, `'SUB'`) while Wormhole/Blackhole uses `mathop`, so those axes stay one-sided rather than a fake shared axis. `loop_factor` 16 vs 32 is ignored. `approx_mode` remains the pre-existing perf-only axis on same-arch functional vs perf.

## C — quant and COPY_DEST

- Branch: `ndivnic/sfpu-quant-copy-dest` at `8c9ff23861c`
- PR: https://github.com/tenstorrent/tt-metal/pull/60517
- Relates to #58855. Does not close it.

The kernels already existed. The Wormhole/Blackhole driver did not call them.

**Quant.** Quasar's `QUANT_SWEEP` / `_run_quant` is ported onto Wormhole/Blackhole functional and perf. Dispatch goes through `calculate_quant_int32` / `requant` / `dequant` in `ckernel_sfpu_quant.h` and the production entry points in `tt_metal/hw/inc/api/compute/quantization.h`. `BinaryOp` on Wormhole and Blackhole had no QUANT/REQUANT/DEQUANT members. They are appended at the end of the enum in both `ckernel_defs.h` files so existing values stay put:

```text
QUANT     = 47
REQUANT   = 48
DEQUANT   = 49
COPY_DEST = 50
```

Quasar's enum is not changed. Six rows: three ops times `sign_magnitude`. `sign_magnitude` is the shared axis with Quasar. Quasar still parametrizes `binary_op` strings (`'QUANT'`, `'REQUANT'`, `'DEQUANT'`) rather than `mathop`, so the cross-arch report shows `mathop` as Blackhole-only and `binary_op` as Quasar-only. That is expected. Do not invent a fake shared name.

**Golden, and why Blackhole differs.** Scale is uniform on `[0.01, 0.05)`, zero point is 3.0. Quant and requant are round-half-even of `a * scale + zp`, clamped to `[-127, 127]`. A negative value that rounds to 0 is sign-magnitude `-0` on Wormhole: `SFPSTORE` of INT32 keeps `0x80000000` for sign-magnitude, and the `INT32_2S_COMP` store maps it to `-1`. Blackhole's `FP32_TO_INT8` flush of that case is `+0`, and its sign-magnitude store is a no-op, so both encodings stay `+0`. The golden applies the negative-zero rewrite only when the arch is not Blackhole. Dequant is fp32 and is not rewritten. `ZERO_POINT` is a runtime struct field (`params.ZERO_POINT`); a bare `ZERO_POINT` does not compile in the math kernel. The inner quant iteration count stays 8 (`PER_FACE_ITERATIONS`). The wrapper's dest increment plus the kernel's `dest += 2` is the same pattern add already uses.

**COPY_DEST.** `ckernel_sfpu_copy_dest_values.h` exists on Wormhole/Blackhole, but `test_sfpu_domains.py` used to assert the enum member was absent. The enum value is added, dispatch follows Quasar's `copy_dest_value`, that guard is retired (`test_copy_dest_and_quant_are_declared_on_wh_bh`), and the op is on the float sweep, so it rides both functional and perf. A passing COPY_DEST row does not by itself prove the face walk: uncomputed faces stay as operand A, which is the correct copy-dest result.

`_PERF_EXCLUDED_MATHOPS` is still present on this branch. Removing the int32-div exclusion belongs only to branch A.

Checked:

- `test_copy_dest_and_quant_are_declared_on_wh_bh` passed for `tt_llk_wormhole_b0` and `tt_llk_blackhole`.
- Schema header gate passed (new columns `sign_magnitude`, `zero_point_bits`).
- Blackhole silicon: all six `test_eltwise_binary_sfpu_quant` rows passed, and `SfpuCopyDest` on Float16_b, broadcast none, `dest_acc=No` passed.
- Same five comparer commands as branch B, all exit 0. The quant family pairs with Quasar. COPY_DEST is inside the float `mathop` set on both sides; the sets still differ for other ops. `approx_mode` remains the pre-existing perf-only axis.

## D — structural measurement

- Branch: `ndivnic/sfpu-perf-structural` at `69f1b2e6d1e`
- PR: https://github.com/tenstorrent/tt-metal/pull/60518
- Relates to #58855. Does not close it.

These change what a row means, so they are not on branch B. This branch does not carry A's exclusion removal, B's mul/rounding/Float16/shallow edits, or C's quant and COPY_DEST.

**Mixed formats (Gap 4).** One directed Wormhole/Blackhole point: Float16_b x Bfp8_b to Float16_b, `SfpuElwadd`, `dest_acc=No`, one tile per operand (`input_dimensions=[32, 32]`). Quasar has no block-float, so there is no Quasar twin.

Both operands cannot share `buffer_A`. `Operand::operator[]` strides one tile size, Float16_b is 2048 bytes, and Bfp8_b is 1088. When `input_format != input_format_B`, each operand is generated on its own, A goes in `buffer_A`, B goes in `buffer_B`, and the golden is interleaved after quantizing the block-float operand. The same-format path is unchanged. `tile_count` for the kernel is the sum, so dest still holds the pair.

**Reconfig.** Production `eltwise_binary_sfpu_no_bcast.cpp` calls `reconfig_data_format_srca` between the two `copy_tile` loops. The test kernel cannot copy all of the left-hand side and then all of the right-hand side: the src fifo is 2-deep, and that order deadlocks. The harness equivalent, only when `formats.unpack_A_src != formats.unpack_B_src`, is per pair: unpack A, unpack-reconfig to B with `TILE_SIZE_UNPACK_B`, unpack B, reconfig back to A. Math does the same around the two datacopies (`_llk_math_reconfig_data_format_srca_`), including the math-isolate path so that measurement includes the reconfig. For this pair both unpack destinations are Float16_b, so the math reconfig is nearly a no-op; the unpacker reconfig (L1 format and tile size) is the real switch. Both are called. `if constexpr` drops the mixed branch from every same-format build, so those streams stay put. The LLK reconfig APIs are used, not the ttnn circular-buffer `reconfig_data_format_srca`.

**`work_tiles`.** One CSV column, `loop_factor * num_blocks * num_tiles_in_block`, on the Wormhole/Blackhole schema (`WORK_TILES` in `test_variant_parameters.py`, schema v5 to v6, `wide_schema.py`). It is emitted from all three runtime lists in `test_eltwise_binary_sfpu.py` (`sfpu_binary`, add-top-row, and broadcast). The postprocess divisor (`loop_factor * tile_cnt`) is not changed. The comparer does not see CSV columns. The header gate does.

**16-bit `dest_acc` confound.** `get_num_blocks_and_num_tiles_in_block` halves dest capacity when `dest_acc=Yes`, so Float16_b at `[256, 128]` goes from 4 blocks of 8 tiles to 8 blocks of 4, and the Yes-vs-No delta includes extra dest handshakes. The float perf family only (`pin_block_geometry` when `is_perf` on `test_eltwise_binary_sfpu_float`) computes the block shape with `DestAccumulation.Yes`, so Yes and No share 8x4. Float32 is already divisor 2 for both, because the input is 32-bit. Functional float geometry is not pinned. Branch B's 2-tile slice is not reused.

`loop_factor` stays 16 on Wormhole/Blackhole and 32 on Quasar.

Checked:

- Schema header gate passed.
- Blackhole silicon: `test_eltwise_binary_sfpu_mixed_format` passed (the functional twin of the perf point; perf does not check a golden).
- Same five comparer commands, all exit 0. On Blackhole, functional vs perf, `mixed_format` matches on `formats`, `mathop`, and `dest_acc` (1 variant each). `approx_mode` is the pre-existing perf-only axis. Cross-arch, `mixed_format` is unmatched on Quasar. `loop_factor` 16 vs 32 is ignored.

## Checks that are shared

On the first rebased tree (the division branch, which already contained main's comparer):

```text
pytest test_compare_test_and_perf.py \
  test_perf_header_gate.py::test_perf_test_schemas_match \
  test_perf_header_gate.py::test_no_published_column_without_a_perf_test
```

20 passed. The schema gate was run again on B, C, and D because each publishes different columns. B, C, and D each ran:

```text
python compare_test_and_perf.py --arch blackhole \
  test_eltwise_binary_sfpu.py perf_eltwise_binary_sfpu.py
python compare_test_and_perf.py --arch wormhole \
  test_eltwise_binary_sfpu.py perf_eltwise_binary_sfpu.py
python compare_test_and_perf.py --arch quasar \
  quasar/test_eltwise_binary_sfpu_quasar.py \
  quasar/perf_eltwise_binary_sfpu_quasar.py
python compare_test_and_perf.py --cross-arch blackhole quasar \
  perf_eltwise_binary_sfpu.py quasar/perf_eltwise_binary_sfpu_quasar.py
python compare_test_and_perf.py --cross-arch blackhole quasar \
  test_eltwise_binary_sfpu.py quasar/test_eltwise_binary_sfpu_quasar.py
```

All exited 0. Exit 1 from a directory sweep (`--cross-arch blackhole quasar --kind perf` or `--kind func`) means the tool found coverage differences across the whole tree, not that it crashed. A full directory sweep is not part of these four PRs' results.

## If two coverage PRs merge

B, C, and D each start from the same main and each set `perf_eltwise_binary_sfpu` to version 6 with a different column set. Whichever merges second must rebase, keep both column sets, and bump the version again. B and C both edit `sfpu_operations.h`. D edits `sfpu_binary_test.cpp` and `test_eltwise_binary_sfpu.py`, which B also edits. D does not need B's 2-tile slice; it pins block geometry itself.
