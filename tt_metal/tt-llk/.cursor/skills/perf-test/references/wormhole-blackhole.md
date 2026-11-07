# Wormhole and Blackhole perf specifics

Read with `perf-test`. Wormhole and Blackhole share the files in
`tests/python_tests/` and `tests/sources/`; `CHIP_ARCH` selects the
architecture at import and build time.

## Stand-ins

Helpers are in `tests/helpers/include/perf.h`:

- `_perf_unpack_stream_set_valid<set_a, set_b, PERF_RUN_TYPE>(count)` stands
  in for unpack in `MATH_ISOLATE`.
- `_perf_math_stream_clear_valid<clear_a, clear_b, PERF_RUN_TYPE>(count)`
  stands in for math in `UNPACK_ISOLATE`.

On Wormhole isolate runs these drive the handshakes through the MOP expander
(`perf_mock_mop`), so the stand-in does not set the measured value. Elsewhere
they fall back to the `_perf_*_loop_*` RISC-V loop themselves, so call the
`_stream_` names on both architectures. Several older kernels still call the
`_loop_` names directly; move them to `_stream_` when you touch them.

## Handshake counts

`_perf_src_handshake_iters_(tilize_en, num_tiles, num_faces)` in
`tests/helpers/include/llk_lib_unpack_wrappers.h` is the count shared by
unpack and math for unary unpack paths. Multiply it by `LOOP_FACTOR`.

- `unpack_A` posts one source dvalid per face.
- Tilize posts architecture-specific dvalids: one per tile on Blackhole, one
  per face on Wormhole (`_llk_unpack_tilize_num_dvalids_wrapper_`, which
  `_perf_src_handshake_iters_` calls; call the shared helper rather than the
  wrapper).
- `#1230` is `tenstorrent/budabackend#1230`: a hardware workaround that posts
  a SrcB dvalid next to SrcA. The Wormhole tilize MOP and the binary SFPU
  `NONE` path both do it, every face, including `dest_acc=No`. Keep the
  `#1230` reference on stand-in calls that reproduce it.
- Both tilize MOPs post SrcB, so a standalone tilize kernel's stand-ins are
  `<true, true>`. Datacopy-style unary kernels: unpack sets SrcB for
  `tilize_en || is_fp32_dest_acc_en`, and math clears `<true, true>`.
- Unpack-to-dest posts no source dvalids. Unpack writes Dest and hands it to
  math through a semaphore, so in `UNPACK_ISOLATE` math keeps its real
  unpack-to-dest path (the block helper, no source wait) as the stand-in, and
  in `MATH_ISOLATE` there is no math stage to measure: leave that run type
  out for unpack-to-dest-only sweeps, or report it as empty.

## Sweep and build

- The datacopy `tilize` argument exists only on Blackhole (a software
  workaround for a hardware bug), and never with a `Bfp8_b` input; see
  `get_valid_tilize_datacopy` in `test_eltwise_unary_datacopy.py`. Blackhole
  then builds variants that Wormhole skips: compile both.
- Run the coverage comparer once per architecture: the same file imports
  differently under each `CHIP_ARCH`.

## Reference implementations

- `tests/sources/eltwise_unary_datacopy_test.cpp` with
  `test_eltwise_unary_datacopy.py` and `perf_eltwise_unary_datacopy.py`.
- `tests/sources/pack_untilize_test.cpp` with `test_pack_untilize.py` and
  `perf_pack_untilize.py` (#63515).
- `tests/sources/pack_dest_bank_test.cpp` with `test_pack_dest_bank.py` and
  `perf_pack_dest_bank.py`, once #63744 merges. Until then main still has the
  split `pack_dest_bank_perf.cpp`; do not copy from main's version.

## Measured relations (pack dest bank, Wormhole)

These are examples of healthy numbers, not limits:

- `L1_TO_L1` is about 1.5–1.8× the slowest isolate for one-tile blocks and
  about 1.1–1.4× for 4–8 tile blocks; the gap is the cross-thread handshake.
- Repeated runs of one build: 0% spread.
- Unrelated idle-thread changes moved single configs ±8–12% between two
  levels (about 39 and 42 cycles per tile for pack) while the pack ELF was
  identical.
