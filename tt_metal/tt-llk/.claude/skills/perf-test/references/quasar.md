# Quasar perf specifics

Read with `perf-test`. Quasar tests live in `tests/python_tests/quasar/`
(`test_[op]_quasar.py`, `perf_[op]_quasar.py`) and kernels in
`tests/sources/quasar/[op]_quasar_test.cpp`. Quasar runs on the UMD
simulator through `run-test`.

## Python harness

The perf module imports the functional test function and calls it with perf
axes:

```python
from quasar.test_[op]_quasar import test_[op]_quasar as run_[op]_quasar

_PERF_AXES = dict(
    run_types=PERF_RUN_TYPES_QUASAR,          # helpers.llk_params
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],    # 32
    is_perf=[True],
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(**OP_SWEEP, **_PERF_AXES)
def test_perf_[op]_quasar(perf_report, <sweep args>, run_types, loop_factor, is_perf):
    run_[op]_quasar(<sweep args>, run_types=run_types, loop_factor=loop_factor,
                    is_perf=is_perf, perf_report=perf_report)
```

- The functional function takes keyword-only `is_perf=False` and
  `perf_report=None`, builds one `test_config_kwargs`, and calls
  `create_test_or_perf_config`, which adds `PERF_RUN_TYPE(L1_TO_L1)` to the
  functional templates itself.
- Many Quasar tests precompute a composite sweep list with a shared generator
  that takes `is_perf` (for example `PERF_PACK_COMBINATIONS` from
  `test_pack_quasar.py`, `PERF_DATACOPY_COMBINATIONS`). The perf module
  parametrizes on that list plus `_PERF_AXES`. That is the same contract: one
  generator, perf narrowing only inside it through the helpers below.
- 4-TRISC kernels use `PERF_RUN_TYPES_QUASAR_4_TRISC`, which adds
  `SFPU_ISOLATE`.
- Perf runs `DestSync.Half` only. The comparer still reports the missing
  `DestSync.Full` as a perf subset; that one difference is expected.
- Do not put `approx_mode` on `_PERF_AXES` when a family already constrains
  it: a keyword after `**SWEEP` that the dict already holds raises
  `TypeError`. Copy the sweep and replace the key (see `perf-coverage`).

## Tile shapes and dimensions

Perf shares the functional generator with `is_perf=True`, which narrows tile
shapes and dimensions through `helpers/param_config.py`. This is the one
documented perf narrowing on Quasar; do not add another.

**Tile shapes.** `select_perf_tile_sizes()` intersects the functional list
with `PERF_TILE_SIZES`:

| Class | Size | Why |
|---|---|---|
| 4-face | `(32, 32)` | Default FPU, pack and unpack path |
| 1-face | `(16, 16)` | MX-legal single-face path |
| 2-face narrow | `(32, 16)` | Different face grid and strides |
| Tiny | `(1, 32)` | Tiny-tile MOP (shared with 2/4/8×32) |

Skip sizes the functional set lacks (`pack_untilize` keeps `(32, 32)` and
`(1, 32)`). Keep the existing MX, unpack-to-dest, and 32-bit Dest filters.
Do not add all of `(1, 2, 4, 8, 16)×32`, and do not hardcode `(16, 16)` or
`(32, 32)` as the only perf tile.

**Input matrices.** Call `generate_perf_input_dimensions(dest_acc, dest_sync,
tile_shape)` for every selected tile. It emits Dest-full tall
`(max_tiles, 1)` and wide `(1, max_tiles)` in tile counts, then multiplies by
the tile shape. Do not match `PERF_INPUT_DIMENSIONS` element sizes such as
`[256, 32]` against a non-32×32 tile; they assume 32×32 and drop the wide
orientation. Do not copy the 50-way `generate_unary_input_dimensions` grid,
and skip Dest index and SFPU `tile_indices`: those are register offsets, not
distinct MOPs.

**Matmul.** Dest-full tall and wide output grids (`mt,nt` =
`(1, max_tiles)` and `(max_tiles, 1)`) × `kt={1, 4}` cover both `ct>=rt` and
`ct<rt` addr_mod branches and unpack-heavy versus math-heavy K. MX,
Float16_b, and Int8 inputs are LoFi-only; Float16 still sweeps LoFi–HiFi4.
`matmul_tile_dimensions()` in `quasar/test_matmul_quasar.py` sizes the grid
from `max_dest_tiles(dest_acc, dest_sync)`, which halves when `dest_acc=Yes`, so the
`dest_acc=No` grid is 8 tiles and `dest_acc=Yes` is 4: geometry is linked to
`dest_acc` (see `perf-parameter-impact`).

## Kernel

- Use the target Quasar correctness kernel for current LLK signatures and
  `TensorShape` APIs. Generic and older-architecture perf kernels use other
  APIs; do not copy their calls.
- Quasar Dest dvalid handshakes are configured per thread, and CFG state can
  persist between run types. `quasar_test_common.h` has the helpers:
  - the chains `set_up_fpu_to_pack_dest_dvalid_chain<thread>()`,
    `set_up_unpack_to_pack_...`, `set_up_unpack_to_fpu_to_pack_...`,
    `set_up_fpu_to_sfpu_to_pack_...`, `set_up_unpack_to_sfpu_to_pack_...`;
  - `set_up_zero_dest_dvalid_handshake_for_{unpack,math,sfpu,pack}()`, which
    clear that thread's wait mask so it runs without a partner.
- In `INIT`, set up a chain only when its producer and consumer are both
  active in this run type (`L1_TO_L1`). A thread that runs alone in an
  isolate clears its own wait mask instead (pack in `PACK_ISOLATE`, unpack
  with unpack-to-dest in `UNPACK_ISOLATE`, math in `MATH_ISOLATE` when its
  path waits on Dest). This handshake setup is the one allowed per-run-type
  difference in `INIT` (see `perf-test`, "INIT").
- `MATH_ISOLATE` runs math without `_llk_math_set_dvalid_` and without
  setting up a chain to the idle pack.
- Unpack-to-dest synchronizes through a semaphore. `L1_TO_L1` signals the
  active pack; `UNPACK_ISOLATE` does not; `MATH_ISOLATE` must not emit source
  dvalids with no math consumer. When math is compiled out, `MATH_ISOLATE`
  measures an empty run: its metric is missing or is only zone overhead.
  Exclude it from analysis and say so; it is not a math cost.

## Stand-ins

Quasar has no MOP stand-in. The `_perf_*_stream_*` helpers fall back to the
loop, so they are equivalent to the `_perf_unpack_loop_set_valid` and
`_perf_math_loop_clear_valid` names existing Quasar kernels call.
`quasar_test_common.h` adds `perf_unpack_set_srcb_once_then_srca_per_face`
and `perf_math_clear_srca_per_face_then_srcb_once` for kernels that post SrcB
once and SrcA per face.

Known counts:

- Scalar binary broadcast clears SrcAB once on the final outer-loop
  iteration: one handshake per tile, not per face.
- Row and column broadcast may handshake per face; inspect the MOP.
- Unary datacopy through math posts one source dvalid per tile. With 32-bit
  Dest (`is_fp32_dest_acc_en`) it uses ELWADD: unpack produces a real SrcA and
  a dummy SrcB, so the stand-ins use `<true, is_fp32_dest_acc_en>`. 32-bit
  data that goes unpack-to-dest posts no source dvalids.
- Check pulse counts in the code, not the comments: the variable-tile unpack
  MOP comment says "per face" but it pulses once per tile
  (`llk_unpack_unary_operand.h`).
- Binary SFPU stand-ins post per tile, and SrcB only when Dest is 32-bit:
  `<true, is_fp32_dest_acc_en>(LOOP_FACTOR * TILE_CNT)`. Wormhole and
  Blackhole post a zeroed SrcB every face instead (see
  `wormhole-blackhole.md`).
- A block kernel commonly needs `LOOP_FACTOR * BLOCK_RT_DIM * BLOCK_CT_DIM`.

## Coverage commands

From `tests/python_tests`:

```bash
python3 compare_test_and_perf.py --dir quasar --arch quasar
python3 compare_test_and_perf.py --arch quasar quasar/test_[op]_quasar.py quasar/perf_[op]_quasar.py
```

Composite splits name `list` as the input matrix and `tuple` as
`tile_dimensions`. Cross-architecture comparison is in `perf-coverage`.
