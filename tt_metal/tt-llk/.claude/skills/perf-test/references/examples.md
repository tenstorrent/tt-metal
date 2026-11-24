# Perf test examples

Worked cases behind the rules in [../SKILL.md](../SKILL.md). Each heading
names the rule it illustrates. They show what the rule caught; they are not
part of the contract. Paths are relative to `tt_metal/tt-llk/`.

## Fixed values become axes or config reads

A body constant becomes a single-value axis: `loop_factor=[32]`,
`dest_sync=[DestSync.Half]`. Kernel literals that had to move to config
reads include `DEST_SYNC`, `NUM_FACES`, `VECTOR_MODE` and `REUSE_DEST_TYPE`;
`DstSync::SyncHalf` and `num_faces = 4` were the typical hard-coded forms.

## Tile shape axis and pytest ids

On Wormhole and Blackhole the tile shape is swept as a `tile_dimensions`
axis (`[[32, 32], [16, 32]]`), as Quasar does, rather than as `num_faces` or
`tile_rows`. Faces and face rows come from `get_tile_params`
(`tests/python_tests/helpers/tile_constants.py`). Changing the axis renamed
pytest ids that were hard-coded in
`tests/hardware_free/kernels/test_gathering_config.py` and
`tests/python_tests/ttnop/FOCUS.md`.

## One constraints helper

`get_valid_math_fidelities` in `tests/python_tests/helpers/constraints.py`
is the arch-aware helper for the fidelity sweep; module-local wrappers that
only forwarded to it were deleted.

## Per-block init inside `TILE_LOOP`

In the bcast-col-custom kernel the functional loop repeated the math init
per block. Carried into `TILE_LOOP`, it tripled `MATH_ISOLATE`. Moving it to
`INIT` in both modes removed that cost.

## Runtime trip counts in the normal build

In bcast col custom, reading the tile geometry and block split from the
runtime struct made the normal-build `MATH_ISOLATE` per-tile cost about
twice the compile-time version. The speed-of-light build is unaffected.

## Report-only fields

`RELU_TYPE` and `SFPLOADMACRO` are `TemplateParameter`s whose
`convert_to_cpp` returns `""`: they add a CSV column and emit no code. A
report-only `Optional[Enum]` field added to a runtime parameter instead was
serialized to L1 and broke `test_pack`. The `NUM_GUARD_TILES` field is named
`num_guard_tiles`, not `count`, so its column is self-describing.

## Ambiguous column names

Generic names such as `capacity` or `dest_tiles` mean different things to
different operations in the shared warehouse; qualify them with the
operation.

## Attribute a move by experiment

A `PACK_ISOLATE` +5% was first blamed on buffer addresses. Reverting the
address change did not restore the number; reverting the runtime trip
counts did.
