# Perf and test axis naming

WH/BH and Quasar tests and perf modules are compared by name (`tests/python_tests/compare_test_and_perf.py`),
and the perf CSV columns are derived from the same names. Keeping names identical across architectures lets
the comparer join axes directly instead of translating them.

## Rules for `@parametrize` axes

- One argname per axis, the same on every architecture: `formats`, `dest_acc`, `dest_sync`,
  `input_dimensions`, `tile_dimensions`, `mathop`, `pool_type`, `unpack_to_dest`. Do not use local variants such
  as `dest_sync_mode`.
- Do not bundle several axes into one tuple parameter (for example `formats_dest_acc_sync_dims`). The comparer
  can only split a bundle whose slots are enums or dataclasses, so bundles hide `dest_acc`, `dest_sync` and
  `pool_type` from the join.
- An axis that depends on earlier ones is a callable whose argument names are the axes it depends on, for
  example `dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats)`. It returns a list, or a
  single value.
- An existing generator that returns a list of tuples can be exposed as flat axes with
  `helpers.param_config.flat_axes(names, combinations)`. The cross product equals the original list as a set of cases (the order can
  differ, because cases are grouped by first-seen value), and columns wrapped in `runtime()` stay runtime-only.
- Axes that only change the input data, not the compiled kernel, are wrapped in `runtime()`.

## Perf CSV columns

- A CSV column comes from the field name of a template or runtime parameter class in
  `helpers/test_variant_parameters.py`. `test_perf_header_gate.py` fails when two parameter classes declare the
  same field name, so a shared column such as `mathop` or `pool_type` is emitted through `MATH_OP`.
- A column that records a setting without changing the build uses a parameter whose `convert_to_cpp` returns
  an empty string (`QUASAR_VARIANT`, `SFPLOADMACRO`, `RELU_TYPE`).
- A schema change in `helpers/perf/test_schemas.py` bumps that test's `version`. A renamed column keeps its
  old name in `aliases` (old name to new name) so earlier reports stay comparable.

## `tile_cnt` is not one quantity

The `tile_cnt` column carries a different count depending on the module. Check the module before comparing
values across architectures. Verified so far:

| Module | `TILE_COUNT` value |
| --- | --- |
| `perf_math_matmul.py` | `variant_tile_count * num_blocks` |
| `test_matmul.py` | `output_tile_cnt` (`rt_dim * ct_dim`) |
| Quasar eltwise binary | `tile_cnt`, `input_tile_cnt` and `output_tile_cnt` columns |
| WH/BH eltwise binary | `tile_cnt` only |

The remaining modules have not been audited.
