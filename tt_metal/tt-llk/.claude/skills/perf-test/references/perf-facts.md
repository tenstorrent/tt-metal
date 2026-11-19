# Perf facts

The single home for facts that every perf skill relies on. Other skills link
here instead of restating them; correct a fact here, once.

## Run types

| Run type | Measures | Threads not measured |
|---|---|---|
| `L1_TO_L1` | the functional run, end to end, `LOOP_FACTOR` times | none |
| `UNPACK_ISOLATE` | unpack | math stands in by clearing the source dvalids unpack posts; pack idles |
| `MATH_ISOLATE` | math | unpack stands in by posting the source dvalids math consumes; pack idles |
| `PACK_ISOLATE` | pack | unpack and math idle; pack runs without waiting for math |
| `SFPU_ISOLATE` | SFPU (Quasar only) | as the test defines it |

`ALL_PERF_RUN_TYPES` (`helpers/perf/core.py`) is the Wormhole/Blackhole set;
`PERF_RUN_TYPES_QUASAR` (`helpers/llk_params.py`) is the Quasar set. An isolate
is a valid stage measurement only when it runs the same loop body as
`L1_TO_L1` (see `perf-test`). The stand-in conventions per architecture are in
`wormhole-blackhole.md` and `quasar.md` next to this file.

## Build kinds

Each of these changes the measured code. Never mix them in one report or one
comparison.

- **Speed of light.** `--speed-of-light` turns runtime parameters into
  compile-time constants (`SPEED_OF_LIGHT` is defined). CI gates both kinds,
  in separate gates: the scheduled **LLK perf** run measures and gates speed
  of light; the PR, merge and `llk-perf-gate.yaml` nightly gates build the
  normal kind. A gate refuses to compare the two kinds ("CI" below). Normal
  builds also load the runtime-parameter struct, so they move more under
  layout changes. Both kinds describe the same sweep point (`perf-test`,
  "Parameters that compile in every build").
- **Asserts.** Wormhole and Blackhole CI build perf with
  `TT_LLK_DISABLE_ASSERTS=1`, which drops `-DENABLE_LLK_ASSERT`; Quasar perf
  CI keeps asserts. `run_test.sh --perf` matches that default per
  architecture. The setting is part of the ELF variant hash, so producer and
  consumer cannot silently mix asserts-on and asserts-off binaries.
  `perf_compare_commits.sh` defaults to asserts off on both sides.
- **Perf counters.** `--enable-perf-counters` compiles counter code and is a
  separate report kind: compare its `mean(...)` columns only with other
  counter reports.
- **Coverage.** `--coverage` instruments the kernels and invalidates perf.
- **`LLK_PROFILER`** is defined in every profiler build, functional or perf.
  It means "profiler instrumentation is compiled", not "this is a perf
  config", so it must never select parameter values.

## CSV and normalization

- Markers: `INIT` and `TILE_LOOP` are the kernel's `START_PERF_MEASURE`
  zones (`counters.h`: the profiler zone plus, in counter builds, the counter
  window);
  `KERNEL` is the zone `trisc.cpp` wraps around every profiler build.
- The raw `<module>.csv` holds per-marker means. The `<module>.post.csv`
  divides the `mean(...)` and `std(...)` columns of `TILE_LOOP` rows by
  `loop_factor * tile_cnt`: cycles per staged tile. `INIT` and `KERNEL` rows
  stay unnormalized. Analysis uses `.post.csv`.
- Columns come from the template and runtime dataclass fields that the test
  passes. The per-module column set is in
  `helpers/perf/test_schemas.py`; the hardware-free header gate
  (`tests/hardware_free/perf/test_perf_header_gate.py`) fails on drift.
- The warehouse schema is chosen per architecture by `schema_for_arch` in
  `helpers/perf/parquet.py`: `helpers/perf/wide_schema.py` for Wormhole and
  Blackhole, `helpers/perf/wide_schema_quasar.py` for Quasar. Column
  placement and naming rules are in `perf-test`, "Perf schema".

## Accounting and comparability

- The schema version is `PERF_TEST_SCHEMAS[<module>]["version"]` in
  `helpers/perf/test_schemas.py`, not a CSV column. Bump it once per PR,
  relative to main, when the module's columns change. Rows from different
  versions are not comparable, except across a rename recorded in `aliases` or
  `test_name_aliases`, which the comparer maps (`perf-test`, "Renames";
  `perf-regression-check`, "Interpreting results").
- A `tile_cnt` accounting change makes rows incomparable without a version
  bump. When `tile_cnt` counts both operand tiles of a two-operand kernel,
  per-tile `MATH_ISOLATE` is about half a result tile; that halving is not a
  speedup. Do not shrink `TILE_CNT` to restore an older number: it is also the
  unpack loop bound and the stand-in dvalid count.
- `tile_cnt` means different counts in different modules. Check what a module
  counts before renaming an axis to it: the strided column reduce's
  `max_tiles_per_iter` became `tile_cnt` only because it is exactly the input
  tile count that `TILE_COUNT(tile_cnt_A)` carries.
- At 16-bit, `dest_acc=Yes` can double the Dest handshakes relative to
  `dest_acc=No`, so a `dest_acc` delta mixes a datapath change with twice the
  section-done traffic.
- Compare like with like: architecture, build kind, `loop_factor`, marker,
  schema version, and report kind.
- A comparison with few or no matched points is inconclusive
  (`perf-regression-check`, "Interpreting results").

### `tile_cnt` per module

`tile_cnt` is the `TILE_COUNT` runtime value, the divisor of the `.post.csv`
normalization (`loop_factor * tile_cnt`). Each test sets it; there is no
shared formula. The tables list every module in `PERF_TEST_SCHEMAS` and
`PERF_TEST_SCHEMAS_QSR` that has a `tile_cnt` column, with the value read from
the `TILE_COUNT(...)` call of the test the perf module runs (`test_*.py`, or
the perf module itself where it passes `TILE_COUNT` directly). `rt`, `ct`,
`kt` are the tile-grid dims of the output rows, output columns and the
contraction.

Wormhole and Blackhole:

| Module | `tile_cnt` | Related columns |
| --- | --- | --- |
| `perf_bcast` | `tile_cnt_A` (`generate_stimuli`, operand A tiles) |  |
| `perf_block_max8` | 1 |  |
| `perf_csa_index_remap` | 9 (fixed) |  |
| `perf_dense_pack_untilize` | `tile_cnt_A` |  |
| `perf_eltwise_add_scalar` | `tile_cnt_input`, the input tiles of operand A | output tiles are input tiles divided by the accumulation depth |
| `perf_eltwise_bcast_col_custom` | `tile_cnt_A` |  |
| `perf_eltwise_binary` | `tile_cnt_A` of `input_dimensions`: `(rows // tile_rows) * (cols // tile_cols)` | with `acc_to_dest`, `num_blocks * num_tiles_in_block` is half of `tile_cnt` (see below) |
| `perf_eltwise_binary_reuse_dest` | `tile_cnt_input`, the input tiles of operand A |  |
| `perf_eltwise_binary_broadcast` | `tile_cnt_A` (runs the `test_eltwise_binary` body) | same `acc_to_dest` note as `perf_eltwise_binary` |
| `perf_eltwise_binary_sfpu` | `tile_cnt_A` for the float and integer sweeps; `tile_cnt_A + tile_cnt_B` for mixed formats and the broadcast kernel |  |
| `perf_eltwise_mul_scalar_block` | `len(scalars) * block_size` (4 scalars times `block_size`) |  |
| `perf_eltwise_unary_datacopy` | `tile_cnt_A` |  |
| `perf_eltwise_unary_datacopy_custom` | `tile_cnt_A` |  |
| `perf_eltwise_unary_typecast` | `tile_cnt_A` |  |
| `perf_eltwise_unary_sfpu` | `tile_cnt_A` |  |
| `perf_fast_tilize` | `tile_count`, the height tiles times the width tiles of `input_dimensions` |  |
| `perf_fast_tilize_full` | `rt * ct` |  |
| `perf_fast_tilize_metal_api` | `rt * ct` |  |
| `perf_fast_untilize` | `rt * ct` |  |
| `perf_hadamard` | `num_tiles`, the row count of `inputs` (one transform per tile) |  |
| `perf_math_matmul` | `rt * ct * kt * num_blocks` | `num_blocks` is its own column; `num_tiles_in_block = rt * ct` |
| `perf_mul_reduce_scalar` | `num_tiles` axis (operand A tiles) |  |
| `perf_sdpa_bcast_col_srca_srcb_reuse` | 1 (fixed) |  |
| `perf_sdpa_weighted_reduce` | `tile_cnt_qk`, the qk operand tiles, not the weight tiles (`tile_cnt_w`) |  |
| `perf_sfpu_sdpa` | 1, or `len(src_tiles)` for the correction variant |  |
| `perf_sum_reduce_scalar` | `num_tiles` axis (operand A tiles) |  |
| `perf_rmsnorm_clear_product_tile` | `output_tiles = CYCLES * (capacity - 1) * dest_tiles`, the output tiles, not the `dest_tiles` operand tiles |  |
| `perf_mul_reduce_scalar_chunked` | `num_tiles` axis (operand A tiles) |  |
| `perf_rmsnorm_bcast_scalar_dest_reuse` | `num_tiles` axis (tiles in operand A; operand B is one tile) |  |
| `perf_rope` | `tiles`, the Dest tiles of the geometry |  |
| `perf_sdpa_bcast_col_srcb_reuse` | `num_tiles` axis (1 or 2) |  |
| `perf_top32_rm` | ordinary variants: `2 * ceil(row_elements / 64)` (1024, 1088, 1152 elements give 32, 34, 36); pre-sorted variant: `2 * (row_elements // 1024 + ceil((row_elements % 1024) / 64))` (1024, 1088, 1152 give 2, 4, 6) | |
| `perf_topk_pack_supplied_indices` | `2 * tiles`, `tiles = ceil(k / ELEMENTS_PER_TILE)` |  |
| `perf_topk_xl` | `num_rows * num_chunks * tiles_per_seq` |  |
| `perf_topk_xl_split` | `num_rows * num_chunks * tiles_per_seq` |  |
| `perf_transpose_dest` | the `tile_cnt` axis (16) |  |
| `perf_matmul` | `rt * ct * kt` |  |
| `perf_pack_l1_acc` | `tile_cnt_A` |  |
| `perf_pack` | `tile_cnt_A` | `num_blocks * num_tiles_in_block` equals `tile_cnt` |
| `perf_pack_tiny_tile_block` | `tile_cnt_A` |  |
| `perf_pack_tiny_tile_reconfig` | `tile_cnt_A` |  |
| `perf_pack_untilize` | `tile_cnt_A` |  |
| `perf_reduce` | `tile_cnt_A` | `output_tile_cnt` is 1 for reduce-to-one, else `tile_cnt` |
| `perf_reduce_block_max` | `block_ct_dim` |  |
| `perf_sfpu_binop_scalar` | `tile_cnt_A` |  |
| `perf_sfpu_ema` | `tile_cnt` (operand A tiles) |  |
| `perf_sfpu_reduce` | `tile_count`, the input tiles |  |
| `perf_sfpu_reduce_sdpa` | `tile_count` / `tile_cnt_A`, the input tiles |  |
| `perf_sfpu_ternary` | `tile_count`, the operand A tiles of `input_dimensions` |  |
| `perf_vif_equiv_sweep` | `tile_cnt_A` (runs the unary SFPU test body) |  |
| `perf_unpack_a_bcast_eltwise` | `tile_count`, the input tiles (stimuli carry `reuse_factor` A tiles) |  |
| `perf_unpack_reduce_col_tilizeA_strided` | `tile_cnt_A` |  |
| `perf_unpack_tilize` | `tile_cnt_A` |  |
| `perf_unpack_transpose` | `tile_count`, the input tiles |  |
| `perf_custom_mm` | `ct * kt` |  |
| `perf_matmul_custom` | `rt * ct` (output tiles) |  |
| `perf_matmul_in1_column_stride` | `2 * ct` |  |
| `perf_compressed_custom_mm` | `rt * ct` (output tiles) |  |
| `perf_matmul_custom_compressed` | `ct` (`run_compressed`) |  |
| `perf_matmul_face_compressed` | `ct` (`run_compressed`) |  |
| `perf_unpack_A_custom` | `tile_cnt_A` |  |

Quasar:

| Module | `tile_cnt` | Related columns |
| --- | --- | --- |
| `perf_eltwise_binary_broadcast_quasar` | `tile_cnt_A` | `output_tile_cnt` is the result tiles |
| `perf_eltwise_binary_quasar` | `tile_cnt_A` | `output_tile_cnt` is the result tiles (input tiles divided by tiles per accumulation); with `acc_to_dest`, `num_blocks * num_tiles_in_block` is half of `tile_cnt` |
| `perf_eltwise_binary_reuse_dest_quasar` | `tile_cnt_input`, the input tiles of operand A | `output_tile_cnt` is the reuse-dest output grid |
| `perf_eltwise_binary_sfpu_quasar` | tiles staged in `buffer_A` (`max(src0_idx, src1_idx, dst_idx) + 1` for the two-operand kernels), times `num_blocks` in the dest-depth variant |  |
| `perf_eltwise_unary_datacopy_quasar` | `tile_cnt_A` |  |
| `perf_eltwise_unary_sfpu_quasar` | `tile_cnt_A` (1 for the single-tile variants) |  |
| `perf_eltwise_unary_typecast_quasar` | `tile_cnt_A` (runs the unary SFPU test body) |  |
| `perf_matmul_int4_dequant_quasar` | `output_tile_cnt * kt`, i.e. `rt * ct * kt` |  |
| `perf_matmul_quasar` | `output_tile_cnt * kt`, i.e. `rt * ct * kt` |  |
| `perf_pack_l1_acc_quasar` | `tile_cnt` of `input_dimensions` |  |
| `perf_pack_quasar` | `tile_cnt_A` |  |
| `perf_pack_untilize_quasar` | `tile_cnt_A` |  |
| `perf_reduce_quasar` | `tile_cnt` (input tiles) | no `output_tile_cnt` column; the test's local `output_tile_cnt` (1 for reduce-to-one, else `tile_cnt`) only sizes the result stimuli |
| `perf_sfpu_add_parallel_matmul_quasar` | `tile_cnt_add`, the SFPU add operand tiles (the parallel matmul is not counted) |  |
| `perf_sfpu_exp_parallel_matmul_quasar` | `tile_cnt_exp`, the SFPU exp input tiles (the parallel matmul is not counted) |  |
| `perf_sfpu_reduce_quasar` | `tile_cnt`, the input tiles |  |
| `perf_sfpu_where_quasar` | `3 * tile_cnt_single`: condition, true and false tiles (3 at the perf size of one tile each) |  |
| `perf_transpose_dest_quasar` | `tile_cnt_A` |  |
| `perf_unary_broadcast_quasar` | `tile_cnt` of `input_dimensions` |  |
| `perf_unpack_reduce_col_tilizeA_strided_quasar` | `tile_cnt_A` |  |
| `perf_unpack_tilize_quasar` | `tile_cnt_A` |  |
| `perf_unpack_unary_operand_quasar` | `tile_cnt_A` |  |

`perf_fused`, `perf_fused_quasar`, `perf_sfpu_sdpa_fw`,
`perf_custom_mm_reuse_dest_srcb` and `perf_sdpa_custom_mm_reuse_dest_srcb`
have no `tile_cnt` column in their schema. `postprocess_tile_loop` fills a
missing `tile_cnt` with 1, so their `TILE_LOOP` rows are divided by
`loop_factor` alone.

Differences that matter when comparing architectures:

- The same matmul name carries three formulas: WH/BH `perf_matmul` and Quasar
  count `rt * ct * kt`, `perf_math_matmul` multiplies by `num_blocks`, and
  `perf_matmul_custom` counts only the output tiles `rt * ct`. A per-tile
  cycle count from one is not comparable with another.
- Binary SFPU kernels count operand tiles, not result tiles, so per-tile
  cycles are about half a result tile (see above).
- `tile_cnt` is not always operand A: `perf_sdpa_weighted_reduce` counts the
  qk operand tiles and `perf_rmsnorm_clear_product_tile` the output tiles.
- With `acc_to_dest`, `num_tiles_in_block` is the output-based value (WH/BH
  computes it from the accumulated output dimensions; the Quasar eltwise
  binary test also passes the output value unsuffixed), so
  `num_blocks * num_tiles_in_block` is half of `tile_cnt`. The per-block
  `input_num_tiles_in_block` column, `num_tiles_in_block * 2` there, is the one
  that matches: `num_blocks * input_num_tiles_in_block == tile_cnt`.
- `input_tile_cnt` is only a retired alias of `tile_cnt`; the columns that
  exist are `tile_cnt` and, on a few modules, `output_tile_cnt`.

## Where reports land

- Workers write `<module>.<worker>.csv` and `.post.csv` into
  `/tmp/tt-llk-build/temp_perf_data/` (`$RUNNER_TEMP/tt-llk-build/...` under
  GitHub Actions). The worker is `gw0`, `gw1`, … under `-n`, else `master`.
- At session finish they are combined into
  `tt_metal/tt-llk/perf_data/runs/<tag>/<module>/` and `perf_data/latest`
  points at that run. The producer phase writes no report.
- `PERF_RUN_TAG` names the run (default `local-<utc timestamp>-<pid>` from
  `run_test.sh`). `run_test.sh` refuses a tag whose directory exists: a reused
  tag mixes rows from two runs.
  `PERF_KEEP_RUNS` (default 10) bounds how many runs are kept; raise it for
  A/B or repeat series, or older runs are pruned mid-analysis.
- `perf_data/` is gitignored. A report is identified by test, architecture,
  commit, and exact command.

## Determinism and layout sensitivity

- On Wormhole, one build normally gives identical cycles run after run. Spread
  within one build points to sticky state or a race, not noise.
- Two builds whose measured code is identical can still differ by about ±10%,
  with configs sitting at two or three fixed levels. An unrelated change,
  often to the idle threads' code, moves configs between the levels. The
  diagnosis recipe is "Layout sensitivity" in `perf-regression-check`.

## CI

- Workflow **LLK perf** (`llk-perf.yaml`) runs `tests/run_llk_perf_wormhole.sh`
  and `tests/run_llk_perf_blackhole.sh`: producer `-n 10`, consumer
  `--dist loadgroup -n 15`, `-m "perf and not accuracy"`, `--timeout=60`,
  speed of light for the scheduled run, `TT_LLK_DISABLE_ASSERTS=1`. The
  consumer stops at the first failure (`-x`, `LLK_PERF_EXITFIRST`); local
  `run_test.sh --perf` does not.
- Quasar perf CI (`build-quasar-perf.yml`) keeps `LLK_ASSERT`; local
  `run_test.sh --perf --arch quasar` does too.
- CI cuts the perf sweep into groups of equal estimated time. It prefers
  measured per-test durations from earlier runs (`LLK_SPLIT_DURATIONS`,
  summed by `perf/sum_split_durations.py`) and otherwise falls back to the
  per-module seconds in `tests/perf_split_costs.json` and
  `tests/perf_split_costs_sol.json`. A module missing from that table counts
  as 1 s. When a PR adds a perf module or changes its build count a lot,
  update its entries and say so in the PR.
  These values schedule CI only; changing them does not establish a new
  performance baseline and must not be used to explain away unmatched gate
  points.
- The perf gates and `perf_compare_commits.sh` share
  `perf/regression_compare.py`: a point regresses only when it is slower by
  both more than the threshold and more than the cycle floor. Thresholds
  differ by architecture, gate and build kind, and some gates give modules
  their own. Each gate calls `llk-perf-gate-impl.yaml`; read the values at
  the call site:
  - Speed-of-light nightly gate: the `gate` job in
    `.github/workflows/llk-perf.yaml`, a per-architecture `include` matrix
    (`threshold`, `module_thresholds`) and a literal `min_cycles`.
  - PR gate: the `llk-perf-gate` job in `.github/workflows/llk-perf-gate.yaml`,
    a per-architecture `threshold` matrix (no module thresholds) and a
    literal `min_cycles`.
  - Normal-build nightly gate: the `nightly-gate` job in the same file, a
    per-architecture matrix with `module_thresholds` and a literal
    `min_cycles`.
  - Merge gate: the `thresholds` output and the literal `min_cycles` in
    `.github/workflows/merge-gate.yaml`.
  - Local comparisons: the `perf_compare_commits.sh` defaults (`--help`).

  When reproducing a gate verdict locally, match its build kind and pass its
  values to `perf_compare_commits.sh` with `--threshold` (the module's own
  threshold when it has one) and `--min-cycles`.
