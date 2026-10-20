# VAD-variant per-op tests (`tests/vad_variant_ops/`)

Isolated tests for **every ttnn op a VAD/SSR-style autonomous-driving model ran**, with the
shapes, dtypes, memory configs and compute configs that model used. There are about 236 cases
in 25 files.

## Where the shapes come from

The model can't be run or landed here; it lives on a fork. These tests
import **no** model code. Every case was reverse-engineered from a **tt-metal kernel
cache** of one Blackhole run of the model (`tt-metal-cache/3204133098312896700`):

- Each `kernels/<kernel>/<hash>/` entry has the generated defines, CB data formats and math
  fidelity (`chlkc_descriptors.h`), plus the compiled ELFs.
- The compile-time args survive in each ELF's DWARF: `kernel_compile_time_args` has a
  `DW_AT_const_value`, readable with `dwarfdump --debug-info`. Named compile-time args and
  constexpr locals show up the same way.
- The program factories map those args back to op parameters: per-core blocking, K/N, page
  sizes (= row widths and dtypes), shard specs, fused activations, broadcast type, and so on.
  Main has moved on since the fork, so the factories were read at the matching main revisions
  (`6b9b6a0e0e2~1`, `2508e6e197a`, `036db1441f1^`). Docstrings cite them.
- **Runtime args are not cached.** Shapes that only appear at runtime (most interleaved
  eltwise ops, slice ranges, concat dims, scalar values, LN epsilon) are lost.

Each test docstring marks how sure its shapes are:

| Mark | Meaning |
|---|---|
| `[H]` | read directly from compile-time args |
| `[M]` | op / dtype / memory config read directly; shape inferred from the architecture |
| `[L]` | op known; shape is a representative value inside the recovered bounds |

Case ids end with the first 6 digits of the cache hash of the compute (or main) kernel
variant they reproduce.

## Model facts recovered

- ResNet-50 backbone on 6 cameras at 384×640, run one camera at a time. The stride-32 map
  is 12×20 = 240 keys; the FPN output is 2048→256 over all 6 cameras.
- BEVFormer-style encoder: BEV 100×100 = 10000 queries, embed 256, 8 heads × 32, FFN 512.
- Temporal self-attention uses a 2-frame queue and 4 points.
- Spatial cross-attention uses 8 points. Its cameras are grouped by hit count, padded to a
  multiple of 32: (1, 3680), (1, 3072), (2, 1376), (2, 992).
- The run used a **5×4 = 20-core worker grid**. Every work split and sharded tensor accessor
  has 20 banks. Model-faithful sharded cases shard over `CoreRange((0,0),(4,3))` and skip on
  smaller grids.

## Fork-only ops

The fork's custom ops are not in main:

- deformable attention
- nonzero
- getitem
- setitem
- same-width reshard
- flip variant

`test_equivalents.py` runs the closest stock op for each at the recovered shapes, i.e. the
op `models/experimental/vadv2` uses for the same step:

| Fork op | Stock op tested |
|---|---|
| deformable attention | `ttnn.experimental.multi_scale_deformable_attn` |
| nonzero | `ttnn.nonzero` |
| getitem | `ttnn.embedding` / `moreh.getitem` |
| setitem | `ttnn.scatter_add` / `ttnn.experimental.slice_write` |
| reshard | `ttnn.to_memory_config` |

`flip` variant (`read/write_kernel_rm_ip`) has no source and no recoverable layout, so it is
**not covered**.

## Running

```bash
# Full suite (real Blackhole):
pytest models/experimental/ops/quasar/tests/vad_variant_ops/ -v

# Emulator subset (small tensors, single device, not blackhole_scale):
pytest models/experimental/ops/quasar/tests/vad_variant_ops/ -m emulator -v
```

- `blackhole_scale`: cases that reproduce the model's 20-core sharded state, program configs
  pinned to the 5×4 grid, or very large tensors (including tensors allocated in the test body
  that conftest cannot see, e.g. the (10000, 256) query-gather tables). Excluded from `-m emulator`.
- `conftest.py` classifies everything else from a `shape` / `in_shape` / `a_shape` tuple, an
  `hw` int, or a conv/pool input volume (`n * h * hw * in_ch`), using the thresholds in `op_utils.py` (`EMU_MAX_HW`, `EMU_MAX_ELEMS`; heuristic,
  tune on the emulator).

## Conventions

These mirror `tests/yolo_ops/`:

- Helpers and model constants live in `op_utils.py`, which re-exports the generic
  `llama32_1b_quasar` / `yolo_ops` helpers. There is one file per op family.
- Memory configs in case tables are device-free specs: `"dram"`, `"l1"`,
  `("hs", shard)`, `("bs", shard[, grid_x])`. They are resolved with
  `U.resolve_memcfg(spec, mesh)`.
- Correctness:
  - `assert_pcc` against a torch reference computed on the values the device actually holds
    (bfloat8_b-quantized where applicable).
  - `assert_lossless` (bit-exact) for value-preserving data movement.
  - Exact equality for comparison / fill / nonzero.

## Status

- **Wormhole:** the full suite (including `blackhole_scale` cases, which fit the 8×8 grid)
  passes locally.
- **Blackhole:** not yet run.
- **Quasar emulator (`-m emulator`):** not yet run. Known expected failures, marked
  `xfail` on Quasar via `U.xfail_on_quasar`:
  - `test_max_pool2d*` with TILE output: the tiled-output pool path is broken on Quasar
    (pre-existing; see `tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_pool2d.py`,
    `tiled_out_*`). `test_max_pool2d_small[rm-out]` keeps ROW_MAJOR pool coverage.
  - `test_nonzero`: no Quasar port of `ttnn.nonzero` yet.

Known low-confidence areas:

- tiled reshape shapes (only the mapping-table size survives)
- the four `bmm.cpp` matmuls (only K and tiles per core known)
- MHA batch size
- scalar values
