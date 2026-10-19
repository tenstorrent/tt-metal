# OFT per-op tests (`tests/oft_ops/`)

Isolated tests for **every ttnn op the OFT (Orthographic Feature Transform) monocular 3D
detection model ran on device**, with the shapes, dtypes, memory configs and compute configs
of that run. There are 118 cases in 15 files. The tests import **no** model code.

## Where the shapes come from

Every case was reverse-engineered from a **tt-metal kernel cache** of one Blackhole
end-to-end run (`tt-metal-cache-oft-e2e/tt-metal-cache1088b10b01/17329853860278502937`):

- The cache directory is keyed by main commit **`1088b10b014`** (2026-02-02). The program
  factories at that commit map each kernel's compile-time args back to op parameters.
- Compile-time args survive in each ELF's DWARF (`kernel_compile_time_args`, readable with
  `dwarfdump --debug-info`), alongside the generated defines, CB data formats and math
  fidelity. Runtime args are **not** cached.

Each test docstring marks how sure its shapes are:

| Mark | Meaning |
|---|---|
| `[H]` | read directly from compile-time args / defines / CB formats |
| `[M]` | op / dtype / memory config read directly; shape inferred from the architecture |
| `[L]` | op known; shape is a representative value inside the recovered bounds |

Case ids end with the first 6 digits of the cache hash of the kernel variant they reproduce.

## The run differs from the in-tree model

OFT lived in `models/experimental/oft` until `ae4ef6b7c16` removed it, but the cached run was
built from a modified out-of-tree checkout and does **not** match that code. The cache is
treated as ground truth:

| | In-tree code | Cached run |
|---|---|---|
| GroupNorm | `ttnn.group_norm` | on host (no group_norm kernels); convs carry folded biases, conv1 has fused ReLU |
| Conv weights | bfloat8_b | bfloat16 |
| ResNet-18 on device | all layers | stem, maxpool, layer1, layer2 (layer2 in float32) |
| OFT scales on device | 1/8, 1/16, 1/32 | 1/8 only (`lat8`) |
| grid_sample | bilinear, 18 slices | nearest, 44 slices of 576 rows, float32 |
| OFT linear / topdown / head | 256 channels | 128 channels |
| Topdown convs | DRAM-sliced | L1 full |
| Dims decode | permute + row broadcast | NCHW, per-channel scalar broadcast |

Other recovered facts:

- Input image 1x3x384x1280; BEV grid 159x159 (+63 pad rows = 25344 = 44 x 576).
- The run used a **5x4 = 20-core worker grid**. Every sharded tensor, work split and
  mcast has 20 cores. Model-faithful cases shard over `CoreRange((0,0),(4,3))` and skip on
  smaller grids.

## Coverage

| File | Ops |
|---|---|
| `test_conv2d.py` | 13 convs: stem 7x7/s2 (4 DRAM slices), layer1/layer2 3x3 + downsample, lat8 1x1 (matmul path), topdown conv1/conv2, head (2 DRAM slices), encoder NMS 5x5. Auto grid, pinned 5x4 grid, and emulator proxies |
| `test_max_pool2d.py` | stem maxpool (bfloat8_b TILE), encoder NMS pool with `return_indices` |
| `test_intimg.py` | `ttnn.experimental.intimg`, float32 [1, 48, 160, 256] |
| `test_grid_sample.py` | nearest, precomputed grid, 7-point channel batching, sharded grid |
| `test_linear.py` | OFT linear 1792 -> 128 with the cached 1D program config (LoFi, fused ReLU) |
| `test_binary.py` | float32 sharded in-place `sub_` / `add_` / `mul_`; residual adds with fused ReLU (bf16, bf16 + bfloat8_b legacy path, float32, sharded + DRAM); decode broadcasts; scalar +/- pi |
| `test_unary.py` | `exp`, `reciprocal`, `atan`, `gtz`, `ltz`, `eqz` |
| `test_comparison.py` | `ge(x, 0.0)`, `logical_and` |
| `test_where.py` | `where` tensor-scalar-scalar / tensor-scalar-tensor / tensor-tensor-tensor |
| `test_atan2.py` | the composite `ttnn.atan2` the pieces above come from |
| `test_to_layout.py` | float32 untilize / sharded tilize, bf16 and uint16 tilize-with-padding, sharded `untilize` |
| `test_sharding.py` | interleaved <-> sharded moves, `ttnn.move`, `sharded_to_interleaved_partial` over all 44 slices |
| `test_slice.py` | tile / row-major slices (incl. 5D), `padded_slice`, `slice_write` (bf16 RM C=9, bfloat8_b TILE) |
| `test_reshape.py`, `test_permute.py` | weight prep, conv input, head output and NMS heatmap reshapes / permutes |

Not covered:

- `ttnn.typecast` and `ttnn.concat`/`split`: the in-tree code uses them but no kernels exist.
- The conv1 `padded_slice` with 3 input channels (16-B non-aligned rows) and the 1-element-row
  interleaved-to-sharded copy feeding the NMS conv: conv-internal, exercised through
  `test_conv2d.py` cases A and K rather than standalone.
- Host-side ops (GroupNorm, `nonzero`, top-k peak selection).

## Running

```bash
# Full suite (Blackhole):
pytest models/experimental/ops/quasar/tests/oft_ops/ -v

# Emulator subset (small tensors, single device, not blackhole_scale):
pytest models/experimental/ops/quasar/tests/oft_ops/ -m emulator -v
```

- `blackhole_scale`: cases pinned to the model's 5x4 grid or sharded state. They skip when
  the device grid is smaller and are excluded from `-m emulator`.
- `conftest.py` classifies everything else from a `shape` / `in_shape` / `a_shape` tuple or an
  `hw` int, using the thresholds in `op_utils.py`. Most model-sized cases have an
  emulator-sized `proxy-*` sibling with the same op config.

## Conventions

These mirror `tests/vad_variant_ops/`, whose 5x4-grid helpers `op_utils.py` re-exports:

- Ops are called through `U.op(name)`, which returns the `ttnn.experimental.quasar` port on
  Quasar and the mainline op elsewhere. Mainline factories that still build legacy
  `DataMovementKernel`s are rejected on Quasar. Ops whose mainline factory is already
  Metal 2.0 (unary, `intimg`, `permute`, interleaved <-> sharded) are called directly.
- Ops with neither a port nor a Metal 2.0 factory xfail on Quasar (`U.xfail_no_quasar_port`):
  `grid_sample`, `where` with a tensor branch, and `sharded_to_interleaved_partial`.
- Memory configs in case tables are device-free specs: `"dram"`, `"l1"`, `("hs", shard)`.
  They are resolved with `U.resolve_memcfg(spec, mesh)`.
- Correctness:
  - `assert_pcc` against a torch reference computed on the values the device holds.
  - Bit-exact (`assert_lossless`) for data movement, nearest grid_sample, `where`,
    comparisons and max-pool values / indices.

## Status

Authored from static analysis only.

Expect Quasar first-run touch-ups:

- tensors with partly empty shards on the 5x4 grid (576 rows over 20 x 32-row shards,
  25281 rows over 1280-row shards)
- row-major shards with narrow rows (C=9 head output, 16-wide uint16 indices)
- `max_pool2d` on a 1-channel row-major input
- conv DRAM slicing combined with a pinned `core_grid`

Known low-confidence areas:

- which residual add carries a fused ReLU in the topdown (cache kernels 852339 vs 232375)
- the positions-decode add and the `grid[1:, 1:]` slices (`[L]`)
