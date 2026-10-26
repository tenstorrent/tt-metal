# OFT per-op tests (`tests/oft_ops/`)

Isolated tests for **every ttnn op the OFT (Orthographic Feature Transform) monocular 3D
detection model ran on device**, with the shapes, dtypes, memory configs and compute configs
of those runs. There are about 300 cases in 25 files. The tests import **no** model code.

## Where the shapes come from

Every case was reverse-engineered from one of two **tt-metal kernel caches** of Blackhole
end-to-end runs. The two runs are different builds of the model (see below):

| | Cache #1 | Cache #2 |
|---|---|---|
| Directory | `tt-metal-cache-oft-e2e/tt-metal-cache1088b10b01/17329853860278502937` | `tt-metal-cache930749440442549079` |
| Source tree | modified out-of-tree checkout | fork of main (`tt-metal-bos`) |
| Factories read at | main `1088b10b014` (2026-02-02, the cache key) | main `628a4797bf7` (dated from the sources the kernels include: 2026-04-16 .. 04-24) |
| Model | modified OFT (see table below) | the in-tree `models/experimental/oft` of that date |

- Compile-time args survive in each ELF's DWARF (`kernel_compile_time_args`, readable with
  `dwarfdump --debug-info`), alongside the generated defines, CB data formats and math
  fidelity. The program factories map them back to op parameters. Runtime args are **not**
  cached.
- Both runs used a **5x4 = 20-core worker grid**: every sharded tensor, work split and mcast
  has 20 cores. Model-faithful cases shard over `CoreRange((0,0),(4,3))` and skip on smaller
  grids.

Each test docstring marks how sure its shapes are:

| Mark | Meaning |
|---|---|
| `[H]` | read directly from compile-time args / defines / CB formats |
| `[M]` | op / dtype / memory config read directly; shape inferred from the architecture |
| `[L]` | op known; shape is a representative value inside the recovered bounds |

Case ids end with the first 6 digits of the cache hash of the kernel variant they reproduce.
Cache-#2 cases sit in `*_v2` / `*_oft2` tests or in the cache-#2-only files listed below.

## How the runs differ

OFT lived in `models/experimental/oft` until `ae4ef6b7c16` removed it. The caches are treated
as ground truth where they disagree with that code:

| | In-tree code (`628a4797bf7`) | Cache #1 | Cache #2 |
|---|---|---|---|
| GroupNorm | `ttnn.group_norm` | on host (convs carry folded biases, conv1 fused ReLU) | on device, as in-tree |
| Conv weights | bfloat8_b | bfloat16 | bfloat8_b |
| ResNet-18 on device | all layers | stem, maxpool, layer1, layer2 (layer2 in float32) | all layers |
| OFT scales on device | 1/8, 1/16, 1/32 | 1/8 only (`lat8`) | all three |
| Integral image | `intimg` (default strategy `None`) | `intimg`, float32 | `"to_uint32"`: typecast -> uint32 `cumsum` (dims 1, 2) -> typecast |
| grid_sample | bilinear, precomputed | nearest, 44 slices of 576 rows, float32 | bilinear, precomputed, 18/12/11 slices, bf16 |
| OFT linear / topdown / head | 256 channels | 128 channels | 256 channels |
| Topdown convs | DRAM-sliced, row-major | L1 full | DRAM-sliced, row-major |
| atan2 | `ttnn.atan2` | composite (`where`, `ge`, `logical_and`, `atan`, ...) | native binary SFPU `atan2` |
| Conv `packer_l1_acc` | off | - | on, except the topdown convs |
| Conv config tensors | DRAM | - | L1 |
| Stem conv / maxpool input | row-major / bfloat8_b | - | bf16 TILE / bf16 TILE in DRAM |
| Input normalization divide | `fast_and_approximate_mode=True` | - | exact SFPU divide |

Other recovered facts:

- Input image 1x3x384x1280; BEV grid 159x159 (+63 pad rows = 25344).
- Cache #2 keeps one model bug out of the tests: the stem GroupNorm passes the 64-channel
  gamma/beta to both 32-channel halves, so both halves get the first 32 values.
- `compute_pool_2d_quant` (cache #2) only exists in the fork: it is mainline `compute_pool_2d`
  with int8 defines (both off). The six cache-#2 `compute_pool_2d` SUM variants are the bilinear
  `grid_sample` compute kernel, not a pool.

## Coverage

| File | Ops |
|---|---|
| `test_conv2d.py` | Cache #1: 13 convs (stem 4 DRAM slices, layer1/2, lat8 1x1, topdown, head, NMS 5x5). Cache #2 (`v2`): 14 height-sharded convs with bfloat8_b weights and `packer_l1_acc` (stem TILE-in, layer1-4 + downsamples, topdown 256-ch DRAM-sliced row-major, head 256->9), 3 auto-sharded lateral 1x1 convs (matmul path). Auto grid, pinned 5x4 grid, emulator proxies |
| `test_max_pool2d.py` | stem maxpool (bfloat8_b TILE; cache #2: bf16 DRAM input, 2 width slices), NMS pool with `return_indices` |
| `test_group_norm.py` | sharded GroupNorm (HS / BS, TILE / row-major, negative mask, in-place), DRAM GroupNorm with `num_out_blocks` |
| `test_intimg.py` | `ttnn.experimental.intimg`, float32 [1, 48, 160, 256] (cache #1) |
| `test_cumsum.py`, `test_typecast.py` | uint32 `cumsum` over dims 1 and 2 at 3 scales, the full quantize -> cumsum -> dequantize chain; bf16 <-> uint32 `typecast` (cache #2) |
| `test_grid_sample.py` | nearest (cache #1) and bilinear (cache #2), precomputed grid, 7-point channel batching, sharded output |
| `test_linear.py` | OFT linear 1792 -> 128 (cache #1) and -> 256 (cache #2) with the cached 1D program configs (LoFi, fused ReLU) |
| `test_binary.py` | sharded in-place `sub_` / `add_` / `mul_` (float32, bf16); residual adds with fused ReLU; integral-image scale multiply by a 0-d tensor; decode broadcasts; scalar +/- pi |
| `test_resnet_add.py` | cache-#2 residual adds without activation (incl. HS + bfloat8_b DRAM, DRAM + DRAM) |
| `test_binary_bcast.py` | input normalization (row-broadcast sub / div, row-major lhs), encoder decode row-broadcast mul / add, `centers` add, fast divide by 2.0 |
| `test_unary.py` | `exp`, `reciprocal`, `atan`, `gtz`, `ltz`, `eqz` (cache #1), `relu` over 10 memory configs (cache #2) |
| `test_exp_decode.py` | `exp` on [1, 159, 159, 3], TILE permutes of the decode offsets / heatmap |
| `test_comparison.py`, `test_where.py` | `ge(x, 0.0)`, `logical_and`, the three `where` forms: pieces of cache #1's composite `atan2` |
| `test_atan2.py` | `ttnn.atan2` (native in cache #2 and on current main) |
| `test_to_layout.py` | float32 / bf16 tilize and untilize (incl. sharded, with unpadding), image 3-channel tilize, `tilize_with_val_padding` |
| `test_encoder_layout.py` | TILE BEV-grid slices (tile-aligned and unaligned begins), 3-channel untilize, 1-channel tilize |
| `test_sharding.py` | interleaved <-> sharded moves, `sharded_to_interleaved`, `ttnn.move` (sharded and DRAM), `interleaved_to_sharded_partial`, `sharded_to_interleaved_partial` (cache #1: 44 slices; cache #2: stem and 3 OFT scales) |
| `test_reshard.py`, `test_concat.py`, `test_split.py` | layer4 BS -> HS reshard, same-width reshard; sharded width concat; stem channel split |
| `test_slice.py` | tile / row-major slices (incl. 5D), `padded_slice` (tile and row-major input), `slice_write` |
| `test_reshape.py`, `test_permute.py` | weight prep, conv input, grid, feature, head output and NMS heatmap reshapes / permutes |

Not covered:

- Host-side ops (cache #1 GroupNorm, `nonzero`, top-k peak selection).
- Op-internal kernels, exercised through their parent op rather than standalone: conv-internal
  `padded_slice` / halo / `pack_untilize` / interleaved-to-sharded copies (`test_conv2d.py`),
  and `cumsum`'s internal uint32 transposes / permutes (`test_cumsum.py`).
- Cache #2: one untilize variant (`[10,8]`, 256 wide) and one L1-input `reshape_tiled` variant
  could not be mapped.

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
  Metal 2.0 (unary, `intimg`, `cumsum`, `permute`, `concat`, `split`, interleaved <-> sharded)
  are called directly.
- Memory configs in case tables are device-free specs: `"dram"`, `"l1"`, `("hs", shard)`.
  They are resolved with `U.resolve_memcfg(spec, mesh)`. Cache-#2 files that need other grids
  (block sharding, the 15-core 5x3 layer4 grid) build them with a local `_memcfg`.
- Correctness:
  - `assert_pcc` against a torch reference computed on the values the device holds.
  - Bit-exact (`assert_lossless`) for data movement, nearest grid_sample, `where`,
    comparisons, integer `cumsum` / `typecast` and max-pool values / indices.

## Status

Authored from static analysis only.

Expect first-run touch-ups:

- tensors with partly empty shards on the 5x4 grid (576 rows over 20 x 32-row shards,
  25281 rows over 1280-row shards)
- row-major shards with narrow rows (C=9 head output, 16-wide uint16 indices)
- `max_pool2d` on a 1-channel row-major input
- conv DRAM slicing combined with a pinned `core_grid`

Known low-confidence areas:

- which residual add carries a fused ReLU in the cache-#1 topdown (kernels 852339 vs 232375)
- the positions-decode add and the `grid[1:, 1:]` slices (`[L]`)
- cache #2: the `reshard_same_width` source, the L1-interleaved -> 20-core reshard source,
  the row-major `padded_slice` slice rows, two `slice_write` shapes, the 1-channel tilize and
  the hash suffixes of two feature reshapes (`[L]`)
- cache #2: the DRAM GroupNorm uses named compile-time args the ELF does not keep, so its
  shape comes from the model (`[M]`)
