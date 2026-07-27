# The bf4 decode matmul ceiling, and the one small thing blocking a fix

Investigation notes for gemma2-9B on 1xP150. Everything below is measured, not
projected, unless it says "projected".

## The problem

Decode on gemma2-9B is weight-bandwidth bound, so bfp4 weights should be roughly
2x faster than bfp8. They are not. Profiling the production `dram_sharded` matmul:

| weight dtype | achieved BW | MBU (of 550 GB/s) |
| --- | --- | --- |
| bfp8 | ~490 GB/s | 89% |
| bfp4 | ~308 GB/s | 55% |

bfp4 halves the bytes but does not halve the time, so most of the theoretical
gain is lost. Because bfp8 reaches 89% MBU on the same kernel and same shapes,
this is not a DRAM limit.

## Cause: batch-1 is padded to a 32-row tile

`tt_transformers` decode carries the activation as `[1, 1, 32, dim]` in standard
`[32,32]` tiles: one real row and 31 rows of padding. The matmul does the MAC
work for all 32. That waste is invisible when DRAM time dominates (bfp8) and
becomes the bottleneck when DRAM time is halved (bfp4).

DeepSeek's `dram_streaming_matmul` avoids it by using `[1,32]` tiny tiles, doing
1 row of MAC work instead of 32. Benchmarked at gemma2 FF1/FF3 shapes
(K=3584, N=14336, bfp4 weights, LoFi):

| kernel | m | time | achieved BW |
| --- | --- | --- | --- |
| production `dram_sharded` | 32 | 93.9 us | 308 GB/s |
| `dram_streaming_matmul` | 1 | 63.2 us | 453-500 GB/s |

PCC 0.99 against golden. So the 308 GB/s ceiling is a property of this kernel at
m=32, not of bfp4 and not of the hardware.

Sanity check that isolated numbers mean anything here: the same isolated harness
measures the production kernel at 93.9 us and the in-model profiler measures it
at 97.7 us, a 4% gap.

## What it would be worth

Substituting the streaming kernel for the DRAM-bound decode matmuls, projected:

- FF1/FF3 only: +9.5% end-to-end
- naive bridging on all decode matmuls: +23%
- chained bridging through the MLP: +30%

## How to feed it: CB tile aliasing (resolved)

The streaming kernel needs `in0` as a `[1,32]`-tiled row, replicated one copy per
compute core, height-sharded. **This is solved and needs no new ttnn API.** Skip
to "The bridge that works" for the answer; the dead end below is recorded because
it is the obvious first thing to try.

### The dead end

Trying to produce a `[1,32]`-tiled *tensor* from a decode activation with device
ops only:

| step | result |
| --- | --- |
| `untilize_with_unpadding` -> extract row 0 | works |
| `ttnn.repeat` -> replicate to 8 cores | works |
| `ttnn.tilize` -> make it `[1,32]`-tiled | **"Physical shard shape (1, 3584) must be tile {32,32} sized"** |
| `ttnn.copy` into a preallocated `[1,32]` buffer | **"Input tensor layout (ROW_MAJOR) must equal output tensor layout (TILE)"** |
| reshard, then `ttnn.copy` | same |
| `ttnn.assign` | same |
| `to_layout(TILE)`, then `ttnn.copy` | **"Input and output tensors must have the same tile shape"** |

No device op can produce a `[1,32]`-tiled tensor, and every copy path enforces
tile-shape equality. Host-side `from_torch(tile=...)` is the only constructor,
which would mean a host round-trip per token. Repro:
`deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge.py` and
`test_stream_mm_bridge2.py`.

That is a real limitation, but it is the wrong thing to want. A tensor-level
retile is unnecessary.

## The bridge that works

`dram_streaming_matmul` already demonstrates the answer internally: CB4, CB6 and
CB7 view a `[1,32]`-tiled `mm_out_tensor`'s memory as `[16,16]` by building a CB
descriptor from the tensor and then overwriting one field:

```python
cb7 = ttnn.cb_descriptor_from_sharded_tensor(cb_id, mm_out_tensor)
cb7.format_descriptors[0].tile = ttnn.TileDescriptor(ttnn.Tile([16, 16]))
cb7.format_descriptors[0].page_size = mul_tile_size
```

Tile aliasing is therefore already available in Python inside any `generic_op`,
independent of what the backing tensor declares. Applying it to CB0 lets the op
accept a plain ROW_MAJOR activation:

```python
row = ttnn.untilize_with_unpadding(x, [0, 0, 0, K - 1])   # [1,1,1,K] row-major
rep = ttnn.repeat(row, ttnn.Shape([1, 1, num_cores, 1]))  # replicate per core
in0 = ttnn.to_memory_config(rep, height_sharded_1xK)      # [1,K] per core
out = DRAMStreamingMatmul.op(in0, w, out_t, ..., in0_tile=ttnn.Tile([1, 32]))
```

Measured PCC 0.993 against golden, every step on device, no host round-trip and
no retile op. Repro: `test_stream_mm_cb_alias.py`. This needed a 3-line
`in0_tile` override in `micro_ops/dram_streaming_matmul/op.py`, defaulting to the
previous behaviour.

### Why the aliasing is legal

A `[1,32]` tile is 2 faces of `[1,16]`, i.e. 32 contiguous values, which is
exactly a row-major row. Measured on device for `[1,1,8,3584]` bf16 height-sharded
`[1,3584]` (`test_tile_alias_equiv.py`):

| | row-major | `[1,32]`-tiled |
| --- | --- | --- |
| total bytes | 57344 | 57344 |
| shard shape | (1, 3584) | (1, 3584) |
| pages x page size | 8 x 7168 | 896 x 64 |
| values via `to_torch` | identical | identical |

Same bytes in the same order. Only the pagination metadata differs, which is why
a CB can read one as the other.

## Next step

Wire this into `tt_transformers` MLP decode for FF1/FF3 behind an env flag,
PCC-check against the reference model, and measure the real end-to-end delta
against the predicted +9.5%. Nothing external is blocking it.

## Caveats

- The +23-30% assumes the isolated matmul numbers translate in-model. The 4%
  agreement on the production kernel supports that but does not prove it. Only
  the in-model prototype settles it.
- The bridge adds three ops per matmul (unpad, repeat, reshard). Measured cheap
  in isolation (1.4-8.7 us device time), but they eat into the 30 us/matmul
  saving and the real cost in-model is not yet measured.
- The streaming kernel's LLK (`kernel_includes/.../custom_mm.h`) restricts `in0`
  tile height to {1,2,4,8}. m=32 is not merely slow there, it produces garbage,
  so "just use the streaming kernel at m=32" is not an option.
- CB aliasing bypasses tensor-level spec checking. It is correct here only
  because row-major and `[1,32]`-tiled are provably byte-identical; it is not a
  general-purpose escape hatch.
- Weights need the one-time column-major shuffle (`shuffle_tensor_tiles`) and
  DRAM width-sharding at load time, which changes weight caching.

## Reproducing

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_cb_alias.py -s    # the bridge that works
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_tile_alias_equiv.py -s       # byte equivalence
python models/demos/deepseek_v3_b1/tests/unit_tests/bench_stream_mm.py               # 453-500 GB/s

# the dead end, kept for the record
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge.py -s       # tilize blocker
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge2.py -s      # copy-path blockers
```
