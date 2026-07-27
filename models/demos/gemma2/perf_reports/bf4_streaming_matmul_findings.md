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

## Why it is not wired up yet

The streaming kernel needs `in0` as a `[1,32]`-tiled row, replicated one copy per
compute core, height-sharded. Building that from a real decode activation using
only device ops:

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
tile-shape equality. The only way to get one is host-side
`from_torch(tile=...)`, which would mean a host round-trip per token and defeats
the purpose. Repro: `deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge.py`
and `test_stream_mm_bridge2.py`.

## The missing primitive is a free one

A `[1,32]` tile is 2 faces of `[1,16]`, i.e. 32 contiguous values, which is
exactly a row-major row. Measured on device for `[1,1,8,3584]` bf16 height-sharded
`[1,3584]` (`test_tile_alias_equiv.py`):

| | row-major | `[1,32]`-tiled |
| --- | --- | --- |
| total bytes | 57344 | 57344 |
| shard shape | (1, 3584) | (1, 3584) |
| pages x page size | 8 x 7168 | 896 x 64 |
| values via `to_torch` | identical | identical |

Same bytes in the same order. Only the pagination metadata differs, and a
re-spec'd view recomputes that. So the conversion needs **no kernel and no data
movement** -- only a `Tensor` that aliases the same buffer with a different
`TensorSpec`.

C++ already has the pieces: `Tensor(DeviceStorage)` exists, and `DeviceStorage`'s
copy constructor explicitly "shares the underlying device memory". None of it is
exposed to Python.

## The ask

Expose a device-side tile-spec view, something like
`ttnn.reinterpret_tile(tensor, tile)`, returning a tensor that aliases the same
buffer with a different tile spec. Validated to require identical total size and
shard shape.

It is small, it is metadata-only for this case, and it unlocks a measured
453-500 GB/s vs 308 GB/s on bfp4 batch-1 decode -- for every model on the stack,
not just gemma2. If an aliasing view is unacceptable, DeepSeek's `tilize_8x32`
(159 lines of Python plus a 23-line kernel) is a working template for a
`tilize_1x32` copy op instead.

## Caveats

- The +23-30% assumes the isolated matmul numbers translate in-model. The 4%
  agreement on the production kernel supports that but does not prove it. The
  real number needs the prototype, which needs the primitive above.
- The streaming kernel's LLK (`kernel_includes/.../custom_mm.h`) restricts `in0`
  tile height to {1,2,4,8}. m=32 is not merely slow there, it produces garbage,
  so "just use the streaming kernel at m=32" is not an option.
- Aliasing is a sharp tool and would need spec validation to be safe.

## Reproducing

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_tile_alias_equiv.py -s      # byte equivalence
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge.py -s      # tilize blocker
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge2.py -s     # copy-path blockers
python models/demos/deepseek_v3_b1/tests/unit_tests/bench_stream_mm.py               # 453-500 GB/s
```
