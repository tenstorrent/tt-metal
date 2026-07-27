# Escalation drafts: device-side tile-spec view

Two versions of the same ask. Evidence and repros live in
`bf4_streaming_matmul_findings.md` in this folder.

---

## A. GitHub issue (tenstorrent/tt-metal)

**Title:** Expose a device-side tile-spec view (`[32,32]` <-> `[1,32]`) to unblock
tiny-tile decode matmuls

### Ask

A ttnn API returning a tensor that aliases an existing device buffer with a
different tile spec:

```python
view = ttnn.reinterpret_tile(tensor, ttnn.Tile([1, 32]))
```

Validated to require identical total buffer size and shard shape. No data
movement.

### Why it matters

Batch-1 decode carries activations as `[1, 1, 32, dim]` in `[32,32]` tiles: one
real row and 31 rows of padding. The matmul does the MAC work for all 32. That
waste is invisible while DRAM time dominates, and becomes the bottleneck the
moment weights get narrower.

Measured on gemma2-9B, 1x P150 (Blackhole), production `dram_sharded` matmul:

| weight dtype | achieved BW | MBU of 550 GB/s |
| --- | --- | --- |
| bfp8 | ~490 GB/s | 89% |
| bfp4 | ~308 GB/s | 55% |

bfp4 halves the bytes but not the time. Since bfp8 reaches 89% on the same
kernel and shapes, this is not a DRAM limit.

`models/demos/deepseek_v3_b1` already solves this with `dram_streaming_matmul`,
which uses `[1,32]` tiny tiles and does 1 row of MAC work instead of 32.
Benchmarked at gemma2 FF1/FF3 shapes (K=3584, N=14336, bfp4 weights, LoFi):

| kernel | m | time | achieved BW |
| --- | --- | --- | --- |
| production `dram_sharded` | 32 | 93.9 us | 308 GB/s |
| `dram_streaming_matmul` | 1 | 63.2 us | 453-500 GB/s |

PCC 0.99 vs golden. Projected +23-30% end-to-end on gemma2-9B decode. This
applies to any batch-1 decode workload on the stack, not just gemma2.

That isolated numbers are meaningful here: the same harness measures the
production kernel at 93.9 us and the in-model profiler measures it at 97.7 us,
a 4% gap.

### What is blocked

`dram_streaming_matmul` needs `in0` as a `[1,32]`-tiled row, replicated per
compute core, height-sharded. Building that from a real decode activation with
device ops only:

| step | result |
| --- | --- |
| `untilize_with_unpadding` -> extract row 0 | works |
| `ttnn.repeat` -> replicate to 8 cores | works |
| `ttnn.tilize` | `Physical shard shape (1, 3584) must be tile {32,32} sized!` |
| `ttnn.copy` into preallocated `[1,32]` buffer | `Input tensor layout (Layout::ROW_MAJOR) must equal output tensor layout (Layout::TILE)` |
| reshard, then `ttnn.copy` | same |
| `ttnn.assign` | same |
| `to_layout(TILE)`, then `ttnn.copy` | `Input and output tensors must have the same tile shape when layout is TILE` |

No device op can produce a `[1,32]`-tiled tensor, and every copy path enforces
tile-shape equality. The only route is host-side `from_torch(tile=...)`, i.e. a
host round-trip per token, which erases the gain.

### Why a view rather than a kernel

A `[1,32]` tile is 2 faces of `[1,16]` = 32 contiguous values, identical to a
row-major row. Measured on device for `[1,1,8,3584]` bf16 height-sharded
`[1,3584]`:

| | row-major | `[1,32]`-tiled |
| --- | --- | --- |
| total bytes | 57344 | 57344 |
| shard shape | (1, 3584) | (1, 3584) |
| pages x page size | 8 x 7168 | 896 x 64 |
| values via `to_torch` | identical | identical |

Same bytes, same order. Only pagination metadata differs, which a re-spec'd view
recomputes. So this needs no kernel and no data movement.

C++ appears to have the pieces already: `Tensor(DeviceStorage)` exists and
`DeviceStorage`'s copy constructor explicitly shares the underlying device
memory. It is not exposed to Python.

### Alternative

If an aliasing view is unacceptable for safety reasons, a `tilize_1x32` copy op
would also unblock us, at some runtime cost. `deepseek_v3_b1/micro_ops/tilize_8x32`
is a working template at 159 lines of Python plus a 23-line kernel.

### Repro

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_tile_alias_equiv.py -s    # byte equivalence
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge.py -s    # tilize blocker
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge2.py -s   # copy-path blockers
python models/demos/deepseek_v3_b1/tests/unit_tests/bench_stream_mm.py             # 453-500 GB/s
```

### Caveats

- The +23-30% assumes isolated matmul numbers translate in-model. The 4%
  agreement on the production kernel supports this but does not prove it;
  proving it needs the prototype, which needs this primitive.
- The streaming kernel's LLK (`kernel_includes/.../custom_mm.h`) restricts `in0`
  tile height to {1,2,4,8}, so running it at m=32 is not an option -- it
  produces garbage, not just slow results.
- Aliasing is a sharp tool and needs spec validation to be safe.

---

## B. Slack version

> Hit a platform blocker on gemma2 decode perf that I think is a small fix, and it
> affects more than just us.
>
> Batch-1 decode pads the activation to a 32-row tile, so our matmuls do 32x the
> arithmetic they need. Invisible at bfp8 (89% MBU) but it's exactly why bfp4 only
> gets 55% MBU instead of the ~2x we should see — bfp4 halves the bytes and
> doesn't halve the time.
>
> DeepSeek already solved this: their `dram_streaming_matmul` uses `[1,32]` tiny
> tiles. I benchmarked it at our FF1/FF3 shapes — 453-500 GB/s vs our 308 GB/s,
> PCC 0.99. Worth ~+23-30% end-to-end on 9B.
>
> I can't use it. ttnn can't produce a `[1,32]`-tiled tensor on device — `tilize`
> rejects it, and `copy`/`assign`/`to_layout` all enforce tile-shape equality. Six
> different routes, all blocked. Only host-side `from_torch(tile=...)` works, which
> means a host round-trip per token and kills the gain.
>
> The annoying bit: I measured it and the conversion is **free**. Row-major and
> `[1,32]`-tiled shards are the same 57344 bytes in the same order — only the
> pagination metadata differs. So this needs no kernel, just a tensor view that
> aliases the buffer with a different tile spec. C++ already has
> `Tensor(DeviceStorage)` and DeviceStorage explicitly shares device memory; it's
> just not exposed to Python.
>
> Ask: expose something like `ttnn.reinterpret_tile(tensor, tile)`. Small, and it
> unlocks bfp4 batch-1 decode for every model on the stack.
>
> Full writeup + repro tests: `models/demos/gemma2/perf_reports/bf4_streaming_matmul_findings.md`
> on `gemma2-bringup`.
