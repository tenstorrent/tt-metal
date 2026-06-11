# TTNN Collective Communication Operations (CCL)

User-facing documentation for TTNN's multi-chip collective operations, structured after [NCCL's collectives guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html). Where NCCL operates on ranks over CUDA devices, TTNN collectives operate on a `MeshDevice` over TT-Fabric; the device count and layout come from the mesh shape, and "rank" corresponds to the device's mesh coordinate.

## Prerequisites — the TTNN equivalent of NCCL communicator setup

```python
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)   # before open; FABRIC_1D_RING on T3K
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))    # n300: (1,2); T3K: (2,4); Galaxy: (8,4)
```

Rules (each can fail silently otherwise):
- `set_fabric_config()` must run before `open_mesh_device`; never switch configs on an open mesh.
- One mesh held for the whole CCL phase. Inputs distributed via `mesh_mapper`, TILE_LAYOUT, bf16/bf8_b preferred (ROW_MAJOR + bfloat8_b is unsupported; ROW_MAJOR/fp32 is unsafe at Galaxy scale).
- Async/experimental ops additionally need persistent output buffers, `ttnn.create_global_semaphore(...)`, and `ttnn.synchronize_device(mesh)` before results are read.
- `cluster_axis` selects a mesh axis: every row/column performs an independent collective. Omitted = whole mesh.

## Stable collectives

### AllGather — `ttnn.all_gather(input, dim, *, cluster_axis=None, num_links=None, topology=None, memory_config=None)`

Each device starts with a shard; ends with all shards concatenated along `dim` (out = concat over devices of in_i). NCCL AllGather (in-place opposite: NCCL concatenates by rank; here by mesh coordinate along `cluster_axis`).

```
dev0: [A]  dev1: [B]  dev2: [C]  dev3: [D]   →   every device: [A B C D]
```

ROW_MAJOR or tile-padding on `dim` silently falls back to composite (all-broadcast based, slower).

### ReduceScatter — `ttnn.reduce_scatter(input, dim, *, cluster_axis=None, ...)`

Elementwise Sum across devices, result scattered along `dim` (device i keeps slice i). NCCL ReduceScatter; equally rank-order sensitive.

```
all devices: [x0 x1 x2 x3]   →   dev_i: [Σ x_i]
```

### AllReduce — `ttnn.all_reduce(input, *, cluster_axis=None, num_links=None, topology=None, ...)`

Sum across devices; identical full tensor everywhere. **Sum only** — unlike NCCL there is no op argument; min/max/prod via all_gather + local reduce (the experimental signature's `math_op` is ignored).

### Broadcast — `ttnn.broadcast(input, sender_coord, cluster_axis, mesh_device)` ≈ NCCL Broadcast; root = `MeshCoordinate`.

### AllBroadcast — `ttnn.all_broadcast(input, ...)` — every device returns a list of all devices' tensors (no NCCL analog; substrate for composite paths).

### Send/Recv — `ttnn.point_to_point(input, sender_coord, receiver_coord, topology, ...)`

### AlltoAll — `ttnn.all_to_all_dispatch` / `ttnn.all_to_all_combine` — token routing (MoE); generic AlltoAll only experimental.

NCCL Reduce/Gather/Scatter have no built-ins — compose them.

## Experimental tier

`ttnn.experimental.*` (`all_gather_async`, `all_reduce_async`, `reduce_scatter_minimal_async`, fusions): higher perf, caller-managed semaphores/buffers, persistent intermediates, signature drift; production sequencing in `llama3_70b_galaxy/tt/llama_ccl.py`. Tune `num_links` (WH 1–4, BH 1–2), `topology` Ring vs Linear.

## Pitfalls
all_reduce is Sum-only · TILE+bf16 at Galaxy scale · composite fallback hides slowness · semaphore reuse across cached runs · stale semaphores survive process exit · don't mix MeshDevice + single-device handles.

References: TT-Fabric report; Mesh-of-Devices report; `tests/ttnn/unit_tests/operations/ccl/`.
