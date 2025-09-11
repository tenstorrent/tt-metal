# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Embedding1D — TTNN Ops Analysis

Module: `models/demos/deepseek_v3/tt/embedding_1d.py`

This document enumerates the TTNN ops executed by Embedding1D in both decode and prefill paths and records the exact non‑tensor arguments (memory_config, program_config, compute_kernel_config, topology) and per‑device shapes derived from the HF config and mesh.

Terminology and symbols
- `H = hf_config.hidden_size`
- `V = hf_config.vocab_size`
- `D = mesh_device.get_num_devices()` with mesh shape `(rows, cols)`; `D = rows × cols`
- Tile size: `T = 32`

Weights
- Storage: DRAM, 1D width‑sharded across mesh columns (replicated across rows).
- Construction path (from `convert_weights`):
  - Start with Torch weight `state_dict["weight"]` of shape `[V, H]`.
  - Reshape to `[V, cols, H / cols]` to expose the mesh‑column split.
  - Convert to TTNN tensor with sharding across the last dimension and mesh columns: `ttnn.as_tensor(torch_weight, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (2, 1)), memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)`.
  - Reshape back to `[V, H / D]` per device shard (remove the helper mesh‑column dimension).
- Per‑device shard shape: `[V, H / D]` in DRAM, tile layout, dtype BF16.

Forward path (prefill and decode)
Ops executed in order:
1) Embedding lookup: `ttnn.embedding(x, **cfg["embedding"])`
   - `cfg["embedding"] = EmbeddingConfig(weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)), memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)`.
   - Output memory: DRAM interleaved; layout: TILE; dtype: BF16.
   - Output logical shape per device: input ids shape appended with `H / D` on the last dim, i.e., `[1, 1, B, H / D]` when `x` is `[1, 1, B]`.

2) Unsqueeze: `ttnn.unsqueeze(embeddings, 0)`
   - Adds a leading dimension for downstream topology; logical shape becomes `[1, 1, 1, B, H / D]`.

3) Typecast: `embeddings_tc = ttnn.typecast(embeddings, **cfg["typecast"])`
   - `cfg["typecast"] = TypecastConfig(dtype=ttnn.float32)`.
   - Converts BF16 → FP32 prior to collective.

4) All‑gather async (width combine across devices): `ttnn.experimental.all_gather_async(embeddings_tc, **cfg["all_gather"])`
   - `cfg["all_gather"] = AllGatherAsyncConfig(mesh_device=MeshDeviceStub(mesh_device.shape), cluster_axis=0, dim=-1, topology=ttnn.Topology.Linear, multi_device_global_semaphore=<from CCL>, barrier_semaphore=<from CCL>, num_links=<from CCL>)`.
   - Topology: `ttnn.Topology.Linear`.
   - Gathers along the last dimension (`dim=-1`) across the cluster axis; concatenates `H / D` shards into full `H`.
   - Output logical shape per device: `[1, 1, 1, B, H]` (post‑gather).
   - Note: `memory_config` is currently `None` in the run config (commented in source until segfault issue resolution, Issue #26672).

5) Temporary buffers are freed via `ttnn.deallocate(...)` at each step as in source.

Decode‑specific padding/slicing
- If `B % T != 0`, decode pads `x` on the sequence dimension up to the next multiple of `T` before the embedding lookup to satisfy current collective constraints. After all‑gather, it slices the sequence back to the original `B`.
- Prefill asserts `B % T == 0` (temporary limitation until sub‑tile gather support is available).

Activation and state details
- The module’s `create_state` provides `multi_device_global_semaphore`, `barrier_semaphore`, and `num_links` for the all‑gather collective via `CCL1D`.
- No explicit program_config or compute_kernel_config are used for the ops above; only memory/layout and topology are set as shown.

Per‑device shapes (logical, before tiling/padding)
- Input ids: `[1, 1, B]` (both prefill and decode), where `B` is batch/sequence length on the device.
- Post‑embedding (pre‑gather): `[1, 1, 1, B, H / D]`.
- Post‑gather: `[1, 1, 1, B, H]`.

Example instantiation (from demo run)
- Mesh: `4 × 8` → `D = 32`.
- HF dims: `H = 7168`, `V = 129280` → `H / D = 224`.
- Weight shard: `[129280, 224]` in DRAM interleaved, tile layout, BF16.
- Run config excerpt reflects: `EmbeddingConfig(..., memory_config=MemoryConfig(layout=INTERLEAVED, buffer=DRAM), layout=TILE)`, `TypecastConfig(dtype=FLOAT32)`, and `AllGatherAsyncConfig(dim=-1, cluster_axis=0, topology=Linear, memory_config=None)`.

Notes
- The collective’s `memory_config` is temporarily `None` in the run config; the source contains a TODO to set a sharded memory config once the all‑gather async segfault is resolved.
- All shapes above are logical; shard shapes used by memory configs will be padded to multiples of `T` when applicable.

