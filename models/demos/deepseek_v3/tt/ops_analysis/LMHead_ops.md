# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# LMHead — TTNN Ops Analysis

Module: `models/demos/deepseek_v3/tt/lm_head.py`

This document enumerates the TTNN ops executed by LMHead in both decode and prefill paths and records the exact non‑tensor arguments (memory_config, program_config, compute_kernel_config, topology) and per‑device shapes derived from the HF config and mesh.

Terminology and symbols
- `H = hf_config.hidden_size`
- `V = hf_config.vocab_size`
- `D = mesh_device.get_num_devices()` (typically `mesh_rows × mesh_cols`)
- Tile size: `T = 32`

Weights
- Storage: DRAM, width‑sharded across DRAM cores.
- Construction path: `ttnn.from_torch(..., dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=dram_sharded_weight_config(H, V/D, mesh_device.dram_grid_size()), mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, -1))`.
- Shape per device shard: `[H, V/D]` (Torch weight transposed to `[in_features=H, out_features=V/D]`).

Decode path
Ops executed in order:
1) Composite scatter: `mesh_scatter(x, mesh_shape=tuple(mesh_device.shape), scatter_idx=(input_row_idx, None))`
   - Underlying TTNN primitives used by the composite helper:
     - `ttnn.to_memory_config` to convert interleaved tensors to L1/DRAM interleaved as required for P2P.
     - Multiple `ttnn.point_to_point(..., topology=ttnn.Topology.Linear, optional_output_tensor=tensor)` across rows/columns to fan‑out from the source mesh coordinate(s).
     - `ttnn.to_memory_config` to restore the original memory layout; `ttnn.copy` to write back; `ttnn.deallocate` for temporaries.
   - Topology: `ttnn.Topology.Linear`.

2) Matmul: `ttnn.linear(x, **cfg["linear"])`
   - Compute kernel: `COMPUTE_KERNEL_CONFIG_LOFI` (LoFi; packer L1 acc enabled).
   - Memory config (output): `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG`.
   - Program config: `get_dram_sharded_matmul_config(M, K, N, input_num_cores, output_num_cores)` where
     - `M = MAX_BATCH_SIZE` (used for scheduling; data can be ≤ `M`).
     - `K = H`.
     - `N = V/D`.
     - `input_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(H, max_num_cores))`.
     - `output_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(V/D, max_num_cores))`.
     - Expands to `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with:
       - `m_tiles = divup(M, T)`
       - `k_tiles = divup(K, T)`
       - `n_tiles = divup(N, T)`
       - `in0_block_w = find_largest_divisor(k_tiles / input_num_cores)`
       - `per_core_M = m_tiles`
       - `per_core_N = n_tiles / output_num_cores`
       - `fused_activation = None`.

3) `ttnn.deallocate(x)`

Input/Output activation memory configs (decode)
- Input: `_get_decode_activation_memory_config(width=H, activation_sharding_num_cores=input_num_cores, mesh_device)` which calls `ttnn.create_sharded_memory_config_` with
  - `shape = (roundup(MAX_BATCH_SIZE, T), roundup(even_int_div(H, input_num_cores), T))`
  - `core_grid = ttnn.num_cores_to_corerangeset(input_num_cores, ttnn.CoreCoord(mesh_device.core_grid.x, mesh_device.core_grid.y), row_wise=True)`
  - `strategy = WIDTH_SHARDED`, `orientation = ROW_MAJOR`, `tile_layout = True`, `use_height_and_width_as_shard_shape = True`.
- Output: same builder but with `width = V/D` and `activation_sharding_num_cores = output_num_cores`.

Prefill path
Ops executed in order:
1) Composite scatter: same `mesh_scatter` invocation/details as decode.

2) Optional chunking for long sequences:
   - If `seq_len > SEQ_LEN_CHUNK_SIZE`, reshape: `x = ttnn.reshape(x, [1, seq_len / SEQ_LEN_CHUNK_SIZE, SEQ_LEN_CHUNK_SIZE, -1])`.

3) Matmul: `ttnn.linear(x, program_config=_get_prefill_pc(seq_len, **cfg["linear_pc_gen"]), **cfg["linear"])`
   - Compute kernel: `COMPUTE_KERNEL_CONFIG_LOFI` (LoFi).
   - Memory config (output): `ttnn.DRAM_MEMORY_CONFIG`.
   - Program config from `_get_prefill_pc(seq_len, hidden_dim=H, vocab_size=V, num_devices=D, core_grid_size=mesh_device.core_grid)`
     - `per_device_in_features = H`
     - `per_device_out_features = V/D`
     - `per_core_M_tiles = divup(seq_len, T * core_grid_size.y)`
     - `K_tiles = divup(H, T)`
     - `per_core_N_tiles = divup(V/D, T * core_grid_size.x)`
     - Returns `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` with:
       - `compute_with_storage_grid_size = core_grid_size`
       - `in0_block_w = find_largest_divisor(K_tiles)`
       - `out_subblock_h = 1`
       - `out_subblock_w = find_largest_divisor(per_core_N_tiles, out_subblock_h)`
       - `per_core_M = per_core_M_tiles`, `per_core_N = per_core_N_tiles`
       - `transpose_mcast = False`, `fused_activation = None`, `fuse_batch = False`.

4) `ttnn.deallocate(x)`

5) De‑chunk output (if chunked): reshape back to `[1, 1, seq_len, V/D]`.

Activation memory configs (prefill)
- Input and output: `ttnn.DRAM_MEMORY_CONFIG`.

Per‑device shapes (logical, before tiling/padding)
- Decode input: `[1, 1, M_decode, H]` (with `M_decode ≤ MAX_BATCH_SIZE`).
- Decode output: `[1, 1, M_decode, V/D]`.
- Prefill input: `[1, 1, S, H]` or `[1, num_chunks, SEQ_LEN_CHUNK_SIZE, H]` when chunked.
- Prefill output: `[1, 1, S, V/D]` or `[1, num_chunks, SEQ_LEN_CHUNK_SIZE, V/D]` when chunked.

Example instantiation (from demo run)
- Mesh: `4 × 8` → `D = 32`.
- HF dims: `H = 7168`, `V = 129280` → `V/D = 4040`.
- Weight shard shape: `[7168, 4040]` in DRAM width‑sharded.
- Decode program config characteristics:
  - `K_tiles = 7168 / 32 = 224` → `input_num_cores` tends to the max divisor ≤ 64, e.g., `56`, giving `in0_block_w = 224 / 56 = 4`.
  - `N_tiles = ceil(4040 / 32) = 127` → prime within ≤ 64, so `output_num_cores = 1`, `per_core_N = 127`.
  - Memory configs: input/output width‑sharded L1 over the selected core ranges.

Notes
- The decode `RunConfig` also contains an `all_gather` entry configured for `Topology.Linear` and DRAM output, but LMHead’s `forward_decode` does not invoke it; the combine step is handled by downstream components.
- All memory configs explicitly use width sharding with row‑major shard orientation and tile layout; shapes are padded to multiples of `T` when building shard shapes.

