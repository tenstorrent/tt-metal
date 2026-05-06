# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.sampling.op import SamplingOp
from models.demos.deepseek_v3_b1.utils import float_to_uint32

# ---------------------------------------------------------------------------
# DeepseekMetadata binary layout (see models/demos/deepseek_v3_b1/metadata/metadata.hpp):
#   bytes 0..63   : 13 scalar fields + 3 uint32 padding words (64B header)
#   bytes 64..191 : p_indices[32] (uint32)   -> 128B
#   bytes 192..255: p_scores[32]  (uint16)   ->  64B packed bfloat16
# ---------------------------------------------------------------------------
_METADATA_BYTES = 256
_METADATA_U32_WORDS = _METADATA_BYTES // 4  # 64
_METADATA_P_INDICES_OFFSET = 64
_METADATA_P_SCORES_OFFSET = 192


def _decode_p_metadata(ttnn_metadata, k: int, device_idx: int | None = None):
    """
    Extract (p_indices[:k], p_scores[:k]) from the device-side metadata tensor.

    For a single-device tensor pass device_idx=None.  For a mesh tensor, pass
    the index of the final (producing) device, which selects the right shard.
    The returned `p_scores` is torch.bfloat16 and `p_indices` is torch.int64.
    """
    if device_idx is None:
        meta_torch = ttnn.to_torch(ttnn_metadata)
    else:
        shards = ttnn.get_device_tensors(ttnn_metadata)
        meta_torch = ttnn.to_torch(shards[device_idx])

    meta_bytes = meta_torch.cpu().reshape(-1).numpy().astype(np.uint32).tobytes()
    assert (
        len(meta_bytes) == _METADATA_BYTES
    ), f"metadata tensor must be exactly {_METADATA_BYTES}B; got {len(meta_bytes)}"

    p_indices_np = np.frombuffer(
        meta_bytes[_METADATA_P_INDICES_OFFSET : _METADATA_P_INDICES_OFFSET + 4 * k],
        dtype=np.uint32,
    ).copy()
    p_scores_u16 = np.frombuffer(
        meta_bytes[_METADATA_P_SCORES_OFFSET : _METADATA_P_SCORES_OFFSET + 2 * k],
        dtype=np.uint16,
    ).copy()

    # Reinterpret bf16 bit-pattern as fp32 by promoting to the high 16 bits.
    p_scores_f32 = (p_scores_u16.astype(np.uint32) << 16).view(np.float32).copy()

    return (
        torch.from_numpy(p_indices_np.astype(np.int64)),
        torch.from_numpy(p_scores_f32).to(torch.bfloat16),
    )


def _assert_p_metadata_matches_golden(
    ttnn_metadata,
    *,
    k: int,
    torch_scores: torch.Tensor,
    torch_indices: torch.Tensor,
    p: float,
    temperature: float,
    rand_value: float,
    device_idx: int | None = None,
):
    """Compare kernel-written p_scores/p_indices against the PyTorch golden."""
    _, _, p_scores_golden, p_indices_golden = SamplingOp.golden(
        torch_scores,
        torch_indices,
        k=k,
        p=p,
        temperature=temperature,
        rand_value=rand_value,
        return_p_metadata=True,
    )

    p_indices_kernel, p_scores_kernel = _decode_p_metadata(ttnn_metadata, k=k, device_idx=device_idx)

    logger.info(f"Kernel p_indices[:{k}]: {p_indices_kernel.tolist()}")
    logger.info(f"Golden p_indices[:{k}]: {p_indices_golden.tolist()}")
    logger.info(f"Kernel p_scores[:{k}]: {p_scores_kernel.float().tolist()}")
    logger.info(f"Golden p_scores[:{k}]: {p_scores_golden.float().tolist()}")

    assert p_indices_kernel.tolist() == p_indices_golden.tolist(), (
        f"p_indices mismatch:\n  kernel: {p_indices_kernel.tolist()}\n" f"  golden: {p_indices_golden.tolist()}"
    )
    rtol = 2e-2
    atol = 2e-3
    assert torch.allclose(p_scores_kernel.float(), p_scores_golden.float(), rtol=rtol, atol=atol), (
        f"p_scores not allclose at rtol={rtol}, atol={atol}:\n  kernel: {p_scores_kernel.float().tolist()}\n"
        f"  golden: {p_scores_golden.float().tolist()}, max_abs_error: {torch.max(torch.abs(p_scores_kernel.float() - p_scores_golden.float()))}, max_rel_error: {torch.max(torch.abs(p_scores_kernel.float() - p_scores_golden.float()) / p_scores_golden.float())}"
    )


def _mesh_shape(mesh_device):
    mesh_rows, mesh_cols = mesh_device.shape
    return int(mesh_rows), int(mesh_cols)


def _mesh_num_devices(mesh_device):
    mesh_rows, mesh_cols = _mesh_shape(mesh_device)
    return mesh_rows * mesh_cols


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _mesh_scratch_shape_per_device(mesh_device, k: int = 1):
    mesh_rows, mesh_cols = _mesh_shape(mesh_device)
    total_slots = mesh_rows + mesh_cols
    topk_min_alignment = 32
    bf16_tile_size = 2 * 32 * 32
    uint32_tile_size = 4 * 32 * 32
    stage1_tiles = (mesh_rows * topk_min_alignment + 1023) // 1024
    stage2_tiles = (mesh_cols * topk_min_alignment + 1023) // 1024
    total_tiles = stage1_tiles + stage2_tiles
    scores_bytes = total_tiles * bf16_tile_size
    indices_bytes = total_tiles * uint32_tile_size
    scores_width_bf16 = _round_up(scores_bytes, 2) // 2
    indices_width_uint32 = _round_up(indices_bytes, 4) // 4
    return (1, scores_width_bf16), (1, indices_width_uint32)


def _mesh_device_index(final_mesh_coord, mesh_device):
    _, mesh_cols = _mesh_shape(mesh_device)
    return int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])


def _run_sampling_argmax_single_device_101_cores(device, seed: int, final_core_idx: int):
    grid_size = device.compute_with_storage_grid_size()
    all_device_cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    active_cores = all_device_cores[:101]
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in active_cores})
    assert 0 <= final_core_idx < len(active_cores), f"final_core_idx={final_core_idx} out of range"
    final_core = active_cores[final_core_idx]

    num_cores = len(active_cores)
    scores_shape = (1, 160 * num_cores)
    input_shard_shape = (1, 160)
    output_shape = (1, 1)
    tile_1x32 = ttnn.Tile([1, 32])

    logger.info(
        f"Testing sampling argmax: single-device/101-cores, seed={seed}, final_core_idx={final_core_idx}, "
        "160 values per core"
    )

    torch.manual_seed(seed)
    torch_scores = torch.randn(scores_shape, dtype=torch.bfloat16)
    torch_indices = torch.arange(scores_shape[1], dtype=torch.int32).reshape(scores_shape)

    torch_expected_idx, _ = SamplingOp.golden(torch_scores, torch_indices, k=1, p=1.0)

    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    final_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(final_core, final_core)})
    output_shard_spec = ttnn.ShardSpec(
        final_core_grid,
        output_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )
    ttnn_scores = ttnn.from_torch(
        torch_scores,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile_1x32,
    )

    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    ttnn_output_index = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=output_mem_config,
    )

    ttnn_result = SamplingOp.op(
        scores_tensor=ttnn_scores,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        k=1,
        p=1.0,
        final_core_coord=final_core,
        final_mesh_coord=None,
    )

    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == output_shape, f"Expected output shape {output_shape}, got {output_torch.shape}"
    logger.info(f"Golden output: {output_torch}")
    logger.info(f"Golden expected index: {torch_expected_idx}")
    assert torch.equal(
        output_torch.to(torch.uint32), torch_expected_idx
    ), f"Argmax index mismatch. expected={torch_expected_idx.item()}, got={output_torch.item()}"

    logger.info(
        f"Sampling argmax test passed. seed={seed}, final_core_idx={final_core_idx}, index={int(output_torch.item())}"
    )


@pytest.mark.parametrize(
    "seed, final_core_idx",
    [
        (2005, 100),  # last active core (original behavior)
        (17, 0),  # first active core
        (1337, 50),  # middle active core
        (4242, 73),  # non-boundary core
    ],
)
@pytest.mark.requires_grid_size(101)
def test_sampling_argmax_single_device_101_cores(device, seed, final_core_idx):
    """
    Test k=1 sampling (argmax path) for a single device and 101 cores.
    Covers multiple random seeds and different final-core placements.
    """
    _run_sampling_argmax_single_device_101_cores(device, seed=seed, final_core_idx=final_core_idx)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "final_mesh_coord,seed,final_core_idx,forced_winner_device_idx",
    [
        ((1, 1), 2005, 100, None),  # pure random case (tie-break behavior coverage)
        ((1, 0), 52098, 0, 3),  # force winner off device 0
        ((2, 1), 1337, 50, 5),  # force winner off device 0
        ((2, 0), 4242, 73, 7),  # force winner off device 0
    ],
    ids=["test_1", "test_2", "test_3", "test_4"],
)
@pytest.mark.requires_grid_size(101)
def test_sampling_argmax_mesh(bh_2d_mesh_device, final_mesh_coord, seed, final_core_idx, forced_winner_device_idx):
    """
    Mesh extension test on 4x2 only:
    - final coords constrained away from edge rows (non-torus behavior).
    - per-device local 101-core argmax, then mesh x-axis first reduction.
    """
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    mesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    grid_size = mesh_device.compute_with_storage_grid_size()
    all_device_cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    active_cores = all_device_cores[:101]
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in active_cores})
    assert 0 <= final_core_idx < len(active_cores), f"final_core_idx={final_core_idx} out of range"
    final_core = active_cores[final_core_idx]
    logger.debug(f"Final core: {final_core}")
    logger.debug(f"Final mesh coord: {final_mesh_coord}")
    logger.debug(f"Active cores: {active_cores}")

    num_devices = _mesh_num_devices(mesh_device)
    num_cores = len(active_cores)
    scores_shape_per_device = (1, 160 * num_cores)
    input_shard_shape = (1, 160)
    output_shape_per_device = (1, 1)
    scores_scratch_shape, indices_scratch_shape = _mesh_scratch_shape_per_device(mesh_device)
    tile_1x32 = ttnn.Tile([1, 32])

    logger.info(
        "Testing sampling argmax mesh(4x2): "
        f"seed={seed}, final_core_idx={final_core_idx}, final_core_coord={final_core}, final_mesh_coord={final_mesh_coord}"
    )
    torch.manual_seed(seed)

    torch_scores_all = torch.randn((num_devices, *scores_shape_per_device), dtype=torch.bfloat16)
    if forced_winner_device_idx is not None:
        assert 0 <= forced_winner_device_idx < num_devices, "forced_winner_device_idx out of range"
        winner_local_idx = (seed * 9973 + final_core_idx) % scores_shape_per_device[1]
        # Overwrite (not add) to deterministically create a unique global winner.
        torch_scores_all[forced_winner_device_idx, 0, winner_local_idx] = torch.tensor(10.0, dtype=torch.bfloat16)
        logger.info(f"Forced winner on device {forced_winner_device_idx}, local index {winner_local_idx}")

    torch_indices_all = torch.arange(num_devices * scores_shape_per_device[1], dtype=torch.int32).reshape(
        num_devices, *scores_shape_per_device
    )
    torch_expected_idx, _ = SamplingOp.golden(
        torch_scores_all.reshape(1, -1), torch_indices_all.reshape(1, -1), k=1, p=1.0
    )
    if forced_winner_device_idx is not None:
        expected_device_idx = int(torch_expected_idx.item()) // scores_shape_per_device[1]
        assert (
            expected_device_idx == forced_winner_device_idx
        ), f"Expected winner on device {forced_winner_device_idx}, got device {expected_device_idx}"

    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    final_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(final_core, final_core)})
    output_shard_spec = ttnn.ShardSpec(final_core_grid, output_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )
    scores_scratch_shard_spec = ttnn.ShardSpec(final_core_grid, scores_scratch_shape, ttnn.ShardOrientation.ROW_MAJOR)
    indices_scratch_shard_spec = ttnn.ShardSpec(final_core_grid, indices_scratch_shape, ttnn.ShardOrientation.ROW_MAJOR)
    scores_scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        scores_scratch_shard_spec,
    )
    indices_scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        indices_scratch_shard_spec,
    )
    mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    ttnn_scores = ttnn.from_torch(
        torch_scores_all,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=input_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=input_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, *output_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=output_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_scores_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scores_scratch_shape), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=scores_scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_indices_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *indices_scratch_shape), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=indices_scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(mesh_device, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(mesh_device, final_core_grid, 0)
    ttnn.synchronize_device(mesh_device)

    ttnn_result = SamplingOp.op(
        scores_tensor=ttnn_scores,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        k=1,
        p=1.0,
        final_core_coord=final_core,
        final_mesh_coord=final_mesh_coord,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        scores_scratch_tensor=ttnn_scores_scratch,
        indices_scratch_tensor=ttnn_indices_scratch,
        mesh_axis="x",
    )
    ttnn.synchronize_device(mesh_device)

    output_shards = ttnn.get_device_tensors(ttnn_result)
    final_device_idx = _mesh_device_index(final_mesh_coord, mesh_device)
    final_output_torch = ttnn.to_torch(output_shards[final_device_idx])
    final_output_index = final_output_torch.to(torch.uint32).reshape(1, 1)
    logger.info(f"Final output index: {final_output_index}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={int(final_output_index.item())}"


def _build_metadata_tensor(device, final_core, k: int, p: float, temperature: float):
    """
    Build a single-core L1 tensor matching the `DeepseekMetadata` struct layout
    (see models/demos/deepseek_v3_b1/metadata/metadata.hpp).

    The struct is 256B total:
      - 13 leading scalar fields (52B) + 3 uint32 padding (12B) = 64B header
      - p_indices[32] uint32 (128B)
      - p_scores[32]  uint16 (64B, packed bfloat16)

    We pack everything into a 1x64 uint32 tensor (256B) with the sampling-
    relevant fields at indices 10/11/12. Remaining words are zeroed so the
    test can predict what the kernel will overwrite.
    """
    metadata_words = torch.zeros((1, _METADATA_U32_WORDS), dtype=torch.uint32)
    metadata_words[0, 10] = float_to_uint32(temperature)
    metadata_words[0, 11] = int(k)
    metadata_words[0, 12] = float_to_uint32(p)

    final_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(final_core, final_core)})
    metadata_shard_spec = ttnn.ShardSpec(final_core_grid, (1, _METADATA_U32_WORDS), ttnn.ShardOrientation.ROW_MAJOR)
    metadata_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        metadata_shard_spec,
    )
    return ttnn.from_torch(
        metadata_words,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=metadata_mem_config,
    )


def _run_sampling_topk_single_device(
    device,
    seed: int,
    k: int,
    p: float,
    temperature: float,
    final_core_idx: int,
    num_internal_iterations: int,
    from_metadata: bool = False,
    copy_probabilities: bool = False,
):
    """
    Run the top-K sampling kernel (k>1 path) on a single device with rigged
    scores: 32 random vocabulary positions are set to distinct high values so
    they are guaranteed to be the global top-32.  The kernel returns the random
    value it used, enabling exact host-side golden verification.
    """
    grid_size = device.compute_with_storage_grid_size()
    all_device_cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    active_cores = all_device_cores[:101]
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in active_cores})
    assert 0 <= final_core_idx < len(active_cores), f"final_core_idx={final_core_idx} out of range"
    final_core = active_cores[final_core_idx]
    logger.info(f"Final core: {final_core}")
    logger.debug(f"Active cores: {active_cores}")

    num_cores = len(active_cores)
    scores_shape = (1, 160 * num_cores)
    vocab_size = scores_shape[1]
    input_shard_shape = (1, 160)
    output_shape = (1, 1)
    tile_1x32 = ttnn.Tile([1, 32])

    logger.info(
        f"Testing sampling top-K: single-device/101-cores, seed={seed}, k={k}, "
        f"final_core_idx={final_core_idx}, 160 values per core"
    )

    torch.manual_seed(seed)
    torch_scores = torch.randn(scores_shape, dtype=torch.bfloat16)
    torch_indices = torch.arange(vocab_size, dtype=torch.int32).reshape(scores_shape)

    winner_rng = torch.Generator().manual_seed(seed)
    winner_positions = torch.randperm(vocab_size, generator=winner_rng)[:k].sort().values
    for i, pos in enumerate(winner_positions):
        torch_scores[0, pos] = torch.tensor(100.0 - i, dtype=torch.bfloat16)
    winner_indices = set(torch_indices[0, winner_positions].tolist())

    _, golden_topk = SamplingOp.golden(torch_scores, torch_indices, k=k, p=p, temperature=temperature)
    golden_topk_set = set(golden_topk.reshape(-1).tolist())
    assert golden_topk_set == winner_indices, (
        f"Golden top-{k} should match rigged winners.\n"
        f"  Golden: {sorted(golden_topk_set)}\n"
        f"  Rigged: {sorted(winner_indices)}"
    )
    logger.info(f"Rigged {k} winner positions (first 5): {winner_positions[:k].tolist()}...")

    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    final_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(final_core, final_core)})
    output_shard_spec = ttnn.ShardSpec(
        final_core_grid,
        output_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )
    rand_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, output_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_scores = ttnn.from_torch(
        torch_scores,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile_1x32,
    )

    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    ttnn_output_index = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=output_mem_config,
    )

    ttnn_rand_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=rand_output_mem_config,
    )

    ttnn_metadata = None
    # copy_probabilities requires the metadata tensor, so build one automatically
    # if the caller asked for probability copy-out but not explicit metadata input.
    if from_metadata or copy_probabilities:
        ttnn_metadata = _build_metadata_tensor(device, final_core, k=k, p=p, temperature=temperature)
        logger.info(
            f"Metadata tensor populated: k={k}, p={p}, temperature={temperature}, "
            f"l1_addr=0x{ttnn_metadata.buffer_address():x}"
        )

    ttnn_result = SamplingOp.op(
        scores_tensor=ttnn_scores,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        k=k if not from_metadata else 32,
        p=p if not from_metadata else 1.0,
        temperature=temperature if not from_metadata else 0.6,
        seed=seed,
        rand_output_tensor=ttnn_rand_output,
        final_core_coord=final_core,
        final_mesh_coord=None,
        num_internal_iterations=num_internal_iterations,
        metadata_output_tensor=ttnn_metadata,
        copy_probabilities=copy_probabilities,
    )

    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == output_shape, f"Expected output shape {output_shape}, got {output_torch.shape}"
    result_idx = int(output_torch.to(torch.uint32).item())

    rand_torch = ttnn.to_torch(ttnn_rand_output)
    rand_value = rand_torch.float().item()
    logger.info(f"Kernel selected index: {result_idx}, rand_value: {rand_value}")

    golden_idx, golden_topk = SamplingOp.golden(
        torch_scores, torch_indices, k=k, p=p, temperature=temperature, rand_value=rand_value
    )
    golden_selected = int(golden_idx.to(torch.uint32).item())
    logger.info(f"Golden selected index: {golden_selected}")

    assert result_idx in winner_indices, (
        f"Selected index {result_idx} is not in the rigged top-{k} set.\n" f"  Rigged winners: {sorted(winner_indices)}"
    )
    assert result_idx == golden_selected, (
        f"Kernel selected {result_idx} but golden selected {golden_selected} " f"(rand_value={rand_value})"
    )

    if copy_probabilities:
        _assert_p_metadata_matches_golden(
            ttnn_metadata,
            k=k,
            torch_scores=torch_scores,
            torch_indices=torch_indices,
            p=p,
            temperature=temperature,
            rand_value=rand_value,
        )

    logger.info(
        f"Sampling top-K test passed. seed={seed}, k={k}, "
        f"final_core_idx={final_core_idx}, selected={result_idx}, rand={rand_value}"
    )


@pytest.mark.parametrize(
    "seed, final_core_idx, p, temperature, num_internal_iterations, k, from_metadata, copy_probabilities",
    [
        (2005, 100, 0.95, 0.6, 100, 32, True, True),
        (17, 0, 0.995, 0.4, 1, 32, True, True),
        # (1337, 50, 1.0, 0.8, 1, 32, True, True), test 3 skipped due to small tail end precision issue
        (4242, 73, 0.1, 0.6, 1, 32, True, True),
        (52098, 100, 0.95, 0.6, 100, 1, True, True),
        (52098, 100, 1.0, 10, 1, 16, True, True),
    ],
    ids=["test_1", "test_2", "test_4", "test_5", "test_6"],
)
@pytest.mark.requires_grid_size(101)
def test_sampling_topk_single_device(
    device, seed, p, temperature, final_core_idx, num_internal_iterations, k, from_metadata, copy_probabilities
):
    # skip test_3
    """
    Test k=32 top-K sampling path for a single device and 101 cores.

    Scores are rigged so that 32 known positions have a high value (100.0).
    The kernel must select a token from within this set, proving that the
    full pipeline (local top-K, global merge, softmax, temperature) works.
    """
    _run_sampling_topk_single_device(
        device,
        seed=seed,
        k=k,
        p=p,
        temperature=temperature,
        final_core_idx=final_core_idx,
        num_internal_iterations=num_internal_iterations,
        from_metadata=from_metadata,
        copy_probabilities=copy_probabilities,
    )


def _run_sampling_topk_mesh(
    mesh_device,
    seed: int,
    k: int,
    p: float,
    temperature: float,
    final_core_idx: int,
    final_mesh_coord: tuple,
    from_metadata: bool = False,
    copy_probabilities: bool = False,
    iterations: int = 1,
    rigged: bool = True,
):
    """
    Run the top-K sampling kernel on a multi-device mesh with rigged scores.

    32 global winners are chosen at random across the entire vocabulary
    (spanning all devices).  Each winner gets a distinct high score
    (100, 99, …, 69) so the global top-32 is deterministic regardless of
    which devices they land on.  The kernel returns its random value,
    enabling exact golden verification.
    """
    grid_size = mesh_device.compute_with_storage_grid_size()
    all_device_cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    active_cores = all_device_cores[:101]
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in active_cores})
    assert 0 <= final_core_idx < len(active_cores), f"final_core_idx={final_core_idx} out of range"
    final_core = active_cores[final_core_idx]

    logger.info(f"Final mesh coord: {final_mesh_coord}")
    logger.info(f"Final core: {final_core}")
    logger.debug(f"Active cores: {active_cores}")

    num_devices = _mesh_num_devices(mesh_device)
    num_cores = len(active_cores)
    scores_shape_per_device = (1, 160 * num_cores)
    vocab_per_device = scores_shape_per_device[1]
    global_vocab_size = num_devices * vocab_per_device
    input_shard_shape = (1, 160)
    output_shape_per_device = (1, 1)
    scores_scratch_shape, indices_scratch_shape = _mesh_scratch_shape_per_device(mesh_device, k=k)
    tile_1x32 = ttnn.Tile([1, 32])

    logger.info(
        f"Testing sampling top-K mesh: seed={seed}, k={k}, p={p}, temperature={temperature}, "
        f"final_core_idx={final_core_idx}, final_mesh_coord={final_mesh_coord}"
    )
    torch.manual_seed(seed)

    torch_scores_all = torch.randn((num_devices, *scores_shape_per_device), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(global_vocab_size, dtype=torch.int32).reshape(
        num_devices, *scores_shape_per_device
    )

    winner_rng = torch.Generator().manual_seed(seed)
    winner_global_positions = torch.randperm(global_vocab_size, generator=winner_rng)[: k * num_devices]
    for i, gpos in enumerate(winner_global_positions):
        dev_idx = int(gpos) // vocab_per_device
        local_idx = int(gpos) % vocab_per_device
        torch_scores_all[dev_idx, 0, local_idx] = torch.tensor(100.0 - i, dtype=torch.bfloat16)
    winner_indices = set(torch_indices_all.reshape(-1)[winner_global_positions[:k]].tolist())
    logger.info(f"All rigged global positions: {winner_global_positions}")
    logger.info(f"Rigged {k} winners: {winner_indices}")

    _, golden_topk = SamplingOp.golden(
        torch_scores_all.reshape(1, -1),
        torch_indices_all.reshape(1, -1),
        k=k,
        p=p,
        temperature=temperature,
    )
    golden_topk_set = set(golden_topk.reshape(-1).tolist())
    assert golden_topk_set == winner_indices, (
        f"Golden top-{k} should match rigged winners.\n"
        f"  Golden: {sorted(golden_topk_set)}\n"
        f"  Rigged: {sorted(winner_indices)}"
    )

    devices_with_winners = sorted({int(gpos) // vocab_per_device for gpos in winner_global_positions})
    logger.info(f"Rigged {k} winners spread across devices: {devices_with_winners}")

    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    final_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(final_core, final_core)})
    output_shard_spec = ttnn.ShardSpec(final_core_grid, output_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )
    rand_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, output_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )
    scores_scratch_shard_spec = ttnn.ShardSpec(final_core_grid, scores_scratch_shape, ttnn.ShardOrientation.ROW_MAJOR)
    scores_scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        scores_scratch_shard_spec,
    )
    indices_scratch_shard_spec = ttnn.ShardSpec(final_core_grid, indices_scratch_shape, ttnn.ShardOrientation.ROW_MAJOR)
    indices_scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        indices_scratch_shard_spec,
    )

    mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    ttnn_scores = ttnn.from_torch(
        torch_scores_all,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=input_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=input_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, *output_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=output_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_rand_output = ttnn.from_torch(
        torch.zeros((num_devices, *output_shape_per_device), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=rand_output_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_scores_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scores_scratch_shape), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=scores_scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_indices_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *indices_scratch_shape), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=indices_scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphores = [ttnn.create_global_semaphore(mesh_device, final_core_grid, 0) for _ in range(iterations)]
    global_stage2_semaphores = [
        ttnn.create_global_semaphore(mesh_device, final_core_grid, 0) for _ in range(iterations)
    ]
    ttnn.synchronize_device(mesh_device)

    ttnn_metadata = None
    if from_metadata or copy_probabilities:
        metadata_words_per_device = torch.zeros((num_devices, 1, _METADATA_U32_WORDS), dtype=torch.uint32)
        metadata_words_per_device[:, 0, 10] = float_to_uint32(temperature)
        metadata_words_per_device[:, 0, 11] = int(k)
        metadata_words_per_device[:, 0, 12] = float_to_uint32(p)
        metadata_shard_spec = ttnn.ShardSpec(final_core_grid, (1, _METADATA_U32_WORDS), ttnn.ShardOrientation.ROW_MAJOR)
        metadata_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            metadata_shard_spec,
        )
        ttnn_metadata = ttnn.from_torch(
            metadata_words_per_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=metadata_mem_config,
            mesh_mapper=mesh_mapper,
        )
        logger.info(
            f"Metadata tensor populated (replicated across {num_devices} devices): "
            f"k={k}, p={p}, temperature={temperature}, l1_addr=0x{ttnn_metadata.buffer_address():x}"
        )
    ttnn_results = []
    torch_metadata_results = []
    final_device_idx = _mesh_device_index(final_mesh_coord, mesh_device)
    for i in range(iterations):
        ttnn_result = SamplingOp.op(
            scores_tensor=ttnn_scores,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            k=k if not from_metadata else 32,
            p=p if not from_metadata else 1.0,
            temperature=temperature if not from_metadata else 0.6,
            seed=seed,
            rand_output_tensor=ttnn_rand_output,
            final_core_coord=final_core,
            final_mesh_coord=final_mesh_coord,
            global_semaphore=global_semaphores[i],
            global_stage2_semaphore=global_stage2_semaphores[i],
            scores_scratch_tensor=ttnn_scores_scratch,
            indices_scratch_tensor=ttnn_indices_scratch,
            mesh_axis="x",
            metadata_output_tensor=ttnn_metadata,
            copy_probabilities=copy_probabilities,
        )
        # append torch_metadata_results at each iteration
        ttnn_results.append(ttnn_result)
        ttnn.synchronize_device(mesh_device)
        metadata_result = ttnn.get_device_tensors(ttnn_metadata)[final_device_idx]
        torch_metadata_results.append(ttnn.to_torch(metadata_result))
    ttnn.synchronize_device(mesh_device)
    # verify all results are the same
    ttnn_first_result = ttnn.get_device_tensors(ttnn_result)
    torch_first_metadata = torch_metadata_results[0]
    torch_first_results = [ttnn.to_torch(shard) for shard in ttnn_first_result]
    for i in range(1, iterations):
        ttnn_result = ttnn.get_device_tensors(ttnn_results[i])
        torch_result = [ttnn.to_torch(shard) for shard in ttnn_result]
        for j in range(len(torch_result)):
            logger.info(
                f"Iteration {i} shard {j} result: {torch_result[j].tolist()} indices: {torch_metadata_results[i].tolist()[16:48]} scores: {torch_metadata_results[i].tolist()[48:]}"
            )
            assert torch.equal(
                torch_result[j], torch_first_results[j]
            ), f"Iteration {i} shard {j} result does not match iteration 0"
            assert torch.equal(
                torch_metadata_results[i], torch_first_metadata
            ), f"Iteration {i} metadata does not match iteration 0"

    ttnn_result = ttnn_results[0]

    output_shards = ttnn.get_device_tensors(ttnn_result)
    rand_shards = ttnn.get_device_tensors(ttnn_rand_output)
    final_output_torch = ttnn.to_torch(output_shards[final_device_idx])
    result_idx = int(final_output_torch.to(torch.uint32).item())

    rand_torch = ttnn.to_torch(rand_shards[final_device_idx])
    rand_value = rand_torch.float().item()
    logger.info(f"Kernel selected index: {result_idx}, rand_value: {rand_value}")

    golden_idx, _ = SamplingOp.golden(
        torch_scores_all.reshape(1, -1),
        torch_indices_all.reshape(1, -1),
        k=k,
        p=p,
        temperature=temperature,
        rand_value=rand_value,
    )
    golden_selected = int(golden_idx.to(torch.uint32).item())
    logger.info(f"Golden selected index: {golden_selected}")

    assert result_idx in winner_indices, (
        f"Selected index {result_idx} is not in the rigged top-{k} set.\n" f"  Rigged winners: {sorted(winner_indices)}"
    )
    assert result_idx == golden_selected, (
        f"Kernel selected {result_idx} but golden selected {golden_selected} " f"(rand_value={rand_value})"
    )

    if copy_probabilities:
        _assert_p_metadata_matches_golden(
            ttnn_metadata,
            k=k,
            torch_scores=torch_scores_all.reshape(1, -1),
            torch_indices=torch_indices_all.reshape(1, -1),
            p=p,
            temperature=temperature,
            rand_value=rand_value,
            device_idx=final_device_idx,
        )

    logger.info(f"Sampling top-K mesh test passed. seed={seed}, k={k}, " f"selected={result_idx}, rand={rand_value}")


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "final_mesh_coord, seed, final_core_idx, p, temperature, k, from_metadata, copy_probabilities, iterations",
    [
        ((1, 1), 2005, 100, 0.95, 2.0, 32, True, True, 100),
        ((1, 0), 52098, 0, 0.995, 0.4, 16, True, True, 1),
        ((2, 1), 1337, 50, 1.0, 10.0, 32, True, True, 1),
        ((2, 0), 4242, 73, 0.1, 0.6, 32, True, True, 1),
        ((0, 0), 999, 0, 1.0, 0.05, 32, True, True, 1),
        ((0, 1), 996, 97, 0.8, 50.0, 9, True, True, 1),
        ((3, 0), 70, 7, 0.9, 22.0, 1, True, True, 1),
        ((3, 1), 5, 39, 0.5, 6.0, 1, True, True, 1),
    ],
    ids=["test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "test_7", "test_8"],
)
@pytest.mark.requires_grid_size(101)
def test_sampling_topk_mesh(
    bh_2d_mesh_device,
    final_mesh_coord,
    seed,
    final_core_idx,
    p,
    temperature,
    k,
    from_metadata,
    copy_probabilities,
    iterations,
):
    """
    Mesh extension test for k=32 top-K sampling on a 4x2 mesh.

    Each device performs local top-32, then mesh stages merge across devices
    before the final device runs softmax + top-P + random selection.
    Scores are rigged so 32 global winners are spread across devices.
    """
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    logger.debug(
        f"Testing sampling top-K mesh: seed={seed}, k=32, p={p}, temperature={temperature}, "
        f"final_core_idx={final_core_idx}, final_mesh_coord={final_mesh_coord}"
    )
    _run_sampling_topk_mesh(
        submesh,
        seed=seed,
        k=k,
        p=p,
        temperature=temperature,
        final_core_idx=final_core_idx,
        final_mesh_coord=final_mesh_coord,
        from_metadata=from_metadata,
        copy_probabilities=copy_probabilities,
        iterations=iterations,
    )
