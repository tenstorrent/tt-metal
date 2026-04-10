# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.sampling.op import SamplingOp


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
    if k == 1:
        slot_bytes = 16  # bf16 score + uint32 index, padded to 16
        total_bytes = total_slots * slot_bytes
        scratch_width_uint32 = _round_up(total_bytes, 4) // 4
        return (1, scratch_width_uint32)
    else:
        bf16_tile_size = 2 * 32 * 32
        uint32_tile_size = 4 * 32 * 32
        stage1_tiles = (mesh_rows * k + 1023) // 1024
        stage2_tiles = (mesh_cols * k + 1023) // 1024
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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
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
)
@pytest.mark.requires_grid_size(101)
def test_sampling_argmax_mesh_4x2_axis_x(mesh_device, final_mesh_coord, seed, final_core_idx, forced_winner_device_idx):
    """
    Mesh extension test on 4x2 only:
    - final coords constrained away from edge rows (non-torus behavior).
    - per-device local 101-core argmax, then mesh x-axis first reduction.
    """
    grid_size = mesh_device.compute_with_storage_grid_size()
    all_device_cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    active_cores = all_device_cores[:101]
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in active_cores})
    assert 0 <= final_core_idx < len(active_cores), f"final_core_idx={final_core_idx} out of range"
    final_core = active_cores[final_core_idx]

    num_devices = _mesh_num_devices(mesh_device)
    num_cores = len(active_cores)
    scores_shape_per_device = (1, 160 * num_cores)
    input_shard_shape = (1, 160)
    output_shape_per_device = (1, 1)
    scratch_shape_per_device = _mesh_scratch_shape_per_device(mesh_device)
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
    scratch_shard_spec = ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        scratch_shard_spec,
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
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=scratch_mem_config,
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
        fabric_scratch_tensor=ttnn_fabric_scratch,
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


@pytest.mark.parametrize(
    "seed, k, p",
    [
        (42, 32, 0.9),
        (17, 32, 0.5),
        (1337, 32, 1.0),
        (2005, 32, 0.1),
        (999, 32, 0.0),
    ],
)
def test_sampling_topk_topp_golden(seed, k, p):
    """
    Validate the golden reference for top-k + top-p + random sampling.
    Checks that the selected token lives inside the top-k set and inside
    the top-p cumulative probability mass, and that the result is
    deterministic for a given seed.
    """
    torch.manual_seed(seed)
    vocab_size = 16160
    scores = torch.randn((1, vocab_size), dtype=torch.bfloat16)
    indices = torch.arange(vocab_size, dtype=torch.int32).reshape(1, -1)

    result, topk = SamplingOp.golden(scores, indices, k=k, p=p, seed=seed)
    assert result.shape == (1, 1)
    selected_idx = int(result.item())
    assert 0 <= selected_idx < vocab_size

    scores_f32 = scores.float().reshape(-1)
    topk_values, topk_positions = torch.topk(scores_f32, k=k, sorted=True)
    topk_original_indices = indices.reshape(-1)[topk_positions].tolist()
    assert selected_idx in topk_original_indices, f"Selected index {selected_idx} not in top-{k}"

    probs = torch.softmax(topk_values, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    num_kept = int((cum_probs < p).sum().item()) + 1
    num_kept = max(1, min(num_kept, k))
    kept_indices = set(topk_original_indices[:num_kept])
    assert selected_idx in kept_indices, f"Selected index {selected_idx} not in top-p={p} set (kept {num_kept} tokens)"

    result2, _ = SamplingOp.golden(scores, indices, k=k, p=p, seed=seed)
    assert torch.equal(result, result2), "Golden must be deterministic with the same seed"

    logger.info(
        f"Sampling golden test passed: k={k}, p={p}, seed={seed}, " f"selected={selected_idx}, kept_tokens={num_kept}"
    )


def test_sampling_topk_topp_distribution():
    """
    Statistical sanity check: run sampling many times with different seeds
    and verify that (a) every sampled token is inside the top-k set, and
    (b) higher-probability tokens are sampled more often than lower ones.
    """
    vocab_size = 1000
    k = 32
    p = 0.9
    num_samples = 500

    torch.manual_seed(42)
    scores = torch.randn((1, vocab_size), dtype=torch.bfloat16)
    indices = torch.arange(vocab_size, dtype=torch.int32).reshape(1, -1)

    scores_f32 = scores.float().reshape(-1)
    topk_values, topk_positions = torch.topk(scores_f32, k=k, sorted=True)
    topk_indices_set = set(indices.reshape(-1)[topk_positions].tolist())

    probs = torch.softmax(topk_values, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    num_kept = int((cum_probs < p).sum().item()) + 1
    num_kept = max(1, min(num_kept, k))

    counts: dict[int, int] = {}
    for i in range(num_samples):
        result, _ = SamplingOp.golden(scores, indices, k=k, p=p, seed=i)
        idx = int(result.item())
        assert idx in topk_indices_set, f"Sampled index {idx} not in top-{k}"
        counts[idx] = counts.get(idx, 0) + 1

    top1_idx = int(indices.reshape(-1)[topk_positions[0]].item())
    top1_count = counts.get(top1_idx, 0)
    bottom_half_count = sum(
        counts.get(int(indices.reshape(-1)[topk_positions[i]].item()), 0) for i in range(num_kept // 2, num_kept)
    )

    logger.info(
        f"Distribution test: {len(counts)} unique tokens from {num_samples} trials, "
        f"top-1 count={top1_count}, bottom-half count={bottom_half_count}, kept={num_kept}"
    )

    assert top1_count > 0, "Top-1 token should appear at least once in 500 samples"
    assert top1_count > bottom_half_count / max(
        1, num_kept // 2
    ), "Top-1 token should be sampled more often than the average of the bottom half"


def _run_sampling_topk_single_device(device, seed: int, k: int, p: float, temperature: float, final_core_idx: int):
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

    ttnn_result = SamplingOp.op(
        scores_tensor=ttnn_scores,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        k=k,
        p=p,
        temperature=temperature,
        seed=seed,
        rand_output_tensor=ttnn_rand_output,
        final_core_coord=final_core,
        final_mesh_coord=None,
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

    logger.info(
        f"Sampling top-K test passed. seed={seed}, k={k}, "
        f"final_core_idx={final_core_idx}, selected={result_idx}, rand={rand_value}"
    )


@pytest.mark.parametrize(
    "seed, final_core_idx, p, temperature",
    [
        (2005, 100, 0.95, 0.6),
        (17, 0, 0.995, 0.4),
        (1337, 50, 1.0, 0.8),
        (4242, 73, 0.1, 0.6),
    ],ids = ["test_1", "test_2", "test_3", "test_4"],
)
@pytest.mark.requires_grid_size(101)
def test_sampling_topk_single_device(device, seed, p, temperature, final_core_idx):
    """
    Test k=32 top-K sampling path for a single device and 101 cores.

    Scores are rigged so that 32 known positions have a high value (100.0).
    The kernel must select a token from within this set, proving that the
    full pipeline (local top-K, global merge, softmax, temperature) works.
    """
    _run_sampling_topk_single_device(
        device, seed=seed, k=32, p=p, temperature=temperature, final_core_idx=final_core_idx
    )


def _run_sampling_topk_mesh(
    mesh_device, seed: int, k: int, p: float, temperature: float, final_core_idx: int, final_mesh_coord: tuple
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
    winner_global_positions = torch.randperm(global_vocab_size, generator=winner_rng)[:k * num_devices]
    for i, gpos in enumerate(winner_global_positions):
        dev_idx = int(gpos) // vocab_per_device
        local_idx = int(gpos) % vocab_per_device
        torch_scores_all[dev_idx, 0, local_idx] = torch.tensor(100.0 - i, dtype=torch.bfloat16)
    winner_indices = set(torch_indices_all.reshape(-1)[winner_global_positions[:k]].tolist())

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
    scores_scratch_shard_spec = ttnn.ShardSpec(
        final_core_grid, scores_scratch_shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    scores_scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        scores_scratch_shard_spec,
    )
    indices_scratch_shard_spec = ttnn.ShardSpec(
        final_core_grid, indices_scratch_shape, ttnn.ShardOrientation.ROW_MAJOR
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

    global_semaphore = ttnn.create_global_semaphore(mesh_device, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(mesh_device, final_core_grid, 0)
    ttnn.synchronize_device(mesh_device)

    ttnn_result = SamplingOp.op(
        scores_tensor=ttnn_scores,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        k=k,
        p=p,
        temperature=temperature,
        seed=seed,
        rand_output_tensor=ttnn_rand_output,
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
    rand_shards = ttnn.get_device_tensors(ttnn_rand_output)
    final_device_idx = _mesh_device_index(final_mesh_coord, mesh_device)
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

    logger.info(f"Sampling top-K mesh test passed. seed={seed}, k={k}, " f"selected={result_idx}, rand={rand_value}")

def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    "fabric_router_config": create_fabric_router_config(15232),}],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "final_mesh_coord, seed, final_core_idx, p, temperature",
    [
        ((1, 1), 2005, 100, 0.95, 0.6),
        ((1, 0), 52098, 0, 0.995, 0.4),
        ((2, 1), 1337, 50, 1.0, 0.8),
        ((2, 0), 4242, 73, 0.1, 0.6),
        ((0, 0), 999, 0, 1.0, 0.05),
        ((0, 1), 996, 97, 0.8, 0.01),
        ((3, 0), 70, 7, 0.9, 0.6),
        ((3, 1), 5, 39, 0.5, 0.05),
    ],
)
@pytest.mark.requires_grid_size(101)
def test_sampling_topk_mesh(bh_2d_mesh_device, final_mesh_coord, seed, final_core_idx, p, temperature):
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
    logger.debug(f"Testing sampling top-K mesh: seed={seed}, k=32, p={p}, temperature={temperature}, "
                 f"final_core_idx={final_core_idx}, final_mesh_coord={final_mesh_coord}")
    _run_sampling_topk_mesh(
        submesh,
        seed=seed,
        k=32,
        p=p,
        temperature=temperature,
        final_core_idx=final_core_idx,
        final_mesh_coord=final_mesh_coord,
    )
