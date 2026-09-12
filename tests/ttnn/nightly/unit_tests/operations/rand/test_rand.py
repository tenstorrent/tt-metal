# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn


DEFAULT_SHAPE = (32, 32)
SHAPES = [tuple([32] * i) for i in range(6)]
SUPPORTED_DTYPES = (ttnn.bfloat16, ttnn.float32)
LEGACY_DTYPES = (ttnn.bfloat4_b, ttnn.bfloat8_b)
LEGACY_INTEGER_DTYPES = (
    (ttnn.int8, torch.int8, -100, 100),
    (ttnn.int32, torch.int32, -100, 100),
    (ttnn.uint16, torch.uint16, 0, 100),
    (ttnn.uint32, torch.uint32, 0, 100),
)
UNSUPPORTED_DTYPES = (ttnn.uint8, ttnn.fp8_e4m3, ttnn.DataType.INVALID)
TEST_SEED = 17
FLOAT32_MIN_NORMAL = torch.finfo(torch.float32).tiny
FLOAT32_MIN_SUBNORMAL = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item()
FLOAT32_MIN_NORMAL_PLUS_TWO_ULPS = torch.nextafter(
    torch.nextafter(torch.tensor(FLOAT32_MIN_NORMAL), torch.tensor(math.inf)), torch.tensor(math.inf)
).item()


def check_uniform_distribution(data, value_range=(0, 1)):
    n = data.numel()

    if n < 1000:
        print("[Warning] A meaningful analysis requires at least 1000 samples.")
        if n < 2:
            print("[Error] Cannot perform test with less than 2 data points.")
            return False

    start_value, end_value = value_range
    if torch.any(data < start_value) or torch.any(data >= end_value):
        return False

    # torch ops don't support integer data types, convert to list
    data = data.detach().cpu().flatten().tolist()

    # Calculate sample statistics
    sample_mean = sum(data) / n
    sample_variance = sum([(x - sample_mean) ** 2 for x in data]) / n
    sample_std_dev = math.sqrt(sample_variance)

    # Calculate theoretical statistics from the requested half-open interval,
    # not from the observed extrema.
    theoretical_mean = (start_value + end_value) / 2
    theoretical_std_dev = (end_value - start_value) / math.sqrt(12)

    mean_scale = abs(theoretical_mean) if theoretical_mean != 0 else end_value - start_value
    mean_diff = abs(sample_mean - theoretical_mean) / mean_scale * 100
    std_dev_diff = (
        abs(sample_std_dev - theoretical_std_dev) / theoretical_std_dev * 100 if theoretical_std_dev != 0 else 0
    )

    treshold_percentage = 4
    if mean_diff < treshold_percentage and std_dev_diff < treshold_percentage:
        return True

    return False


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_tensor_dtype_and_value_range(device, dtype, layout):
    shape = (1024, 1024)
    low = 0
    high = 1
    tensor = ttnn.rand(shape, dtype=dtype, device=device, layout=layout)

    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(shape)

    torch_tensor = ttnn.to_torch(tensor)

    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
    assert check_uniform_distribution(
        torch_tensor, value_range=(low, high)
    ), "The distribution of random values is not uniform!"


@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_rand_rejects_unsupported_dtype(device, dtype, expect_error):
    with expect_error(RuntimeError, "Output dtype"):
        ttnn.rand(DEFAULT_SHAPE, dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", LEGACY_DTYPES)
def test_rand_preserves_legacy_low_precision_dtypes(device, dtype):
    tensor = ttnn.rand(DEFAULT_SHAPE, dtype=dtype, device=device, seed=TEST_SEED)

    assert tensor.dtype == dtype
    assert torch.isfinite(ttnn.to_torch(tensor)).all()


@pytest.mark.parametrize("dtype, torch_dtype, low, high", LEGACY_INTEGER_DTYPES)
def test_rand_preserves_legacy_integer_dtypes(device, dtype, torch_dtype, low, high):
    tensor = ttnn.rand(DEFAULT_SHAPE, dtype=dtype, device=device, low=low, high=high, seed=TEST_SEED)
    data = ttnn.to_torch(tensor)

    assert tensor.dtype == dtype
    assert data.dtype == torch_dtype
    assert tuple(data.shape) == DEFAULT_SHAPE
    assert torch.unique(data.to(torch.int64)).numel() > 1


def test_rand_defaults(device):
    tensor = ttnn.rand(DEFAULT_SHAPE, device=device)

    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert tensor.storage_type() == ttnn.StorageType.DEVICE
    assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


@pytest.mark.parametrize("shapes", SHAPES)
def test_rand_shapes(device, shapes):
    tensor = ttnn.rand(shapes, device=device)
    assert tuple(tensor.shape) == tuple(shapes)


@pytest.mark.parametrize("dim", [i for i in range(32)])
def test_rand_dims(dim, device):
    shape = (dim, dim)
    tensor = ttnn.rand(shape, device=device)
    assert tuple(tensor.shape) == tuple(shape)


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_rand_with_memory_config(device, mem_config):
    tensor = ttnn.rand(DEFAULT_SHAPE, device=device, memory_config=mem_config)
    assert tensor.memory_config() == mem_config
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


def test_rand_different_from_to_values(device):
    device.enable_program_cache()
    device.clear_program_cache()

    shape = (256, 256)
    dtype = ttnn.float32

    low_1, high_1 = 0.0, 1.0
    tensor_1 = ttnn.rand(shape, device=device, dtype=dtype, low=low_1, high=high_1)
    data_1 = ttnn.to_torch(tensor_1).float()
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after first rand, got {device.num_program_cache_entries()}"

    low_2, high_2 = 5.0, 10.0
    tensor_2 = ttnn.rand(shape, device=device, dtype=dtype, low=low_2, high=high_2)
    data_2 = ttnn.to_torch(tensor_2).float()
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after second rand (cache hit; from/to runtime-only), got {device.num_program_cache_entries()}"

    for torch_tensor, value_range in ((data_1, (low_1, high_1)), (data_2, (low_2, high_2))):
        assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
        assert check_uniform_distribution(
            torch_tensor, value_range=value_range
        ), "The distribution of random values is not uniform!"

    device.disable_and_clear_program_cache()


def test_rand_different_seed_values(device):
    """Regression guard for the seed static/dynamic contract (see PR #45350, which hacked this).

    `seed` is a DYNAMIC runtime value: it is deliberately excluded from compute_program_hash
    (so calls that differ only in seed still cache-hit) BUT must be re-applied to the cached
    program on every dispatch. This test pins BOTH halves of that contract so neither can
    regress:

      * `seed` must NOT grow the program cache  -> guards against re-adding seed to the hash
        (the recompile-per-seed hack). Asserting `== 1` fails the moment seed re-enters the key.
      * a different seed must change the output  -> guards against the freeze bug, where a
        cache hit reuses the first call's baked-in seed and silently returns identical data.
      * the same seed must reproduce the output  -> guards against the dynamic patch wrongly
        re-randomizing on a deterministic (seed != 0) call.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    shape = (256, 256)
    dtype = ttnn.float32

    data_seed_a = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=dtype, seed=1234)).float()
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after first rand, got {device.num_program_cache_entries()}"

    # Same seed, same shape: must reuse the cached program AND reproduce the exact values.
    data_seed_a_again = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=dtype, seed=1234)).float()
    assert device.num_program_cache_entries() == 1, "same seed must reuse the cached program (no new cache entry)"
    assert torch.equal(
        data_seed_a, data_seed_a_again
    ), "same seed must reproduce identical output (deterministic seed path)"

    # Different seed, same shape: seed is excluded from the hash, so it must STILL be a cache
    # hit (no new entry) and yet produce different values (seed re-patched on the cache hit).
    data_seed_b = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=dtype, seed=5678)).float()
    assert device.num_program_cache_entries() == 1, (
        "a different seed must NOT create a new cache entry -- seed is dynamic, not part of the "
        "program hash. A new entry here means seed was (re-)added to compute_program_hash (the "
        "recompile-per-seed hack)."
    )
    assert not torch.equal(data_seed_a, data_seed_b), (
        "a different seed must change the output -- otherwise the cache hit silently reused the "
        "first call's baked-in seed (the frozen-runtime-arg bug)."
    )

    device.disable_and_clear_program_cache()


def test_rand_tiles_have_iid_dispersion_and_low_lane_correlation(device):
    """Guard against correlated SFPU lanes distorting within-tile statistics."""
    shape = (1024, 1024)
    data = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=1)).float()
    assert torch.isfinite(data).all()
    assert abs(data.mean().item() - 0.5) < 0.01

    tiles = data.unfold(0, 32, 32).unfold(1, 32, 32).reshape(-1, 32, 32)
    tile_means = tiles.mean(dim=(1, 2))
    observed_std = tile_means.std(unbiased=False).item()
    iid_std = 1.0 / math.sqrt(12 * 32 * 32)

    assert 0.75 * iid_std < observed_std < 1.25 * iid_std, (
        f"tile-mean std {observed_std:.6f} is not close to the IID expectation {iid_std:.6f}; "
        "random elements may remain correlated within tiles"
    )

    adjacent_lane_correlations = torch.stack(
        [
            torch.corrcoef(torch.stack((tiles[:, :, lane].flatten(), tiles[:, :, lane + 1].flatten())))[0, 1]
            for lane in range(31)
        ]
    )
    worst_lane_correlation = adjacent_lane_correlations.abs().max().item()
    assert worst_lane_correlation < 0.06, f"worst adjacent-lane correlation {worst_lane_correlation:.6f} exceeds 0.06"

    assert (
        torch.unique(tiles.reshape(-1, 32 * 32), dim=0).shape[0] == tiles.shape[0]
    ), "generated RNG tiles were not unique"


@pytest.mark.parametrize("low, high", [(0.0, 5e-7), (2.0, 2.0000005), (-1.0, 0.0)])
def test_rand_respects_narrow_fp32_ranges(device, low, high):
    data = ttnn.to_torch(
        ttnn.rand((256, 256), device=device, dtype=ttnn.float32, low=low, high=high, seed=TEST_SEED)
    ).float()

    assert torch.isfinite(data).all()
    assert torch.all(data >= low)
    assert torch.all(data < high)
    assert torch.unique(data).numel() > 1


@pytest.mark.parametrize("low, high", [(-2.0, -1.0), (1.001, 2.0)])
def test_rand_respects_bfloat16_ranges(device, low, high):
    data = ttnn.to_torch(
        ttnn.rand((256, 256), device=device, dtype=ttnn.bfloat16, low=low, high=high, seed=TEST_SEED)
    ).float()

    assert torch.isfinite(data).all()
    assert torch.all(data >= low)
    assert torch.all(data < high)
    assert torch.unique(data).numel() > 1


def test_rand_rejects_range_without_bfloat16_value(device, expect_error):
    with expect_error(RuntimeError, "contains no value representable"):
        ttnn.rand((32, 32), device=device, dtype=ttnn.bfloat16, low=1.001, high=1.002, seed=TEST_SEED)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "low, high, expected",
    [
        (0.0, FLOAT32_MIN_NORMAL, 0.0),
        (-FLOAT32_MIN_NORMAL, 0.0, -FLOAT32_MIN_NORMAL),
    ],
)
def test_rand_respects_flush_to_zero_ranges(device, dtype, low, high, expected):
    data = ttnn.to_torch(ttnn.rand((32, 32), device=device, dtype=dtype, low=low, high=high, seed=TEST_SEED)).float()

    assert torch.all(data == expected)
    assert torch.all(data >= low)
    assert torch.all(data < high)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_rand_rejects_all_subnormal_range(device, dtype, expect_error):
    with expect_error(RuntimeError, "contains no value representable"):
        ttnn.rand(
            (32, 32),
            device=device,
            dtype=dtype,
            low=FLOAT32_MIN_SUBNORMAL,
            high=FLOAT32_MIN_NORMAL,
            seed=TEST_SEED,
        )


@pytest.mark.parametrize(
    "low, high, error",
    [
        (math.nan, 1.0, "endpoints must be finite"),
        (0.0, math.inf, "endpoints must be finite"),
        (1.0, 1.0, "lower bound must be less than upper bound"),
        (1.0, 0.0, "lower bound must be less than upper bound"),
        (-torch.finfo(torch.float32).max, torch.finfo(torch.float32).max, "too wide"),
        (FLOAT32_MIN_NORMAL, FLOAT32_MIN_NORMAL_PLUS_TWO_ULPS, "subnormal scale"),
    ],
)
def test_rand_rejects_invalid_range(device, expect_error, low, high, error):
    with expect_error(RuntimeError, error):
        ttnn.rand((32, 32), device=device, dtype=ttnn.float32, low=low, high=high, seed=TEST_SEED)


def test_rand_all_ones_seed_does_not_lock_prng(device):
    data = ttnn.to_torch(ttnn.rand((32, 32), device=device, dtype=ttnn.float32, seed=0xFFFFFFFF)).float()

    # A single tile runs on one core, so other cores cannot mask an XNOR LFSR
    # left in the all-ones lock state. A locked state can produce at most one
    # constant value per lane, independent of how lanes map into tile rows.
    assert torch.unique(data).numel() > 32


def test_rand_decorrelates_neighboring_core_seeds(device):
    """Neighboring host seeds must not collapse to the same per-core RNG stream."""
    output = ttnn.to_torch(ttnn.rand((64, 32), device=device, dtype=ttnn.float32, seed=0xFFFFFFFE)).reshape(2, 32, 32)

    # Two tiles run on two cores with host seeds 0xFFFFFFFE and 0xFFFFFFFF.
    # rand_init remaps the all-ones lock state, so omitting the work-range
    # stream ID would make both cores start from the same effective seed.
    assert not torch.equal(output[0], output[1]), "neighboring cores emitted duplicate random tiles"


def test_rand_fp32_uses_both_mantissa_parities(device):
    data = ttnn.to_torch(ttnn.rand((256, 256), device=device, dtype=ttnn.float32, seed=1)).float()
    top_binade = data[(data >= 0.5) & (data < 1.0)]
    odd_mantissa_fraction = (top_binade.contiguous().view(torch.int32) & 1).float().mean().item()

    # SFPCAST's round-to-nearest-even conversion uses the discarded source
    # bits, so both mantissa parities should occur evenly in [0.5, 1).
    assert 0.45 < odd_mantissa_fraction < 0.55


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(2, id="1x2_grid"), pytest.param((2, 1), id="2x1_grid")],
    indirect=True,
)
def test_rand_program_cache_with_mesh_mapper(mesh_device):
    """
    Exercise the program cache across replicated and sharded ttnn.rand calls
    that share the same per-device shape.  Because mesh_dim_is_sharded only
    affects runtime args (seed offset) and override_runtime_arguments handles
    it correctly, the two configurations should share a single cache entry
    while still producing correct output:

      - replicated (no mapper) → identical data on every device
      - sharded (with mapper)  → distinct data per shard device
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices")

    mesh_device.enable_program_cache()

    seed = 42
    shard_dim = 0
    per_device_rows = 256
    cols = 256
    shard_shape = (per_device_rows, cols)
    full_shape = (per_device_rows * num_devices, cols)
    dtype = ttnn.float32
    mesh_shape = tuple(mesh_device.shape)
    placements = _shard_placements(mesh_shape, shard_dim)

    # --- Call 1: no mesh_mapper (replicated) ---
    t_rep = ttnn.rand(shard_shape, mesh_device, dtype=dtype, seed=seed)
    entries_after_rep = mesh_device.num_program_cache_entries()

    # --- Call 2: with mesh_mapper, shard shape == shard_shape (cache hit) ---
    t_shard = ttnn.rand(
        full_shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(placements),
    )
    assert mesh_device.num_program_cache_entries() == entries_after_rep, (
        f"Expected cache entries to stay at {entries_after_rep} (same per-device shape), "
        f"got {mesh_device.num_program_cache_entries()}"
    )

    # --- Call 3: repeat sharded call — still a cache hit ---
    t_shard2 = ttnn.rand(
        full_shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(placements),
    )
    assert mesh_device.num_program_cache_entries() == entries_after_rep, (
        f"Expected cache entries to stay at {entries_after_rep} after repeated sharded call, "
        f"got {mesh_device.num_program_cache_entries()}"
    )

    # --- Call 4: back to replicated — still a cache hit ---
    t_rep2 = ttnn.rand(shard_shape, mesh_device, dtype=dtype, seed=seed)
    assert mesh_device.num_program_cache_entries() == entries_after_rep, (
        f"Expected cache entries to stay at {entries_after_rep} after switching back to replicated, "
        f"got {mesh_device.num_program_cache_entries()}"
    )

    # --- Correctness: replicated calls produce identical data on all devices ---
    for label, tensor in [("rep1", t_rep), ("rep2", t_rep2)]:
        shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]
        for i in range(1, len(shards)):
            assert torch.equal(
                shards[0], shards[i]
            ), f"{label}: device 0 and device {i} should be identical (replicated)"

    # --- Correctness: sharded calls produce distinct data per device ---
    for label, tensor in [("shard1", t_shard), ("shard2", t_shard2)]:
        shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]
        for i in range(1, len(shards)):
            assert not torch.equal(shards[0], shards[i]), f"{label}: device 0 and device {i} should differ (sharded)"

    # --- Correctness: repeated calls with the same seed are deterministic ---
    shard1_data = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(t_shard)]
    shard2_data = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(t_shard2)]
    for i in range(len(shard1_data)):
        assert torch.equal(
            shard1_data[i], shard2_data[i]
        ), f"Device {i}: two sharded calls with the same seed should be deterministic"

    mesh_device.disable_and_clear_program_cache()


def test_rand_invalid_args(device):
    """
    Passing invalid args should raise TypeError.
    """

    with pytest.raises(TypeError):  # allow-pytest.raises: pre-existing binding validation test
        # expected list or tuple
        ttnn.rand(5, device=device)

    with pytest.raises(TypeError):  # allow-pytest.raises: pre-existing binding validation test
        # expected positive dim values
        ttnn.rand([2, -1], device=device)

    with pytest.raises(TypeError):  # allow-pytest.raises: pre-existing binding validation test
        # expected ttnn.LAYOUT type
        ttnn.rand([2, 2], device=device, layout="ROW_MAJOR")

    with pytest.raises(TypeError):  # allow-pytest.raises: pre-existing binding validation test
        # expected  ttnn.MemoryConfig type
        ttnn.rand([2, 2], device=device, memory_config="DRAM")

    with pytest.raises(TypeError):  # allow-pytest.raises: pre-existing binding validation test
        # expected  ttnn.Device type
        ttnn.rand([2, 2], device="WORMHOLE")

    with pytest.raises(TypeError):  # allow-pytest.raises: pre-existing binding validation test
        # expected  ttnn.DataType type
        ttnn.rand([2, 2], device=device, dtype="ttnn.bfloat16")


# ---------------------------------------------------------------------------
# Multi-device tests (mesh_mapper)
# ---------------------------------------------------------------------------


def _shard_placements(mesh_shape, shard_dim):
    """Build a placements list that shards `shard_dim` on the non-trivial mesh axis."""
    return [
        ttnn.PlacementShard(shard_dim) if mesh_shape[i] > 1 else ttnn.PlacementReplicate()
        for i in range(len(mesh_shape))
    ]


def _replicate_placements(mesh_shape):
    return [ttnn.PlacementReplicate() for _ in range(len(mesh_shape))]


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
def test_rand_mesh_shape_override_reshape(mesh_device):
    """A logical 1x4 distribution must map to the four physical devices in row-major order."""
    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need at least 4 devices")

    seed = 77
    shard_shape = (256, 256)
    distribution_shape = ttnn.MeshShape(1, 4)
    expected_mesh_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    sharded_mapper = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)],
        mesh_shape_override=distribution_shape,
    )
    replicated_mapper = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
        mesh_shape_override=distribution_shape,
    )

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()

    sharded = ttnn.rand(
        (shard_shape[0] * 4, shard_shape[1]),
        mesh_device,
        dtype=ttnn.float32,
        seed=seed,
        mesh_mapper=sharded_mapper,
    )
    cache_entries = mesh_device.num_program_cache_entries()
    replicated = ttnn.rand(
        shard_shape,
        mesh_device,
        dtype=ttnn.float32,
        seed=seed,
        mesh_mapper=replicated_mapper,
    )
    sharded_again = ttnn.rand(
        (shard_shape[0] * 4, shard_shape[1]),
        mesh_device,
        dtype=ttnn.float32,
        seed=seed,
        mesh_mapper=sharded_mapper,
    )

    assert mesh_device.num_program_cache_entries() == cache_entries
    for tensor in (sharded, replicated, sharded_again):
        topology = tensor.tensor_topology()
        assert tuple(topology.distribution_shape()) == (1, 4)
        assert [tuple(coord) for coord in topology.mesh_coords()] == expected_mesh_coords

    sharded_data = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sharded)]
    replicated_data = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(replicated)]
    sharded_again_data = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sharded_again)]

    assert len(sharded_data) == 4
    assert len(replicated_data) == len(sharded_again_data) == 4
    composed = ttnn.to_torch(sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    assert torch.equal(composed, torch.cat(sharded_data, dim=0))
    for i in range(4):
        assert torch.equal(sharded_data[i], sharded_again_data[i])
        assert torch.equal(replicated_data[0], replicated_data[i])
        for j in range(i + 1, 4):
            assert not torch.equal(
                sharded_data[i], sharded_data[j]
            ), f"Logical shards {i} and {j} received the same random stream"

    mesh_device.disable_and_clear_program_cache()


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
def test_rand_mesh_shape_override_submesh(mesh_device):
    """A smaller logical distribution must expose and dispatch only on its mapped devices."""
    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need at least 4 devices")

    mapper = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)],
        mesh_shape_override=ttnn.MeshShape(1, 2),
    )
    kwargs = {
        "shape": (512, 256),
        "device": mesh_device,
        "dtype": ttnn.float32,
        "seed": 91,
        "mesh_mapper": mapper,
    }

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    tensor = ttnn.rand(**kwargs)
    cache_entries = mesh_device.num_program_cache_entries()
    tensor_again = ttnn.rand(**kwargs)

    assert mesh_device.num_program_cache_entries() == cache_entries
    assert tuple(tensor.tensor_topology().distribution_shape()) == (1, 2)
    assert [tuple(coord) for coord in tensor.tensor_topology().mesh_coords()] == [(0, 0), (0, 1)]

    shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]
    shards_again = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor_again)]
    assert len(shards) == len(shards_again) == 2
    assert not torch.equal(shards[0], shards[1])
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(shards, shards_again))

    mesh_device.disable_and_clear_program_cache()


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(2, id="1x2_grid"), pytest.param((2, 1), id="2x1_grid")],
    indirect=True,
)
def test_rand_mesh_shard(mesh_device):
    """
    Shard a random tensor across devices along dim 0, then verify:
      - mesh_mapper produces the right per-device shard shapes
      - unique_per_device seeding gives each device a distinct sequence
      - composed result is uniformly distributed
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices")

    seed = 42
    shard_dim = 0
    per_device_rows = 256
    cols = 256
    full_shape = (per_device_rows * num_devices, cols)
    dtype = ttnn.float32
    mesh_shape = tuple(mesh_device.shape)

    sharded_tensor = ttnn.rand(
        full_shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(_shard_placements(mesh_shape, shard_dim)),
    )

    composed = ttnn.to_torch(
        sharded_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim),
    ).float()

    assert tuple(composed.shape) == full_shape, f"Expected {full_shape}, got {tuple(composed.shape)}"
    assert not torch.isnan(composed).any(), "Composed tensor contains NaN values"
    assert check_uniform_distribution(composed), "Composed tensor is not uniformly distributed"

    shards = torch.chunk(composed, num_devices, dim=shard_dim)
    for i in range(1, len(shards)):
        assert not torch.equal(
            shards[0], shards[i]
        ), f"Shard 0 and shard {i} are identical — unique_per_device seeding did not work"


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(2, id="1x2_grid"), pytest.param((2, 1), id="2x1_grid")],
    indirect=True,
)
def test_rand_mesh_replicate(mesh_device):
    """
    Replicate a random tensor across devices with a fixed seed, then verify
    that every device holds the same data.
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices")

    seed = 42
    shape = (256, 256)
    dtype = ttnn.float32
    mesh_shape = tuple(mesh_device.shape)

    replicated_tensor = ttnn.rand(
        shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(_replicate_placements(mesh_shape)),
    )

    device_tensors = ttnn.get_device_tensors(replicated_tensor)
    shards = [ttnn.to_torch(t).float() for t in device_tensors]

    for i in range(1, len(shards)):
        assert torch.equal(
            shards[0], shards[i]
        ), f"Replicated shard 0 and shard {i} differ — replicate seeding is broken"

    assert not torch.isnan(shards[0]).any(), "Replicated tensor contains NaN values"
    assert check_uniform_distribution(shards[0]), "Replicated tensor is not uniformly distributed"


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(2, id="1x2_grid"), pytest.param((2, 1), id="2x1_grid")],
    indirect=True,
)
def test_rand_mesh_shard_matches_single_device(mesh_device):
    """
    Shard 0 of a sharded ttnn.rand equals a replicated ttnn.rand with the same seed and per-device
    shape: both use distribution index 0, so their per-core keys are identical. Every other shard
    uses a different distribution index and must differ.
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices")

    seed = 100
    shard_dim = 0
    per_device_rows = 256
    cols = 256
    full_shape = (per_device_rows * num_devices, cols)
    shard_shape = (per_device_rows, cols)
    dtype = ttnn.float32
    mesh_shape = tuple(mesh_device.shape)

    sharded_tensor = ttnn.rand(
        full_shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(_shard_placements(mesh_shape, shard_dim)),
    )
    shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sharded_tensor)]
    replicated = ttnn.to_torch(
        ttnn.get_device_tensors(ttnn.rand(shard_shape, mesh_device, dtype=dtype, seed=seed))[0]
    ).float()

    assert torch.equal(shards[0], replicated), "shard 0 must equal the replicated call (distribution index 0)"
    for d in range(1, num_devices):
        assert tuple(shards[d].shape) == shard_shape
        assert not torch.equal(shards[d], replicated), f"shard {d} must differ from distribution index 0"


@pytest.mark.parametrize(
    "mesh_device, shard_mesh_dim",
    [
        pytest.param((2, 2), 0, id="2x2_shard_dim0"),
        pytest.param((2, 2), 1, id="2x2_shard_dim1"),
    ],
    indirect=["mesh_device"],
)
def test_rand_mesh_2d_shard_and_replicate(mesh_device, shard_mesh_dim):
    """
    On a 2D mesh, shard along one mesh dimension and replicate along the other.
    Verify:
      - Devices along the replicate axis hold identical data.
      - Devices along the shard axis hold distinct data.
    """
    mesh_shape = tuple(mesh_device.shape)
    rows, cols = mesh_shape
    if rows * cols < 4:
        pytest.skip("Need at least 4 devices for a 2x2 mesh")

    seed = 77
    shard_dim = 0
    per_shard_rows = 256
    num_shards = mesh_shape[shard_mesh_dim]
    full_shape = (per_shard_rows * num_shards, 256)
    dtype = ttnn.float32

    placements = [
        ttnn.PlacementShard(shard_dim) if i == shard_mesh_dim else ttnn.PlacementReplicate()
        for i in range(len(mesh_shape))
    ]

    sharded_tensor = ttnn.rand(
        full_shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(placements),
    )

    device_tensors = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sharded_tensor)]

    replicate_mesh_dim = 1 - shard_mesh_dim

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            coord = (r, c)

            # Check replicas: devices differing only on the replicate axis must match.
            if coord[replicate_mesh_dim] > 0:
                replica_coord = list(coord)
                replica_coord[replicate_mesh_dim] = 0
                replica_idx = replica_coord[0] * cols + replica_coord[1]
                assert torch.equal(device_tensors[idx], device_tensors[replica_idx]), (
                    f"Device {coord} should be a replica of device {tuple(replica_coord)} " f"but data differs"
                )

            # Check shards: devices differing on the shard axis must differ.
            if coord[shard_mesh_dim] > 0:
                shard_neighbor = list(coord)
                shard_neighbor[shard_mesh_dim] = 0
                neighbor_idx = shard_neighbor[0] * cols + shard_neighbor[1]
                assert not torch.equal(device_tensors[idx], device_tensors[neighbor_idx]), (
                    f"Device {coord} and device {tuple(shard_neighbor)} are on different "
                    f"shards but hold identical data"
                )


# ---------------------------------------------------------------------------
# Device-side state (trace replay) and generator selection
# ---------------------------------------------------------------------------

M32 = 0xFFFFFFFF


def _mix32(h):
    h &= M32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M32
    h ^= h >> 16
    return h


def _stream_key(seed, device_index):
    return _mix32(_mix32(seed) ^ ((device_index * 0x9E3779B9 + 0x7F4A7C15) & M32))


def _threefry2x32(x0, x1, k0, k1, rounds=20):
    rot = [13, 15, 26, 6, 17, 29, 16, 24]
    ks = [k0, k1, k0 ^ k1 ^ 0x1BD11BDA]
    x0 = (x0 + ks[0]) & M32
    x1 = (x1 + ks[1]) & M32
    for r in range(rounds):
        x0 = (x0 + x1) & M32
        x1 = ((x1 << rot[r % 8]) | (x1 >> (32 - rot[r % 8]))) & M32
        x1 ^= x0
        if r % 4 == 3:
            j = r // 4 + 1
            x0 = (x0 + ks[j % 3]) & M32
            x1 = (x1 + ks[(j + 1) % 3] + j) & M32
    return x0, x1


def _threefry_reference_tile(tile_index, key0, key1):
    """One 32x32 fp32 tile of ttnn.rand(low=0, high=1, generator=THREEFRY). Counter (16*tile + 4*face + pair, lane_id)
    with lane_id = 2*k for lane index k; the SFPU stores word w of lane k at face row 4*pair + k//8, column 2*(k%8) + w.
    The [0, 1) scale 1 - 2^-24 is folded with 2^-31 and applied by one FMA, so the float64 product rounded once to
    float32 reproduces the device value."""
    scale = float(torch.tensor([0x3F7FFFFF - (31 << 23)], dtype=torch.int32).view(torch.float32)[0])
    out = torch.zeros((32, 32), dtype=torch.float32)
    for f in range(4):
        r0, c0 = 16 * (f // 2), 16 * (f % 2)
        for p in range(4):
            ctr = (tile_index * 16 + f * 4 + p) & M32
            words = [_threefry2x32(ctr, 2 * k, key0, key1) for k in range(32)]
            for w in (0, 1):
                vals = (torch.tensor([float(x[w] >> 1) for x in words], dtype=torch.float32).double() * scale).float()
                for k in range(32):
                    out[r0 + 4 * p + k // 8, c0 + 2 * (k % 8) + w] = vals[k]
    return out


def test_rand_generator_threefry_matches_reference(device):
    """Device Threefry-2x32-20 output equals the host reference bit for bit, on tiles spread across cores."""
    seed = 1
    grid = device.compute_with_storage_grid_size()
    data = ttnn.to_torch(
        ttnn.rand(
            (64, 32 * grid.x * grid.y),
            device=device,
            dtype=ttnn.float32,
            seed=seed,
            generator=ttnn.RandGenerator.THREEFRY,
        )
    ).float()
    tiles = data.reshape(2, 32, -1, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)
    key0 = _stream_key(seed, 0)
    for i in (0, 1, tiles.shape[0] // 2, tiles.shape[0] - 1):
        assert torch.equal(
            _threefry_reference_tile(i, key0, 0), tiles[i]
        ), f"tile {i} differs from the Threefry reference"


@pytest.mark.parametrize("generator", [ttnn.RandGenerator.LFSR, ttnn.RandGenerator.THREEFRY])
def test_rand_generator_uniform_and_deterministic(device, generator):
    shape = (1024, 1024)
    a = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=9, generator=generator)).float()
    b = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=9, generator=generator)).float()
    c = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=10, generator=generator)).float()
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert check_uniform_distribution(a)
    assert a.min() >= 0.0 and a.max() < 1.0


def test_rand_neighbouring_seeds_share_no_tiles(device):
    """Hashed per-core keys: seed s+1 must not be seed s shifted by one core (the old additive contract)."""
    shape = (256, 32 * 130)
    a = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=5)).float()
    b = ttnn.to_torch(ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=6)).float()
    tiles_a = {t.numpy().tobytes() for t in a.reshape(8, 32, -1, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)}
    shared = sum(
        t.numpy().tobytes() in tiles_a for t in b.reshape(8, 32, -1, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)
    )
    assert shared == 0


def test_rand_state_rejects_wrong_tensor(device, expect_error):
    bad = ttnn.zeros((4, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y <= 4:
        pytest.skip("grid too small to make the state tensor short")
    with expect_error(RuntimeError, "state must come from ttnn::rand_state"):
        ttnn.rand((32, 32), device=device, dtype=ttnn.float32, seed=1, state=bad)


@pytest.mark.parametrize("generator", [ttnn.RandGenerator.LFSR, ttnn.RandGenerator.THREEFRY])
def test_rand_state_advances_eagerly_and_resets(device, generator):
    shape = (512, 512)
    state = ttnn.rand_state(device)
    grid = device.compute_with_storage_grid_size()
    assert tuple(state.shape) == (grid.x * grid.y, 32)
    a = ttnn.to_torch(
        ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=4, state=state, generator=generator)
    ).float()
    b = ttnn.to_torch(
        ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=4, state=state, generator=generator)
    ).float()
    assert not torch.equal(a, b), "consecutive calls with state must differ"
    epochs = ttnn.to_torch(state)[:, 0].to(torch.int64)
    assert int(epochs.min()) == 2 and int(epochs.max()) == 2, "every active core advanced twice"
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.zeros((grid.x * grid.y, 32), dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        state,
    )
    again = ttnn.to_torch(
        ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=4, state=state, generator=generator)
    ).float()
    assert torch.equal(again, a), "resetting the state must reproduce the first call"
    assert check_uniform_distribution(b)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
@pytest.mark.parametrize("generator", [ttnn.RandGenerator.LFSR, ttnn.RandGenerator.THREEFRY])
def test_rand_state_trace_replays_fresh_values(device, generator):
    shape = (512, 512)
    state = ttnn.rand_state(device)
    ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=4, state=state, generator=generator)  # compile
    frozen = ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=4, generator=generator)  # no state: compile
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    fresh = ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=4, state=state, generator=generator)
    frozen = ttnn.rand(shape, device=device, dtype=ttnn.float32, seed=4, generator=generator)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    outs, frozens = [], []
    for _ in range(3):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        outs.append(ttnn.to_torch(fresh).float())
        frozens.append(ttnn.to_torch(frozen).float())
    ttnn.release_trace(device, tid)
    assert all(torch.equal(frozens[0], f) for f in frozens[1:]), "without state a replay repeats its values"
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(outs[i], outs[j]), f"replays {i} and {j} with state must differ"
    assert all(check_uniform_distribution(o) for o in outs)


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(2, id="1x2_grid"), pytest.param((2, 1), id="2x1_grid")],
    indirect=True,
)
@pytest.mark.parametrize("generator", [ttnn.RandGenerator.LFSR, ttnn.RandGenerator.THREEFRY])
def test_rand_mesh_state_per_device(mesh_device, generator):
    """Each device advances its own state page; replicated calls stay identical across devices per call."""
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices")
    state = ttnn.rand_state(mesh_device)
    shape = (1024, 1024)  # 1024 tiles: every core runs, so every epoch page advances
    a = [
        ttnn.to_torch(t).float()
        for t in ttnn.get_device_tensors(
            ttnn.rand(shape, mesh_device, dtype=ttnn.float32, seed=3, state=state, generator=generator)
        )
    ]
    b = [
        ttnn.to_torch(t).float()
        for t in ttnn.get_device_tensors(
            ttnn.rand(shape, mesh_device, dtype=ttnn.float32, seed=3, state=state, generator=generator)
        )
    ]
    for d in range(num_devices):
        assert torch.equal(a[0], a[d]), "replicated call: devices agree"
        assert not torch.equal(a[d], b[d]), "second call: state advanced"
    epochs = [ttnn.to_torch(t)[:, 0].to(torch.int64) for t in ttnn.get_device_tensors(state)]
    for e in epochs:
        assert int(e.min()) == 2 and int(e.max()) == 2
