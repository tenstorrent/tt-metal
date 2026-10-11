# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


def _make_operands(device, m, k, n, num_cores, b_dtype):
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < num_cores:
        pytest.skip(f"device has fewer than {num_cores} cores")
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    kc = k // num_cores

    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    a_memory_config = ttnn.create_sharded_memory_config(
        (m, kc),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    b_memory_config = ttnn.create_sharded_memory_config(
        (kc, n),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_b = ttnn.from_torch(
        torch_b, dtype=b_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=b_memory_config
    )
    return core_grid, a_memory_config, tt_b


@pytest.mark.parametrize(
    "m, k, n, num_cores, b_dtype",
    [
        # The hyper-connection fn projection: K = hc * D, N = (2 + hc) * hc padded to a tile.
        (1, 16384, 32, 32, ttnn.bfloat16),
        (1, 16384, 32, 32, ttnn.bfloat8_b),
        # Wider N on the custom_mm path.
        (1, 4096, 256, 8, ttnn.bfloat16),
        # A tree that is not a full power of the fan-in.
        (1, 24 * 512, 32, 24, ttnn.bfloat16),
        # Odd tile count per core: the general matmul path.
        (1, 13 * 96, 64, 13, ttnn.bfloat16),
        # M > 1, with a [M, N] block larger than DST.
        (4, 8192, 64, 16, ttnn.bfloat16),
        (8, 4096, 64, 8, ttnn.bfloat16),
        (32, 4096, 32, 8, ttnn.bfloat16),
        # A single core: no reduction at all.
        (1, 512, 32, 1, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("reduce_fan_in", [2, 4])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
def test_matmul_decode_large_k(device, m, k, n, num_cores, b_dtype, reduce_fan_in, out_dtype):
    torch.manual_seed(0)
    core_grid, a_memory_config, tt_b = _make_operands(device, m, k, n, num_cores, b_dtype)
    # Compare against the weight as stored, so block-float rounding is not counted as error.
    torch_b = ttnn.to_torch(tt_b).float()
    root = ttnn.corerange_to_cores(core_grid, row_wise=True)[0]

    # Several invocations of the same cached program, each with fresh activations, so a
    # semaphore or CB left dirty by one run shows up as a wrong result in the next.
    for _ in range(3):
        torch_a = torch.randn((m, k), dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=a_memory_config
        )
        tt_out = ttnn.experimental.matmul_decode_large_k(tt_a, tt_b, dtype=out_dtype, reduce_fan_in=reduce_fan_in)

        assert tt_out.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tt_out.dtype == out_dtype
        assert list(tt_out.shape) == [m, n]
        shard_spec = tt_out.memory_config().shard_spec
        assert list(shard_spec.shape) == [m, n]
        out_cores = ttnn.corerange_to_cores(shard_spec.grid)
        assert len(out_cores) == 1 and (out_cores[0].x, out_cores[0].y) == (root.x, root.y)

        assert_with_pcc(torch_a.float() @ torch_b, ttnn.to_torch(tt_out).float(), 0.99)


@pytest.mark.parametrize(
    "m, k, n, num_cores, b_dtype",
    [
        # The hyper-connection fn projection: V4 (D = 4096) and V4.1 (D = 5120), hc = 4.
        (1, 16384, 32, 32, ttnn.bfloat16),
        (1, 20480, 32, 32, ttnn.bfloat8_b),
        # Odd tile count per core: the general matmul path.
        (1, 13 * 96, 64, 13, ttnn.bfloat16),
        # M > 1: one sum(A^2) tile per row rides the tree.
        (4, 8192, 64, 16, ttnn.bfloat16),
        # A single core: the root normalizes its own partial.
        (1, 512, 32, 1, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("reduce_fan_in", [2, 4])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.skipif(not is_blackhole(), reason="the fused rms_norm is Blackhole-only")
def test_matmul_decode_large_k_rms_norm(device, m, k, n, num_cores, b_dtype, reduce_fan_in, out_dtype):
    """``rms_norm=True`` computes ``rms_norm(A) @ B`` with the unweighted norm over all of K."""
    torch.manual_seed(0)
    eps = 1e-6
    core_grid, a_memory_config, tt_b = _make_operands(device, m, k, n, num_cores, b_dtype)
    torch_b = ttnn.to_torch(tt_b).float()

    for _ in range(3):
        # Rows of very different magnitude, so a scale applied to the wrong row (or not at all) is
        # caught -- PCC alone is blind to a uniform scale, hence the norm-ratio check as well.
        row_scale = torch.logspace(-1, 2, m).reshape(m, 1)
        torch_a = (torch.randn((m, k)) * row_scale).to(torch.bfloat16)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=a_memory_config
        )
        tt_out = ttnn.experimental.matmul_decode_large_k(
            tt_a, tt_b, dtype=out_dtype, reduce_fan_in=reduce_fan_in, rms_norm=True, rms_norm_epsilon=eps
        )
        assert list(tt_out.shape) == [m, n]
        # The same matmul without the norm, so the scale check below measures the fused norm alone
        # and not the matmul's own precision (custom_mm runs at LoFi).
        tt_plain = ttnn.experimental.matmul_decode_large_k(tt_a, tt_b, dtype=out_dtype, reduce_fan_in=reduce_fan_in)

        a = torch_a.float()
        scale = torch.rsqrt(a.pow(2).mean(dim=-1, keepdim=True) + eps)
        actual = ttnn.to_torch(tt_out).float()
        assert_with_pcc((a * scale) @ torch_b, actual, 0.99)
        expected = ttnn.to_torch(tt_plain).float() * scale
        ratio = actual.norm(dim=-1) / expected.norm(dim=-1)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=0.01), ratio


@pytest.mark.parametrize(
    "hc, d, num_cores, rms_norm",
    [
        # The V4.1 hyper-connection's ROW_MAJOR streams [hc, D] on the 8 collapse cores.
        (4, 5120, 8, True),
        (4, 5120, 8, False),
        (4, 4096, 8, True),
        # Fewer streams and cores.
        (2, 1024, 2, False),
    ],
)
@pytest.mark.skipif(not is_blackhole(), reason="the fused rms_norm is Blackhole-only")
def test_matmul_decode_large_k_fold_rows(device, hc, d, num_cores, rms_norm):
    """``fold_rows=hc`` reads ``[1, 1, hc, D]`` width-sharded streams as the ``[1, hc*D]`` row
    each core's shard lays out core-major, against a weight whose K is in that order."""
    torch.manual_seed(0)
    eps = 1e-6
    n = 32
    k = hc * d
    w = d // num_cores
    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    streams_memory_config = ttnn.create_sharded_memory_config(
        (hc, w),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    b_memory_config = ttnn.create_sharded_memory_config(
        (k // num_cores, n),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_streams = torch.randn((1, 1, hc, d), dtype=torch.bfloat16)
    # Stream-major weight [hc*D, N] (the checkpoint order), reordered to (core, stream, column).
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_b_core_major = torch_b.reshape(hc, num_cores, w, n).permute(1, 0, 2, 3).reshape(k, n)
    tt_streams = ttnn.from_torch(
        torch_streams,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=streams_memory_config,
    )
    tt_b = ttnn.from_torch(
        torch_b_core_major, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=b_memory_config
    )
    rms_kwargs = {"rms_norm": True, "rms_norm_epsilon": eps} if rms_norm else {}
    tt_out = ttnn.experimental.matmul_decode_large_k(tt_streams, tt_b, fold_rows=hc, **rms_kwargs)

    assert list(tt_out.shape) == [1, 1, 1, n]
    a = torch_streams.float().reshape(1, k)
    if rms_norm:
        a = a * torch.rsqrt(a.pow(2).mean(dim=-1, keepdim=True) + eps)
    assert_with_pcc(a @ torch_b.float(), ttnn.to_torch(tt_out).float().reshape(1, n), 0.99)


@pytest.mark.parametrize("m", [1, 8])
def test_matmul_decode_large_k_untilized_activation(device, m):
    """The hyper-connection's A: a tile-padded width-sharded activation, unpadded to a
    ``[M, Kc]`` ROW_MAJOR shard on the same cores by ``untilize_with_unpadding``."""
    k, n, num_cores = 16384, 32, 32
    torch.manual_seed(0)
    _, a_memory_config, tt_b = _make_operands(device, ttnn.TILE_SIZE, k, n, num_cores, ttnn.bfloat16)
    torch_b = ttnn.to_torch(tt_b).float()

    torch_a = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_memory_config
    )
    tt_a = ttnn.untilize_with_unpadding(tt_a, [0, 0, m - 1, k - 1], memory_config=tt_a.memory_config())
    assert list(tt_a.memory_config().shard_spec.shape) == [m, k // num_cores]
    tt_out = ttnn.experimental.matmul_decode_large_k(tt_a, tt_b)

    assert list(tt_out.shape) == [1, 1, m, n]
    assert list(tt_out.memory_config().shard_spec.shape) == [m, n]
    assert_with_pcc(torch_a.float() @ torch_b, ttnn.to_torch(tt_out).float(), 0.99)
