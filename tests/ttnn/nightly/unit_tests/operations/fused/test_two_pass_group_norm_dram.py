# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Full two-pass regression matrices; sanity retains a small representative sample."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from loguru import logger
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.fused.test_group_norm_DRAM import DEVICE_PARAMS_L1_SMALL_SIZE


@pytest.fixture
def enabled_program_cache(device):
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("has_affine", [False, True], ids=["plain", "affine"])
@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize(
    "constant",
    [None, 1e38, -1e38, 1e-37, -1e-37],
    ids=["offset", "large_positive", "large_negative", "small_positive", "small_negative"],
)
def test_group_norm_fp32_large_offset_DRAM(device, has_affine, num_groups, constant):
    """The FP32 finalizer must not truncate x and mean to TF32 before subtraction."""
    torch.manual_seed(7)
    N, C, HW = 1, 64, 32
    x = 1_000_000.0 + 128.0 * (torch.rand((N, 1, HW, C), dtype=torch.float32) - 0.5)
    if constant is not None:
        x.fill_(constant)
    weight = torch.linspace(0.75, 1.25, C, dtype=torch.float32) if has_affine else None
    bias = torch.linspace(-0.25, 0.25, C, dtype=torch.float32) if has_affine else None
    if constant is not None and bias is not None:
        # Keep beta exact through the TF32 affine stage so equality tests the
        # statistics result, not parameter truncation in that later stage.
        bias = bias.to(torch.bfloat16).to(torch.float32)
    if constant is None:
        reference = torch.nn.functional.group_norm(
            x.view(N, HW, C).permute(0, 2, 1).reshape(N, C, 1, HW), num_groups, weight=weight, bias=bias
        ).permute(0, 2, 3, 1)
    else:
        # Constant populations normalise to zero. Avoid an overflowing CPU
        # statistics reference; affine output must equal beta exactly.
        reference = torch.zeros_like(x) if bias is None else bias.view(1, 1, 1, C).expand_as(x)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    input_tensor = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if has_affine:
        [gamma, beta] = ttnn.dram_group_norm_params_from_torch(
            [weight, bias],
            C,
            num_groups,
            device,
            core_grid=ttnn.CoreGrid(y=1, x=1),
            return_mask=False,
            dtype=ttnn.float32,
        )
    else:
        gamma = beta = None
    output = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        weight=gamma,
        bias=beta,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=1, x=1),
        dtype=ttnn.float32,
        compute_kernel_config=compute_kernel_config,
        use_welford=True,
        inplace=False,
    )
    actual = ttnn.to_torch(ttnn.from_device(output)).float()

    error = actual - reference
    assert torch.isfinite(actual).all()
    if constant is not None:
        # Exact equality also detects small means flushed by premature scaling.
        assert torch.equal(actual, reference)
    else:
        assert_numeric_metrics(reference, actual, rtol=0, atol=0.015, frobenius_threshold=0.02)
        assert error.abs().mean() < 0.004


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@run_for_blackhole("The near-capacity allocation is calibrated for Blackhole L1")
def test_group_norm_interleaved_l1_replay_respects_occupied_l1(device, enabled_program_cache):
    torch.manual_seed(20260904)
    N, C, H, W, num_groups = 1, 256, 256, 256, 32
    grid = ttnn.CoreGrid(y=8, x=8)
    torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    reference = torch.nn.functional.group_norm(torch_input, num_groups).permute(0, 2, 3, 1).view(N, 1, H * W, C)
    input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).view(N, 1, H * W, C),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run_group_norm():
        return ttnn.group_norm(
            input_tensor,
            num_groups=num_groups,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=grid,
            num_out_blocks=8,
            use_welford=True,
            inplace=False,
        )

    warm_output = run_group_norm()
    ttnn.synchronize_device(device)
    entries_with_replay = device.num_program_cache_entries()
    warm_output.deallocate(force=True)

    # Occupying 850 KiB/core leaves room for the streaming program, but not for
    # replaying this operation's complete 512 KiB input shard alongside its CBs.
    compute_grid = device.compute_with_storage_grid_size()
    pressure_tiles = (850 * 1024 * compute_grid.x * compute_grid.y + 2047) // 2048
    l1_pressure = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 32, pressure_tiles * 32)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    output = run_group_norm()
    actual = ttnn.to_torch(ttnn.from_device(output))

    assert_numeric_metrics(reference, actual, atol=0.043, frobenius_threshold=0.01)
    assert device.num_program_cache_entries() == entries_with_replay + 1
    assert l1_pressure.is_allocated()

    # Releasing the allocation must select the original replay program, not create another entry.
    output.deallocate(force=True)
    l1_pressure.deallocate(force=True)
    restored_output = run_group_norm()
    restored_actual = ttnn.to_torch(ttnn.from_device(restored_output))
    assert_numeric_metrics(reference, restored_actual, atol=0.043, frobenius_threshold=0.01)
    assert device.num_program_cache_entries() == entries_with_replay + 1


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("num_out_blocks", [13, 17, 73])
@pytest.mark.parametrize("cores_y", [1, 4, 8])
@pytest.mark.parametrize("C, cores_x, groups", [(64, 1, 2), (256, 8, 32)], ids=["strided", "bank_contiguous"])
def test_group_norm_streaming_stats_wrap_DRAM(
    device, enabled_program_cache, dtype, num_out_blocks, cores_y, C, cores_x, groups
):
    available_grid = device.compute_with_storage_grid_size()
    if cores_x > available_grid.x or cores_y > available_grid.y:
        pytest.skip("Requested streaming grid exceeds the available compute grid")
    # One core's input exceeds L1. Odd CB sizes and partial final blocks make
    # statistics batches split at wrap boundaries, including on the second batch.
    # The eight-bank Blackhole case also covers coalesced reads and reordered
    # reader columns, including affine/mask offsets and multi-row multicast groups.
    N = 2
    # The new eight-row, single-bank geometry retains at least 2 MiB of BF16
    # input per core, so even the smallest CB blocks cannot enable full replay.
    rows_per_core = 32768 if C == 256 and cores_y == 8 else 16384
    HW = rows_per_core * max(1, cores_y // N)
    grid = ttnn.CoreGrid(y=cores_y, x=cores_x)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    weight = torch.linspace(0.75, 1.25, C).to(torch_dtype).float()
    bias = torch.linspace(-0.25, 0.25, C).to(torch_dtype).float()
    [gamma, beta], mask = ttnn.dram_group_norm_params_from_torch(
        [weight, bias], C, groups, device, core_grid=grid, return_mask=True, dtype=dtype
    )
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    for seed in (13, 17):
        torch.manual_seed(seed)
        x = torch.randn((N, C, 1, HW)).to(torch_dtype).float()
        reference = torch.nn.functional.group_norm(x, groups, weight, bias, eps=1e-5)
        reference = reference.permute(0, 2, 3, 1).contiguous()
        input_tensor = ttnn.from_torch(
            x.permute(0, 2, 3, 1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.group_norm(
            input_tensor,
            num_groups=groups,
            input_mask=mask,
            weight=gamma,
            bias=beta,
            epsilon=1e-5,
            core_grid=grid,
            num_out_blocks=num_out_blocks,
            inplace=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            use_welford=True,
            compute_kernel_config=config,
        )
        actual = ttnn.to_torch(output).float()
        assert torch.isfinite(actual).all()
        # The 256-channel BF16 control reaches 0.051 absolute error with the original reader too.
        atol = 0.06 if C == 256 and dtype == ttnn.bfloat16 else 0.05
        assert_numeric_metrics(reference, actual, atol=atol, frobenius_threshold=0.01)
        output.deallocate(force=True)
        input_tensor.deallocate(force=True)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "N, cores_y, num_out_blocks, input_layout",
    [
        # The grid fixes num_virtual_rows, hence whether a batch is split across core rows and so
        # which cores hold the padding row-tile.
        pytest.param(1, 1, None, ttnn.TILE_LAYOUT, id="N1_grid1x8_whole_batch_per_core"),
        pytest.param(1, 2, None, ttnn.TILE_LAYOUT, id="N1_grid2x8_batch_split_2"),
        pytest.param(1, 4, None, ttnn.TILE_LAYOUT, id="N1_grid4x8_batch_split_4_block_h_1"),
        pytest.param(2, 1, None, ttnn.TILE_LAYOUT, id="N2_grid1x8_two_batches_per_core"),
        pytest.param(2, 4, None, ttnn.TILE_LAYOUT, id="N2_grid4x8_batch_split_2"),
        # num_out_blocks=3 over block_h=4 leaves the last out-block empty -- the case that breaks a
        # naive "last block, last row" test.
        pytest.param(1, 1, 3, ttnn.TILE_LAYOUT, id="N1_grid1x8_num_out_blocks_3_empty_last_block"),
        pytest.param(2, 1, 3, ttnn.TILE_LAYOUT, id="N2_grid1x8_num_out_blocks_3_empty_last_block"),
        pytest.param(1, 1, 3, ttnn.ROW_MAJOR_LAYOUT, id="N1_grid1x8_num_out_blocks_3_empty_last_block_row_major"),
        pytest.param(1, 1, 4, ttnn.TILE_LAYOUT, id="N1_grid1x8_num_out_blocks_4"),
    ],
)
def test_group_norm_non_tile_aligned_dirty_padding_grids_DRAM(device, N, cores_y, num_out_blocks, input_layout):
    # Which core applies the row mask depends on how the grid splits H*W, and with N > 1 the padding
    # recurs once per batch on the same core -- a single auto-selected grid reaches neither. Also
    # pins the out-block indexing: num_out_blocks not dividing block_h can leave the last out-block
    # with zero rows, so the final row-tile is found by its global index within the batch.
    if input_layout == ttnn.ROW_MAJOR_LAYOUT and device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        pytest.skip("Interleaved row-major GroupNorm is supported only on Wormhole")
    cores_x = 8
    if device.core_grid.y < cores_y or device.core_grid.x < cores_x:
        pytest.skip(f"device grid too small for {cores_x}x{cores_y}")

    torch.manual_seed(0)
    C, HW, G, padded = 512, 100, 32, 128

    real = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)
    ref = torch.nn.functional.group_norm(real.view(N, HW, C).permute(0, 2, 1).reshape(N, C, 1, HW).float(), G)
    ref = ref.permute(0, 2, 3, 1).reshape(N, 1, HW, C)

    buf = torch.zeros((N, 1, padded, C), dtype=torch.bfloat16)
    buf[:, :, :HW, :] = real
    buf[:, :, HW:, :] = 7.0

    tt = ttnn.from_torch(
        buf, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt = ttnn.reshape(tt, ttnn.Shape([N, 1, HW, C]), ttnn.Shape([N, 1, padded, C]))
    out = ttnn.group_norm(
        tt,
        num_groups=G,
        inplace=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=cores_y, x=cores_x),
        num_out_blocks=num_out_blocks,
    )
    out = ttnn.to_torch(ttnn.from_device(out)).float()[:, :, :HW, :]

    max_abs_err = (out - ref).abs().max().item()
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    logger.info(f"N={N} grid={cores_x}x{cores_y} num_out_blocks={num_out_blocks}: max_abs_err={max_abs_err} pcc={pcc}")
    assert max_abs_err < 0.08, (
        f"max abs error {max_abs_err} with dirty tile padding at N={N} grid={cores_x}x{cores_y} "
        f"num_out_blocks={num_out_blocks}; group_norm must be independent of its padding (see #52685)"
    )
    assert pcc > 0.999, f"pcc {pcc} with dirty tile padding (see #52685)"
