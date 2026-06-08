# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import run_for_blackhole

import tests.ttnn.unit_tests.operations.fused.test_group_norm as base


@pytest.mark.parametrize("specify_grid", [True, False])
def test_group_norm_large_ex_external_cb(device, specify_grid):
    torch.manual_seed(0)
    shape = (1, 1, 1280 * 720, 256)  # [N, 1, H*W, C]
    num_groups = 32
    eps = 1e-5

    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    weight = torch.randn((shape[-1],), dtype=torch.bfloat16)
    bias = torch.randn((shape[-1],), dtype=torch.bfloat16)
    c = shape[-1]
    weight_4d = weight.reshape(1, 1, c // 32, 32)
    bias_4d = bias.reshape(1, 1, c // 32, 32)

    # GroupNorm golden: convert [N,1,H*W,C] -> [N,C,1,H*W], apply GN, convert back.
    input_tensor_nchw = input_tensor.permute(0, 3, 1, 2).float()
    golden = torch.nn.functional.group_norm(
        input_tensor_nchw, num_groups=num_groups, weight=weight.float(), bias=bias.float(), eps=eps
    ).permute(0, 2, 3, 1)

    input_tensor_tt = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    w_tt = ttnn.from_torch(weight_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = ttnn.from_torch(bias_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    grid_size = None
    if specify_grid:
        sharded_mem_config, grid_size = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=device,
            num_channels=c,
            num_groups=num_groups,
            input_nhw=1280 * 720,
            is_height_sharded=False,
            is_row_major=False,
        )

    output_tensor_tt = ttnn.group_norm(
        input_tensor_tt,
        num_groups=num_groups,
        epsilon=eps,
        weight=w_tt,
        bias=b_tt,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=-1 if specify_grid else None,
    )
    output_tensor = ttnn.to_torch(output_tensor_tt)
    assert_numeric_metrics(
        golden,
        output_tensor,
        pcc_threshold=0.999,
        rtol=10.519,
        atol=0.322,
        frobenius_threshold=0.043,
    )


@pytest.mark.parametrize("specify_grid", [True, False])
def test_group_norm_sharded_ex_external_cb_gap(device, specify_grid):
    """Sharded analog of test_group_norm_large_ex_external_cb.

    Stresses cb_ex_external gap-byte zeroing on the sharded reader path
    (reader_mcast_sender_unary_sharded_gn_v2.cpp). cb_ex_external is sized as
    a single tile, into which each per-core slot writes only datum_size_bytes
    at a 16-byte pitch; the rest of the tile is read by the downstream
    reduce_tile sum and must be zero.

    Both gap-byte regions structurally exist in the reserved tile; this
    shape exercises both:
      (A) Intra-slot gap: bfloat16 input (datum_size_bytes == 2 < 16) means
          each per-slot remote-core read writes only 2 bytes into its
          16-byte slot, leaving 14 bytes per remote-core slot
          (slots 1..num_mcast_cores-1) not covered by the per-slot writes.
          Slot 0 is the local core's slot and is fully covered by the
          full-tile SELF read described below, so it does not contribute
          an intra-slot gap.
      (B) Trailing tile gap: with num_mcast_cores == grid.y == 4 the
          per-core slots occupy 4 * 16 == 64 bytes of the 2048-byte tile
          (bf16 single-tile size, which is what cb_ex_external currently
          uses because the host op hard-codes program_config.im_data_format
          to BFLOAT16 in groupnorm.cpp). The remaining 1984 bytes are also
          not covered by the per-slot writes. If im_data_format ever
          changes (e.g. to a wider intermediate dtype), single_tile_size
          and therefore the trailing-gap byte count change with it, so
          the specific 64-of-2048 numbers here are tied to today's
          hard-coded bf16 intermediate format. The test also forces
          use_welford=False because welford skips cb_ex_external entirely
          (see the matching guard in groupnorm_mcast_program_factory.cpp)
          and would not exercise this gap at all.

    The sharded reader currently relies on the documented packer-zeroing
    contract of `reduce<…, REDUCE_SCALAR>` (see reduce.h's reduce_init doc)
    to keep those gap bytes zero. The self read from cb_ex_partial is a
    full single_tile_size_bytes copy that uses cb_ex_partial's
    packer-zeroed non-result datums as a free zero-init for cb_ex_external.
    Each group runs two cb_ex_external aggregation passes, one for the
    partial mean (E[x]) and one for the partial variance, and this shape
    has multiple groups per core, so the producer cycles through the
    cb_ex_external SRAM region many times. Any breakage of that zeroing
    (e.g. switching the cb_ex_partial producer to a non-REDUCE_SCALAR pack
    without updating the reader) would corrupt the per-group mean/var
    reduction enough to fail the tight numeric tolerances below.
    """
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=4, x=8)

    N, C, H, W, num_groups = 1, 1280, 16, 16, 32
    eps = 1e-5

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=eps
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Block-shard input across grid: COL_MAJOR orientation puts cores in the
    # same column on the same channel slice, so the partial-sum reduction
    # (cb_ex_partial -> cb_ex_external) mcasts across grid_y == 4 cores.
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        epsilon=eps,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size if specify_grid else None,
        use_welford=False,  # exercise the v2 reader (which has the cb_ex_external slot pattern); welford uses a different reader.
    )

    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.9999,
        rtol=0.04,
        atol=0.04,
        frobenius_threshold=0.01,
    )


# ---------------------------------------------------------------------------
# Nightly wrappers: run each unit-test shape with specify_grid=False so the
# auto-grid path is exercised without bloating the regular CI pipeline.
# All parametrize data is sourced from base.<CONST> / base.<func>() so
# nightly stays in sync when unit-test parameters change.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N, C, H, W, num_groups", base.HEIGHT_SHARDED_SHAPES)
@pytest.mark.parametrize("use_welford", base.welford_flavors, ids=base.welford_ids)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_with_height_sharded(device, N, C, H, W, num_groups, use_welford, specify_grid):
    base.test_group_norm_with_height_sharded(device, N, C, H, W, num_groups, use_welford, specify_grid)


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", base.BLOCK_SHARDED_V2_8X4_SHAPES)
@pytest.mark.parametrize("use_welford", base.welford_flavors, ids=base.welford_ids)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_with_block_sharded_v2_8x4_grid(device, N, C, H, W, num_groups, use_welford, specify_grid):
    base.test_group_norm_with_block_sharded_v2_8x4_grid(device, N, C, H, W, num_groups, use_welford, specify_grid)


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", base.BLOCK_SHARDED_V2_8X8_SHAPES)
@pytest.mark.parametrize("use_welford", base.welford_flavors, ids=base.welford_ids)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_with_block_sharded_v2_8x8_grid(device, N, C, H, W, num_groups, use_welford, specify_grid):
    base.test_group_norm_with_block_sharded_v2_8x8_grid(device, N, C, H, W, num_groups, use_welford, specify_grid)


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", base.BLOCK_SHARDED_V2_8X8_TILE_LAYOUT_SHAPES)
@pytest.mark.parametrize("use_welford", base.welford_flavors, ids=base.welford_ids)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_with_block_sharded_v2_8x8_grid_tile_layout(
    device, N, C, H, W, num_groups, use_welford, specify_grid
):
    base.test_group_norm_with_block_sharded_v2_8x8_grid_tile_layout(
        device, N, C, H, W, num_groups, use_welford, specify_grid
    )


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("input_shape", base.generate_sdxl_test_inputs())
@pytest.mark.parametrize("use_welford", base.welford_flavors, ids=base.welford_ids)
@pytest.mark.parametrize("specify_grid", [False])
# Paramemeters need to stay consistent with usage in
# models/demos/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py::test_block_sharded_group_norm_sdxl_performance
def test_sdxl_base_group_norm(device, input_shape, use_welford, specify_grid):
    base.test_sdxl_base_group_norm(device, input_shape, use_welford, specify_grid)


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("input_shape", base.generate_sdxl_test_inputs())
@pytest.mark.parametrize("use_welford", base.welford_flavors, ids=base.welford_ids)
@pytest.mark.parametrize("specify_grid", [False])
# Oppositive of previous test in terms of inplace, for full coverage purposes.
def test_sdxl_group_norm_reverse_inplace(device, input_shape, use_welford, specify_grid):
    base.test_sdxl_group_norm_reverse_inplace(device, input_shape, use_welford, specify_grid)


@pytest.mark.parametrize("input_shape", base.SDXL_BASE_GROUP_NORM_BH_SHAPES)
@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("specify_grid", [False])
@run_for_blackhole("blackhole specific tests")
def test_sdxl_base_group_norm_bh(device, input_shape, specify_grid):
    base.test_sdxl_base_group_norm_bh(device, input_shape, specify_grid)


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE_SDXL_BG_N_MASK, indirect=True)
@pytest.mark.parametrize("input_shape", base.generate_sdxl_test_inputs_neg_mask())
@pytest.mark.parametrize("specify_grid", [False])
def test_sdxl_base_group_norm_negative_mask(device, input_shape, specify_grid):
    base.test_sdxl_base_group_norm_negative_mask(device, input_shape, specify_grid)


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", base.COMPUTE_CONFIG_SHAPES)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_compute_config(device, N, C, H, W, num_groups, specify_grid):
    base.test_group_norm_compute_config(device, N, C, H, W, num_groups, specify_grid)


@pytest.mark.parametrize("N, C, H, W, num_groups, shard, eps, use_negative_mask", base.GROUP_NORM_OFT_PARAMS)
@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("specify_grid", [False])
@run_for_blackhole("blackhole specific tests")
def test_group_norm_oft(device, N, C, H, W, num_groups, shard, eps, use_negative_mask, specify_grid):
    base.test_group_norm_oft(device, N, C, H, W, num_groups, shard, eps, use_negative_mask, specify_grid)


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", base.NO_INPUT_MASK_SHAPES)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_no_input_mask(device, N, C, H, W, num_groups, specify_grid):
    base.test_group_norm_no_input_mask(device, N, C, H, W, num_groups, specify_grid)


@pytest.mark.parametrize("N, C, H, W, num_groups", base.DRAM_GRID_SIZE_SHAPES)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_dram_grid_size(device, N, C, H, W, num_groups, specify_grid):
    base.test_group_norm_dram_grid_size(device, N, C, H, W, num_groups, specify_grid)


@pytest.mark.parametrize("N, C, H, W, num_groups", base.OPTIONAL_WEIGHT_BIAS_SHAPES)
@pytest.mark.parametrize("use_welford", base.welford_flavors, ids=base.welford_ids)
@pytest.mark.parametrize(
    "has_weight, has_bias", base.OPTIONAL_WEIGHT_BIAS_AFFINE_PARAMS, ids=base.OPTIONAL_WEIGHT_BIAS_AFFINE_IDS
)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_optional_weight_bias(
    device, N, C, H, W, num_groups, use_welford, has_weight, has_bias, specify_grid
):
    base.test_group_norm_optional_weight_bias(
        device, N, C, H, W, num_groups, use_welford, has_weight, has_bias, specify_grid
    )
