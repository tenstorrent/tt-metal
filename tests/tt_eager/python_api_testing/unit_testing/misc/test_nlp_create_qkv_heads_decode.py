# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    skip_for_grayskull,
    nearest_32,
    is_blackhole,
    skip_for_blackhole,
)


def run_test_create_head_interleaved(device, n_local_heads, n_local_kv_heads, head_dim, batch, is_dram):
    ## Split Heads
    seq_len = 1
    total_heads = n_local_heads + n_local_kv_heads * 2
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG if is_dram else ttnn.L1_MEMORY_CONFIG
    # Prepare input
    proj_output = torch.rand(1, seq_len, batch, head_dim * total_heads)
    proj_output_tt = ttnn.from_torch(proj_output, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    proj_output_tt = proj_output_tt.to(device=device, mem_config=input_memory_config)

    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        proj_output_tt,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        memory_config=HEIGHT_SHARDED_MEMCFG,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.shape}, {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.shape}, {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.shape}, {v_heads_tt.memory_config()}")

    # torch operation
    q_heads_torch = proj_output[:, :, :batch, : head_dim * n_local_heads].view(seq_len, batch, n_local_heads, head_dim)
    k_heads_torch = proj_output[
        :, :, :batch, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)
    ].view(seq_len, batch, n_local_kv_heads, head_dim)
    v_heads_torch = proj_output[:, :, :batch, head_dim * (n_local_heads + n_local_kv_heads) :].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )

    # compare
    q_heads_tt_cpu = ttnn.to_torch(q_heads_tt)
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = ttnn.to_torch(k_heads_tt)
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = ttnn.to_torch(v_heads_tt)
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_q_heads, n_kv_heads, head_dim",
    (
        (64, 8, 128),
        (32, 8, 128),
        (8, 4, 96),
        (32, 8, 64),
    ),
    ids=["n64_8_128", "n32_8_128", "n8_4_96", "n32_8_64"],
)
@pytest.mark.parametrize("batch", (1, 2, 4, 8, 16, 32), ids=["b1", "b2", "b4", "b8", "b16", "b32"])
@pytest.mark.parametrize("parallel_factor", (1, 2, 4, 8), ids=["pf1", "pf2", "pf4", "pf8"])
@pytest.mark.parametrize("is_dram", (False, True), ids=["L1", "DRAM"])
def test_create_head_interleaved(
    n_q_heads,
    n_kv_heads,
    head_dim,
    batch,
    parallel_factor,
    device,
    is_dram,
):
    torch.manual_seed(0)
    n_local_heads = n_q_heads // parallel_factor
    n_local_kv_heads = n_kv_heads // parallel_factor
    if n_local_heads > 32 or n_local_kv_heads == 0:
        pytest.skip("Skipping due to impossible parallelization")
    if is_blackhole() and is_dram:
        pytest.skip("Skipping DRAM test on blackhole due to issue #16667")
    for i in range(3):
        # multiple loops to test program caching
        run_test_create_head_interleaved(device, n_local_heads, n_local_kv_heads, head_dim, batch, is_dram)


def run_test_create_head_max_width_shard(device, n_local_heads, n_local_kv_heads, head_dim, batch):
    ## Split Heads
    seq_len = 1
    total_heads = n_local_heads + n_local_kv_heads * 2
    # Prepare input
    proj_output = torch.rand(1, seq_len, batch, head_dim * total_heads)
    proj_output_tt = ttnn.from_torch(
        proj_output,
        layout=ttnn.TILE_LAYOUT,
    )
    # Use ttnn shape to get padding of batch
    padded_batch = proj_output_tt.padded_shape[2]
    shard_spec_1_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(0, 0),
            ),
        }
    )
    CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_1_cores_grid,
            [
                padded_batch,
                head_dim * total_heads,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    proj_output_tt = proj_output_tt.to(device=device, mem_config=CREATE_HEAD_INPUT_MEMCFG)

    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        proj_output_tt,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        memory_config=HEIGHT_SHARDED_MEMCFG,
        # unpadded_batch_size=batch if batch != padded_batch else None,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.shape}, {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.shape}, {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.shape}, {v_heads_tt.memory_config()}")

    # torch operation
    q_heads_torch = proj_output[:, :, :batch, : head_dim * n_local_heads].view(seq_len, batch, n_local_heads, head_dim)
    k_heads_torch = proj_output[
        :, :, :batch, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)
    ].view(seq_len, batch, n_local_kv_heads, head_dim)
    v_heads_torch = proj_output[:, :, :batch, head_dim * (n_local_heads + n_local_kv_heads) :].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )

    # compare
    q_heads_tt_cpu = ttnn.to_torch(q_heads_tt)
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = ttnn.to_torch(k_heads_tt)
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = ttnn.to_torch(v_heads_tt)
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_local_heads, n_local_kv_heads, head_dim, batch",
    ((8, 1, 128, 32), (8, 4, 96, 32), (16, 2, 64, 32), (8, 1, 128, 16), (8, 1, 128, 8), (32, 8, 128, 4)),
)
def test_create_head_max_width_shard(
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    batch,
    device,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_create_head_max_width_shard(device, n_local_heads, n_local_kv_heads, head_dim, batch)


def run_test_create_min_width_shard(
    device,
    batch,
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    overlap_coregrid,
    batch_offset=None,
    slice_size=None,
    sub_core_grids=None,
):
    # Split Heads
    if not overlap_coregrid and (slice_size >= 32 if slice_size is not None else batch >= 32):
        # Test with smaller batch size for CI to pass on devices not utlizing full coregrid
        pytest.skip(
            "Skipping tests for batch_per_device>=32 for non-overlapping coregrid as CI device does not support full coregrid"
        )
    seq_len = 1
    total_heads = n_local_heads + n_local_kv_heads * 2
    total_cores = total_heads * head_dim // 32

    device_core_grid_size = device.compute_with_storage_grid_size()
    device_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device_core_grid_size.x - 1, device_core_grid_size.y - 1),
            ),
        }
    )
    if sub_core_grids is not None:
        sub_core_grids_bounds = sub_core_grids.bounding_box()
        if (
            sub_core_grids_bounds.start.x < 0
            or sub_core_grids_bounds.start.y < 0
            or sub_core_grids_bounds.end.x >= device_core_grid_size.x
            or sub_core_grids_bounds.end.y >= device_core_grid_size.y
        ):
            pytest.skip("Sub core grid is out of bounds")
        device_core_grid = sub_core_grids

    device_core_grid_start = device_core_grid.bounding_box().start
    input_shard_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        device_core_grid_start, total_cores, device_core_grid, True
    )

    CREATE_HEAD_INPUT_SHARD_SPEC = ttnn.ShardSpec(
        input_shard_core_grid,
        [
            32,
            32,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, CREATE_HEAD_INPUT_SHARD_SPEC
    )

    CREATE_HEAD_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            device_core_grid,
            [
                32,
                head_dim,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Prepare tt input
    proj_output = torch.rand(1, seq_len, batch, head_dim * total_heads)
    proj_output_tt = ttnn.from_torch(
        proj_output, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=CREATE_HEAD_INPUT_MEMCFG
    )

    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        proj_output_tt,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        overlap_qk_coregrid=overlap_coregrid,
        batch_offset=batch_offset,
        slice_size=slice_size,
        memory_config=CREATE_HEAD_OUTPUT_MEMCFG,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.shape}, {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.shape}, {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.shape}, {v_heads_tt.memory_config()}")

    if batch_offset is None and slice_size is None:
        batch_offset = 0
        slice_size = batch
    else:
        if isinstance(batch_offset, ttnn.Tensor):
            # convert ttnn.Tensor to torch tensor
            tensor = ttnn.to_torch(batch_offset)
            batch_offset = tensor[0]
        batch = slice_size
    q_heads_torch = proj_output[:, :, batch_offset : batch_offset + slice_size, : head_dim * n_local_heads].view(
        seq_len, batch, n_local_heads, head_dim
    )
    k_heads_torch = proj_output[
        :,
        :,
        batch_offset : batch_offset + slice_size,
        head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads),
    ].view(seq_len, batch, n_local_kv_heads, head_dim)
    v_heads_torch = proj_output[
        :, :, batch_offset : batch_offset + slice_size, head_dim * (n_local_heads + n_local_kv_heads) :
    ].view(seq_len, batch, n_local_kv_heads, head_dim)

    # compare
    q_heads_tt_cpu = ttnn.to_torch(q_heads_tt)
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = ttnn.to_torch(k_heads_tt)
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = ttnn.to_torch(v_heads_tt)
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("batch", (1, 8, 16, 32))
@pytest.mark.parametrize(
    "n_local_heads, n_local_kv_heads, head_dim",
    ((8, 1, 128), (8, 4, 96), (16, 2, 64)),
)
@pytest.mark.parametrize("overlap_coregrid", (True, False))
def test_create_min_width_shard(
    batch,
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    device,
    overlap_coregrid,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_create_min_width_shard(
            device=device,
            batch=batch,
            n_local_heads=n_local_heads,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            overlap_coregrid=overlap_coregrid,
        )

    expected_entries = 1
    assert device.num_program_cache_entries() == expected_entries


@skip_for_blackhole("Requires eth connected devices to run, see #12349")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("batch", (32,))
@pytest.mark.parametrize(
    "n_local_heads, n_local_kv_heads, head_dim",
    ((8, 1, 128),),
)
@pytest.mark.parametrize("overlap_coregrid", (True, False))
@pytest.mark.parametrize("batch_offset", (0, 8, 16, 24))
@pytest.mark.parametrize("slice_size", (8,))
def test_create_heads_with_slice(
    batch,
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    device,
    overlap_coregrid,
    batch_offset,
    slice_size,
):
    torch.manual_seed(0)
    batch_offset_tensor = torch.tensor([batch_offset], dtype=torch.int32)
    # convert to tt tensor
    batch_offset_tensor_tt = ttnn.from_torch(batch_offset_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    for i in range(3):
        # multiple loops to test program caching
        run_test_create_min_width_shard(
            device=device,
            batch=batch,
            n_local_heads=n_local_heads,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            overlap_coregrid=overlap_coregrid,
            batch_offset=batch_offset_tensor_tt,
            slice_size=slice_size,
        )
    # BH does s2i and i2s inside of to_device and from_device as device ops
    expected_entries = 1
    assert device.num_program_cache_entries() == expected_entries


@skip_for_blackhole("Requires eth connected devices to run, see #12349")
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("batch", (1, 8, 16))
@pytest.mark.parametrize(
    "n_local_heads, n_local_kv_heads, head_dim",
    ((8, 1, 128), (16, 2, 64)),
)
@pytest.mark.parametrize("overlap_coregrid", (True, False))
@pytest.mark.parametrize(
    "sub_core_grids",
    (
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            }
        ),
    ),
)
def test_create_min_width_shard_subcoregrid(
    device,
    batch,
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    overlap_coregrid,
    sub_core_grids,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_create_min_width_shard(
            device=device,
            batch=batch,
            n_local_heads=n_local_heads,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            overlap_coregrid=overlap_coregrid,
            sub_core_grids=sub_core_grids,
        )
    assert device.num_program_cache_entries() == 1, "Only one Op program cache should exist"


def run_test_create_width_shard_by_head(
    device,
    n_local_heads,
    n_local_kv_heads,
    head_dim,
):
    ## Split Heads
    batch = 16
    seq_len = 1
    total_heads = n_local_heads + n_local_kv_heads * 2
    total_cores = total_heads
    core_x = min(total_cores, 8)
    core_y = max(1, total_cores // core_x)
    # Prepare input
    proj_output = torch.rand(1, seq_len, batch, head_dim * total_heads)

    # TT configs
    shard_spec_n_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(core_x - 1, core_y - 1),
            ),
        }
    )
    CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_n_cores_grid,
            [
                32,
                head_dim,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    # Prepare tt input
    proj_output_tt = ttnn.from_torch(proj_output, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16).to(
        device=device, mem_config=CREATE_HEAD_INPUT_MEMCFG
    )

    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        proj_output_tt,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        memory_config=HEIGHT_SHARDED_MEMCFG,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.shape}, {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.shape}, {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.shape}, {v_heads_tt.memory_config()}")

    # torch operation
    q_heads_torch = proj_output[:, :, :, : head_dim * n_local_heads].view(seq_len, batch, n_local_heads, head_dim)
    k_heads_torch = proj_output[:, :, :, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )
    v_heads_torch = proj_output[:, :, :, head_dim * (n_local_heads + n_local_kv_heads) :].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )

    # compare
    q_heads_tt_cpu = ttnn.to_torch(q_heads_tt)
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = ttnn.to_torch(k_heads_tt)
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = ttnn.to_torch(v_heads_tt)
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_local_heads, n_local_kv_heads, head_dim",
    ((32, 8, 128),),
)
def test_create_width_shard_by_head(
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    device,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_create_width_shard_by_head(
            device,
            n_local_heads,
            n_local_kv_heads,
            head_dim,
        )
