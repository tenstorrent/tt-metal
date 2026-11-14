# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Quasar matmul host-side config selection, bias routing, output shapes and guards (#57362)."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

qsr = ttnn._ttnn.operations.experimental.quasar


def _dram(t, device, dtype=ttnn.bfloat16, tile=None):
    kwargs = dict(tile=tile) if tile is not None else {}
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs
    )


def test_auto_config_a_batch_broadcast(device):
    """A batch 1 with B batch 4 must not auto-select an mcast_in0 1D config, which the validator rejects."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, 32, 512).bfloat16()
    b = torch.randn(1, 4, 512, 1024).bfloat16()
    out = ttnn.to_torch(ttnn.experimental.quasar.matmul(_dram(a, device), _dram(b, device)))
    assert_with_pcc(a.float() @ b.float(), out.float(), 0.999)


def test_linear_reuse_config_adds_bias(device):
    """MatmulMultiCoreReuseProgramConfig has no fused bias, so linear must add it after the matmul."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, 256, 256).bfloat16()
    b = torch.randn(1, 1, 256, 256).bfloat16()
    bias = (torch.randn(1, 1, 1, 256) * 10).bfloat16()
    program_config = qsr.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=8,
        per_core_N=8,
    )
    out = ttnn.experimental.quasar.linear(
        _dram(a, device), _dram(b, device), bias=_dram(bias, device), program_config=program_config
    )
    assert_with_pcc(a.float() @ b.float() + bias.float(), ttnn.to_torch(out).float(), 0.999)


def test_linear_transpose_a_with_bias_output_shape(device):
    """A config that transposes A on the host must still give the [K, N] output of A^T @ B."""
    torch.manual_seed(0)
    a = torch.randn(64, 128).bfloat16()
    b = torch.randn(64, 32).bfloat16()
    bias = torch.randn(1, 32).bfloat16()
    program_config = qsr.MatmulUnifiedProgramConfig(
        cores=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    )
    out = ttnn.to_torch(
        ttnn.experimental.quasar.linear(
            _dram(a, device),
            _dram(b, device),
            bias=_dram(bias, device),
            transpose_a=True,
            program_config=program_config,
        )
    ).float()
    expected = a.float().T @ b.float() + bias.float()
    assert out.shape == expected.shape
    assert_with_pcc(expected, out, 0.999)


@pytest.mark.parametrize(
    "a_shape, b_shape, bias_shape",
    [
        ((1, 32), (32, 32), (32, 32)),
        ((1, 1, 32, 64), (1, 1, 64, 64), (1, 4, 32, 64)),
    ],
)
def test_linear_reuse_config_bias_broadcasts_into_optional_output(device, a_shape, b_shape, bias_shape):
    """When the post-added bias broadcasts the result, the matmul must not write into the larger output."""
    torch.manual_seed(0)
    a = torch.randn(a_shape).bfloat16()
    b = torch.randn(b_shape).bfloat16()
    bias = torch.randn(bias_shape).bfloat16()
    expected = a.float() @ b.float() + bias.float()
    out = _dram(torch.zeros(expected.shape).bfloat16(), device)
    program_config = qsr.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=b_shape[-1] // 32,
    )
    ttnn.experimental.quasar.linear(
        _dram(a, device),
        _dram(b, device),
        bias=_dram(bias, device),
        program_config=program_config,
        optional_output_tensor=out,
    )
    assert_with_pcc(expected, ttnn.to_torch(out).float(), 0.999)


def test_sparse_matmul_rejects_multiple_m_blocks(device, expect_error):
    """mcast_in0 sends one M block to every core, so per_core_M must cover all of M."""
    torch.manual_seed(0)
    m, k, n, experts = 32, 128, 512, 4
    tile_h = 16
    in0 = _dram(torch.randn(1, 1, m, k).bfloat16(), device, tile=ttnn.Tile((tile_h, 32)))
    in1 = _dram(torch.randn(1, experts, k, n).bfloat16(), device, dtype=ttnn.bfloat8_b)
    sparsity = ttnn.from_torch(
        torch.ones(1, 1, 1, experts).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    program_config = qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=m // tile_h // 2,
        per_core_N=n // 32,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    with expect_error(RuntimeError, "requires num_blocks_y == 1"):
        ttnn.experimental.quasar.sparse_matmul(
            in0,
            in1,
            sparsity=sparsity,
            nnz=experts,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([tile_h, 32]),
            program_config=program_config,
        )


def test_batched_dram_sharded_rejects_split_k_with_multi_row_m(device, expect_error):
    """The in0 reader reads each K block as one contiguous run, which needs per_core_M == 1 or one K block."""
    workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(workers)
    tile = 32
    m = k = 2 * tile
    n = tile * num_banks
    worker_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in workers]
    )
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})

    def sharded(grid, buffer_type, shape):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type,
            ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    in0_t = ttnn.from_torch(
        torch.rand([1, num_banks, m, k], dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=sharded(worker_grid, ttnn.BufferType.L1, (m, k)),
    )
    in1_t = ttnn.from_torch(
        torch.rand([1, num_banks, k, n], dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=sharded(dram_grid, ttnn.BufferType.DRAM, (k, n)),
    )
    params = qsr.MatmulParams()
    params.program_config = qsr.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=1,
        per_core_M=m // tile,
        per_core_N=n // tile,
        fused_activation=None,
    )
    params.output_mem_config = sharded(worker_grid, ttnn.BufferType.L1, (m, n))
    params.output_dtype = ttnn.bfloat16
    attributes = qsr.create_matmul_attributes(in0_t, in1_t, params, [])
    tensor_args = qsr.MatmulInputs()
    tensor_args.input_tensors = [in0_t, in1_t]
    tensor_args.optional_input_tensors = [None]
    output = qsr.MatmulDeviceOperation.create_output_tensors(attributes, tensor_args)
    with expect_error(RuntimeError, "in0_block_w \\(1\\) must equal K \\(2\\)"):
        qsr.MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory.create_descriptor(attributes, tensor_args, output)
