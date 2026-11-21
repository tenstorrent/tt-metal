# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for matmul with batch dimension mismatch.

test_matmul_a_batch1_b_batched: A batch=1, B batch>1 (issue #12834)
test_matmul_batch_mismatch:     A batch>1, B batch=1 (prior regression)
test_matmul_batch_broadcast*:   torch-style batch broadcasting, per program config
test_linear_batch_broadcast*:   the same with a bias
test_matmul_batch_not_broadcastable: batch dims torch would reject
"""

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((1, 1, 128, 2048), (1, 7, 2048, 256)),
        ((1, 128, 2048), (7, 2048, 256)),
        ((1, 1, 256, 512), (1, 3, 512, 64)),
        ((1, 64, 768), (5, 768, 192)),
        ((1, 1, 1, 128, 512), (2, 3, 4, 512, 64)),
        ((1, 1, 1, 1, 64, 512), (2, 3, 4, 5, 512, 128)),
    ],
)
def test_matmul_a_batch1_b_batched(a_shape, b_shape, device):
    """
    Test ttnn.matmul where A has batch=1 and B has batch>1 (issue #12834).
    Previously crashed with "bmm expects input tensors of shapes BCMK*BCKN=BCMN".
    The auto-selector picks a config whose readers broadcast A across B's batches.
    """
    torch.manual_seed(0)
    a_torch = torch.randn(*a_shape)
    b_torch = torch.randn(*b_shape)

    a_ttnn = ttnn.from_torch(a_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(b_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.matmul(a_ttnn, b_ttnn)
    ttnn.synchronize_device(device)

    expected_shape = list(torch.matmul(a_torch, b_torch).shape)
    assert list(output.shape) == expected_shape

    output_torch = ttnn.to_torch(output).to(torch.float32)
    expected = torch.matmul(a_torch, b_torch)

    pcc_passed, pcc_message = comp_pcc(expected, output_torch, 0.99)
    assert pcc_passed, f"PCC check failed: {pcc_message}"


@pytest.mark.parametrize(
    "batch, M, K, N",
    [
        (6, 3000, 256, 256),
    ],
)
def test_matmul_batch_mismatch(batch, M, K, N, device):
    """
    Test ttnn.linear with batch dimension mismatch (input batch >= 1, weight batch = 1).

    This specifically tests the case where:
    - Input: [batch, M, K]
    - Weight: [K, N] (no batch dimension, implicitly batch=1)

    Previously, batch=6 would hang in the matmul kernel due to incorrect argument passed to the in1 reader receiver kernel.
    """
    torch.manual_seed(0)
    # Create input tensor [batch, M, K]
    input_torch = torch.randn(batch, M, K)
    input_ttnn = ttnn.from_torch(input_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Create weight tensor [K, N]
    weight_torch = torch.randn(K, N)
    weight_ttnn = ttnn.from_torch(weight_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Create bias tensor [N]
    bias_torch = torch.randn(N)
    bias_ttnn = ttnn.from_torch(bias_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Run ttnn.linear (internally uses matmul)
    output = ttnn.linear(input_ttnn, weight_ttnn, bias=bias_ttnn)

    # Ensure operation completes (this was the hang point for batch=6)
    ttnn.synchronize_device(device)

    # Verify output shape
    assert list(output.shape) == [batch, M, N]

    # Convert back to torch for correctness check
    output_torch = ttnn.to_torch(output).to(torch.float32)

    # Compute reference
    expected = torch.matmul(input_torch, weight_torch) + bias_torch

    # Verify correctness using PCC (allow for numerical differences due to bfloat16)
    pcc_passed, pcc_message = comp_pcc(expected, output_torch, 0.99)
    assert pcc_passed, f"PCC check failed: {pcc_message}"


@pytest.mark.parametrize("preallocated", [False, True], ids=["new_output", "preallocated_output"])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "a_shape, b_shape, transpose_a, transpose_b",
    [
        # Torch-style broadcasting shapes from test_moreh_matmul.py
        ((3, 1, 2, 1, 4, 1, 319, 95), (4, 2, 95, 470), False, False),
        ((2, 319, 95), (2, 1, 3, 4, 1, 95, 470), False, False),
        ((3, 1, 2, 1, 4, 1, 95, 319), (4, 2, 95, 470), True, False),
        ((2, 319, 95), (2, 1, 3, 4, 1, 470, 95), False, True),
        ((2, 3, 1, 2, 3, 2, 64, 64), (2, 1, 4, 2, 1, 2, 64, 64), False, False),
        # Both sides broadcast, B of higher rank, A unbatched
        ((4, 1, 64, 128), (1, 3, 128, 96), False, False),
        ((2, 1, 64, 512), (1, 3, 512, 64), False, False),
        ((4, 1, 256, 512), (1, 8, 512, 256), False, False),
        ((2, 3, 64, 128), (3, 128, 64), False, False),
        ((64, 128), (6, 96, 128), False, True),
        # An odd K tile count selects the MultiCore config on L1
        ((2, 1, 64, 96), (1, 3, 96, 64), False, False),
        ((64, 96), (2, 3, 96, 64), False, False),
    ],
)
def test_matmul_batch_broadcast(a_shape, b_shape, transpose_a, transpose_b, memory_config, preallocated, device):
    """ttnn.matmul broadcasts batch dims like torch.matmul: aligned from the right, size-1 dims expand."""
    torch.manual_seed(0)
    a_torch = torch.randn(a_shape, dtype=torch.bfloat16)
    b_torch = torch.randn(b_shape, dtype=torch.bfloat16)
    expected = torch.matmul(
        a_torch.float().transpose(-1, -2) if transpose_a else a_torch.float(),
        b_torch.float().transpose(-1, -2) if transpose_b else b_torch.float(),
    )

    a_ttnn = ttnn.from_torch(a_torch, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    b_ttnn = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    output_ttnn = None
    if preallocated:
        output_ttnn = ttnn.from_torch(
            torch.zeros(expected.shape, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )
    output = ttnn.matmul(
        a_ttnn, b_ttnn, transpose_a=transpose_a, transpose_b=transpose_b, optional_output_tensor=output_ttnn
    )

    assert list(output.shape) == list(expected.shape)
    pcc_passed, pcc_message = comp_pcc(expected, ttnn.to_torch(output).float(), 0.999)
    assert pcc_passed, f"PCC check failed: {pcc_message}"


def _mcast_1d_config(mcast_in0, fused_activation=None):
    # Two cores, splitting N when A is multicast and M when B is.
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(2, 1),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2 if mcast_in0 else 1,
        per_core_N=1 if mcast_in0 else 2,
        fuse_batch=False,
        fused_activation=fused_activation,
        mcast_in0=mcast_in0,
    )


# 6 batches x 2 M blocks over 5 cores, so some cores start mid-batch.
_REUSE_CONFIG = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=(5, 1),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=2,
    per_core_N=2,
)

# 6 batches x 2 M blocks with 3 batches per core, so per_core_M spans batches of the broadcast output.
_REUSE_MULTI_BATCH_CONFIG = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=(2, 1),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=6,
    per_core_N=2,
)

# 2 M blocks x 2 N blocks on a 2x2 grid, batches not fused.
_MCAST_2D_CONFIG = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(2, 2),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=1,
    per_core_M=1,
    per_core_N=1,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)


def _assert_pcc_per_batch(expected, output):
    # Each output batch is checked on its own, so one missing or wrong batch can't hide in the total.
    expected = expected.reshape(-1, *expected.shape[-2:])
    output = output.reshape(-1, *output.shape[-2:])
    for batch in range(expected.shape[0]):
        pcc_passed, pcc_message = comp_pcc(expected[batch], output[batch], 0.999)
        assert pcc_passed, f"PCC check failed for batch {batch}: {pcc_message}"


@pytest.mark.parametrize(
    "a_shape, b_shape, transpose_a, transpose_b, program_config",
    [
        ((2, 1, 128, 64), (1, 3, 64, 64), False, False, _REUSE_CONFIG),
        ((2, 1, 64, 128), (1, 3, 64, 64), True, False, _REUSE_CONFIG),
        ((2, 1, 128, 64), (1, 3, 64, 64), False, True, _REUSE_CONFIG),
        ((2, 1, 64, 64), (1, 3, 64, 64), False, False, _REUSE_MULTI_BATCH_CONFIG),
        ((2, 1, 64, 128), (1, 3, 128, 64), False, False, _mcast_1d_config(mcast_in0=True)),
        ((1, 1, 64, 128), (2, 3, 128, 64), False, False, _mcast_1d_config(mcast_in0=True)),
        ((2, 1, 128, 64), (1, 3, 128, 64), True, False, _mcast_1d_config(mcast_in0=True)),
        ((2, 1, 64, 128), (1, 3, 64, 128), False, True, _mcast_1d_config(mcast_in0=True)),
        ((2, 1, 64, 128), (1, 3, 128, 64), False, False, _mcast_1d_config(mcast_in0=False)),
        ((1, 1, 64, 128), (2, 3, 128, 64), False, False, _mcast_1d_config(mcast_in0=False)),
        ((2, 1, 128, 64), (1, 3, 128, 64), True, False, _mcast_1d_config(mcast_in0=False)),
        ((2, 1, 64, 128), (1, 3, 64, 128), False, True, _mcast_1d_config(mcast_in0=False)),
        ((2, 1, 64, 128), (1, 3, 128, 64), False, False, _MCAST_2D_CONFIG),
        ((1, 1, 64, 128), (2, 3, 128, 64), False, False, _MCAST_2D_CONFIG),
        ((2, 3, 64, 128), (1, 1, 128, 64), False, False, _MCAST_2D_CONFIG),
    ],
    ids=[
        "reuse",
        "reuse_transpose_a",
        "reuse_transpose_b",
        "reuse_multi_batch_per_core",
        "mcast_in0",
        "mcast_in0_a_batch1",
        "mcast_in0_transpose_a",
        "mcast_in0_transpose_b",
        "mcast_in1",
        "mcast_in1_a_batch1",
        "mcast_in1_transpose_a",
        "mcast_in1_transpose_b",
        "mcast_2d",
        "mcast_2d_a_batch1",
        "mcast_2d_b_batch1",
    ],
)
def test_matmul_batch_broadcast_program_config(a_shape, b_shape, transpose_a, transpose_b, program_config, device):
    """Batch broadcasting with program configs the auto-selector would not pick for these shapes."""
    torch.manual_seed(0)
    a_torch = torch.randn(a_shape, dtype=torch.bfloat16)
    b_torch = torch.randn(b_shape, dtype=torch.bfloat16)
    expected = torch.matmul(
        a_torch.float().transpose(-1, -2) if transpose_a else a_torch.float(),
        b_torch.float().transpose(-1, -2) if transpose_b else b_torch.float(),
    )

    a_ttnn = ttnn.from_torch(a_torch, device=device, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT)
    output = ttnn.matmul(
        a_ttnn, b_ttnn, transpose_a=transpose_a, transpose_b=transpose_b, program_config=program_config
    )

    assert list(output.shape) == list(expected.shape)
    _assert_pcc_per_batch(expected, ttnn.to_torch(output).float())


@pytest.mark.parametrize("mcast_in0", [True, False], ids=["mcast_in0", "mcast_in1"])
def test_matmul_batch_broadcast_fused_activation(mcast_in0, device):
    """A fused activation applies to every output batch when a batch-1 A is broadcast."""
    torch.manual_seed(0)
    a_torch = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16)
    b_torch = torch.randn((2, 3, 128, 64), dtype=torch.bfloat16)
    expected = torch.relu(torch.matmul(a_torch.float(), b_torch.float()))

    program_config = _mcast_1d_config(mcast_in0, fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU))
    a_ttnn = ttnn.from_torch(a_torch, device=device, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT)
    output = ttnn.matmul(a_ttnn, b_ttnn, program_config=program_config)

    assert list(output.shape) == list(expected.shape)
    _assert_pcc_per_batch(expected, ttnn.to_torch(output).float())


@pytest.mark.parametrize("preallocated", [False, True], ids=["new_output", "preallocated_output"])
def test_linear_batch_broadcast_with_bias(preallocated, device):
    """ttnn.linear with a bias and a batched weight of higher rank than the input."""
    torch.manual_seed(0)
    a_torch = torch.randn((2, 64, 128), dtype=torch.bfloat16)
    b_torch = torch.randn((4, 2, 128, 96), dtype=torch.bfloat16)
    bias_torch = torch.randn((96,), dtype=torch.bfloat16)
    expected = torch.matmul(a_torch.float(), b_torch.float()) + bias_torch.float()

    a_ttnn = ttnn.from_torch(a_torch, device=device, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT)
    bias_ttnn = ttnn.from_torch(bias_torch, device=device, layout=ttnn.TILE_LAYOUT)
    output_ttnn = None
    if preallocated:
        output_ttnn = ttnn.from_torch(
            torch.zeros(expected.shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT
        )
    output = ttnn.linear(a_ttnn, b_ttnn, bias=bias_ttnn, optional_output_tensor=output_ttnn)

    assert list(output.shape) == list(expected.shape)
    _assert_pcc_per_batch(expected, ttnn.to_torch(output).float())


@pytest.mark.parametrize("a_shape, b_shape", [((2, 32, 64), (3, 64, 32)), ((2, 3, 32, 64), (4, 64, 32))])
def test_matmul_batch_not_broadcastable(a_shape, b_shape, device, expect_error):
    """Batch dims that differ and are both larger than 1 are rejected, as in torch."""
    a_ttnn = ttnn.from_torch(torch.randn(a_shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(torch.randn(b_shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "not broadcastable"):
        ttnn.matmul(a_ttnn, b_ttnn)


def test_matmul_batch_broadcast_sharded_input(device, expect_error):
    """Broadcasting reads inputs from interleaved memory, so a sharded input is rejected up front."""
    sharded = ttnn.create_sharded_memory_config(
        (2, 64, 64),
        core_grid=ttnn.CoreGrid(y=1, x=2),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    a_ttnn = ttnn.from_torch(
        torch.randn((2, 64, 64), dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, memory_config=sharded
    )
    b_ttnn = ttnn.from_torch(torch.randn((3, 2, 64, 64), dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "non-sharded inputs and output"):
        ttnn.matmul(a_ttnn, b_ttnn)


def test_matmul_batch_broadcast_sharded_output(device, expect_error):
    """A sharded output is sized from A's batches, so a batch-1 A against 4 B batches is rejected, not cut short."""
    sharded = ttnn.create_sharded_memory_config(
        (64, 64),
        core_grid=ttnn.CoreGrid(y=1, x=4),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(4, 1),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2,
        per_core_N=2,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
    )
    a_ttnn = ttnn.from_torch(torch.randn((1, 1, 64, 64), dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(torch.randn((1, 4, 64, 64), dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "Number of shards along height"):
        ttnn.matmul(a_ttnn, b_ttnn, program_config=program_config, memory_config=sharded)


@pytest.mark.parametrize("a_shape, b_shape", [((0, 32, 64), (1, 64, 32)), ((1, 32, 64), (0, 64, 32))])
def test_matmul_batch_broadcast_empty_batch(a_shape, b_shape, device):
    """A size-0 batch dim against a size-1 one broadcasts to 0, so the output stays empty."""
    a_ttnn = ttnn.from_torch(torch.randn(a_shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(torch.randn(b_shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    output = ttnn.matmul(a_ttnn, b_ttnn)

    assert list(output.shape) == list(torch.matmul(torch.empty(a_shape), torch.empty(b_shape)).shape)
