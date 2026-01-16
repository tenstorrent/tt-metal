# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose


TILE_SIZE = ttnn.TILE_SIZE


def chunk_and_shard_tensor(tensor, num_simulated_devices, device, dim, dtype, layout=ttnn.TILE_LAYOUT):
    """Chunk a tensor and convert each chunk to a ttnn tensor."""
    chunked = torch.chunk(tensor, num_simulated_devices, dim)
    result = [
        ttnn.from_torch(chunk, dtype=dtype, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunked
    ]
    return result


def prepare_row_major_weight_bias(torch_weight_1d, num_simulated_devices):
    """
    Prepare weight/bias tensor for ROW_MAJOR layout, matching DistributedLayerNorm._prepare_torch_state.

    This reshapes the 1D weight/bias tensor to interleave across devices in the specific
    pattern expected by the distributed layernorm operation.

    Args:
        torch_weight_1d: 1D tensor of shape (embedding_dim,)
        num_simulated_devices: Number of simulated TP devices

    Returns:
        Reshaped tensor of shape (embedding_dim // (TILE_SIZE * num_simulated_devices), TILE_SIZE * num_simulated_devices)
    """
    # Reshape to interleave: [mesh_width, -1, TILE_SIZE] -> permute -> reshape
    # This matches: weight.reshape(mesh_width, -1, TILE_SIZE).permute(1, 0, 2).reshape(-1, TILE_SIZE * mesh_width)
    reshaped = (
        torch_weight_1d.reshape(num_simulated_devices, -1, TILE_SIZE)
        .permute(1, 0, 2)
        .reshape(-1, TILE_SIZE * num_simulated_devices)
    )
    return reshaped


def run_distributed_dit_layernorm(
    device,
    num_simulated_devices,
    inp_shape,
    dtype,
    stats_dtype,
    use_affine=True,
    weight_bias_layout=ttnn.TILE_LAYOUT,
):
    """
    Test distributed DIT layernorm by simulating multi-device operation on a single device.

    The test:
    1. Chunks input across simulated devices
    2. Runs pre_allgather on each chunk to get per-device stats
    3. Concatenates stats (simulating all-gather)
    4. Runs post_allgather on each chunk with gathered stats
    5. Concatenates output and compares to torch reference

    Args:
        device: The ttnn device
        num_simulated_devices: Number of TP devices to simulate
        inp_shape: Input tensor shape (batch, channel, seq_len, embedding_dim)
        dtype: Input data type
        stats_dtype: Stats data type
        use_affine: Whether to use weight and bias
        weight_bias_layout: Layout for weight/bias tensors (TILE_LAYOUT or ROW_MAJOR_LAYOUT)
    """
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    torch.manual_seed(1234)

    embedding_dim = inp_shape[-1]
    torch_input = torch.randn(inp_shape) * 4 - 1

    torch_weight = None
    torch_bias = None
    if use_affine:
        # Create 1D weight and bias tensors
        torch_weight = torch.rand(embedding_dim)
        torch_bias = torch.rand(embedding_dim)

    epsilon = 1e-5

    # Chunk input across simulated devices on the last dimension
    tt_inp = chunk_and_shard_tensor(torch_input, num_simulated_devices, device, -1, dtype)

    tt_weight = None
    tt_bias = None
    if use_affine:
        if weight_bias_layout == ttnn.ROW_MAJOR_LAYOUT:
            # For ROW_MAJOR, reshape weight/bias using the interleaved pattern
            # then chunk across devices
            weight_reshaped = prepare_row_major_weight_bias(torch_weight, num_simulated_devices)
            bias_reshaped = prepare_row_major_weight_bias(torch_bias, num_simulated_devices)
            tt_weight = chunk_and_shard_tensor(
                weight_reshaped, num_simulated_devices, device, -1, dtype, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            tt_bias = chunk_and_shard_tensor(
                bias_reshaped, num_simulated_devices, device, -1, dtype, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        else:
            # For TILE_LAYOUT, use simple 2D shape [1, embedding_dim] and chunk
            weight_2d = torch_weight.unsqueeze(0)
            bias_2d = torch_bias.unsqueeze(0)
            tt_weight = chunk_and_shard_tensor(weight_2d, num_simulated_devices, device, -1, dtype)
            tt_bias = chunk_and_shard_tensor(bias_2d, num_simulated_devices, device, -1, dtype)

    tt_stats = []
    for tt_inp_chunk in tt_inp:
        tt_stats.append(
            ttnn.experimental.dit_layernorm_pre_allgather(
                tt_inp_chunk, compute_kernel_config=compute_kernel_config, dtype=stats_dtype
            )
        )

    # Simulate all-gather by concatenating stats from all devices
    tt_stats_gathered = ttnn.concat(tt_stats, -1)

    # Run post_allgather on each chunk with gathered stats
    tt_out = []
    for idx in range(len(tt_inp)):
        tt_out.append(
            ttnn.experimental.dit_layernorm_post_allgather(
                tt_inp[idx],
                tt_stats_gathered,
                epsilon=epsilon,
                weight=tt_weight[idx] if tt_weight is not None else None,
                bias=tt_bias[idx] if tt_bias is not None else None,
                compute_kernel_config=compute_kernel_config,
            )
        )

    # Concat the output on the sharded dimension
    tt_out = ttnn.concat(tt_out, -1)
    tt_out = ttnn.to_torch(tt_out)

    # Reference implementation using torch layer_norm
    out_torch = torch.nn.functional.layer_norm(
        torch_input, normalized_shape=(embedding_dim,), weight=None, bias=None, eps=epsilon
    )
    if use_affine:
        out_torch = out_torch * torch_weight
        out_torch = out_torch + torch_bias

    passing, output_str = comp_allclose(tt_out, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"torch vs tt distributed dit layernorm = {output_str}")
    assert passing


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "embedding_dim",
    [2048, 2432, 3072, 5120],
    ids=["dim2048", "dim2432", "dim3072", "dim5120"],
)
@pytest.mark.parametrize(
    "seq_len",
    [512, 2048, 4096, 9472],
    ids=["len512", "len2048", "len4096", "len9472"],
)
@pytest.mark.parametrize("use_affine", [False, True], ids=["no_affine", "with_affine"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
@pytest.mark.parametrize("num_simulated_devices", [1, 2, 4], ids=["tp1", "tp2", "tp4"])
def test_distributed_dit_layernorm_use_cases(
    device,
    num_simulated_devices,
    embedding_dim,
    seq_len,
    dtype,
    stats_dtype,
    use_affine,
    weight_bias_layout,
    reset_seeds,
):
    """
    Covers all DistributedLayerNorm instantiations seen in tt_dit models.
    Tests the DIT distributed layernorm op on single device, emulating TP devices.
    """
    # Skip if embedding_dim is not divisible by num_simulated_devices * TILE_SIZE
    if embedding_dim % (num_simulated_devices * TILE_SIZE) != 0:
        pytest.skip(
            f"embedding_dim ({embedding_dim}) must be divisible by num_simulated_devices * TILE_SIZE ({num_simulated_devices * TILE_SIZE})"
        )

    # Skip layout test when no affine (layout doesn't matter without weight/bias)
    if not use_affine and weight_bias_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Layout doesn't matter when affine is disabled")

    inp_shape = (1, 1, seq_len, embedding_dim)
    run_distributed_dit_layernorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, use_affine, weight_bias_layout
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [4096],
)
@pytest.mark.parametrize(
    "hidden_dim",
    [64, 128, 256, 320, 384, 512, 2432],
    ids=[
        "hidden_dim64",
        "hidden_dim128",
        "hidden_dim256",
        "hidden_dim320",
        "hidden_dim384",
        "hidden_dim512",
        "hidden_dim2432",
    ],
)
@pytest.mark.parametrize("use_affine", [True], ids=["with_affine"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
@pytest.mark.parametrize("num_simulated_devices", [1], ids=["num_simulated_devices1"])
def test_distributed_dit_layernorm_odd_hidden_dim(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    use_affine,
    weight_bias_layout,
    reset_seeds,
):
    """
    This test case ensures that the hidden dim tiles is a number not divisible by DST size,
    testing that CB sizes are correct in order to prevent hangs.
    """
    # Skip if hidden_dim is not divisible by num_simulated_devices * TILE_SIZE
    if hidden_dim % (num_simulated_devices * TILE_SIZE) != 0:
        pytest.skip(
            f"hidden_dim ({hidden_dim}) must be divisible by num_simulated_devices * TILE_SIZE ({num_simulated_devices * TILE_SIZE})"
        )

    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_dit_layernorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, use_affine, weight_bias_layout
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [9472, 18944],
    ids=["seqlen9472", "seqlen18944"],
)
@pytest.mark.parametrize("hidden_dim", [5120], ids=["hidden_dim5120"])
@pytest.mark.parametrize("use_affine", [True], ids=["with_affine"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
@pytest.mark.parametrize("num_simulated_devices", [2, 4], ids=["num_simulated_devices2", "num_simulated_devices4"])
def test_distributed_dit_layernorm_wan_configs(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    use_affine,
    weight_bias_layout,
    reset_seeds,
):
    """
    Test configurations matching WAN model requirements.
    """
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_dit_layernorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, use_affine, weight_bias_layout
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
def test_distributed_dit_layernorm_affine_layouts_and_dtypes(
    device,
    dtype,
    stats_dtype,
    weight_bias_layout,
    reset_seeds,
):
    """
    Test weight/bias with different layouts (tile vs row-major) and dtypes (bf16 vs fp32).
    Uses a single reasonable configuration: dim=2048, seqlen=512, num_devices=2.
    """
    num_simulated_devices = 2
    hidden_dim = 2048
    seqlen = 512
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_dit_layernorm(
        device,
        num_simulated_devices,
        inp_shape,
        dtype,
        stats_dtype,
        use_affine=True,
        weight_bias_layout=weight_bias_layout,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize("seqlen", [2048])
@pytest.mark.parametrize("hidden_dim", [8192], ids=["hidden_dim8192"])
@pytest.mark.parametrize("use_affine", [True], ids=["with_affine"])
@pytest.mark.parametrize(
    "weight_bias_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile_layout", "row_major_layout"],
)
@pytest.mark.parametrize("num_simulated_devices", [8], ids=["num_simulated_devices8"])
def test_distributed_dit_layernorm_program_cache(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    use_affine,
    weight_bias_layout,
    reset_seeds,
):
    """
    Test program cache behavior by running the same configuration multiple times.
    """
    inp_shape = (1, 1, seqlen, hidden_dim)
    dummy_tensors = []
    for i in range(2):
        run_distributed_dit_layernorm(
            device,
            num_simulated_devices,
            inp_shape,
            dtype,
            stats_dtype,
            use_affine,
            weight_bias_layout,
        )
        # Allocate dummy tensor to potentially invalidate cache
        dummy_tensors.append(
            ttnn.from_torch(
                torch.zeros(32, 32),
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
