# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose


def get_rot_transformation_mat():
    # ROPE op uses a single tile
    rot_emb_matrix = torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE)
    rot_emb_matrix[..., torch.arange(0, ttnn.TILE_SIZE, 2), torch.arange(1, ttnn.TILE_SIZE, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, ttnn.TILE_SIZE, 2), torch.arange(0, ttnn.TILE_SIZE, 2)] = -1
    return rot_emb_matrix


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def chunk_and_shard_tensor(tensor, num_simulated_devices, device, dim, dtype):
    chunked = torch.chunk(tensor, num_simulated_devices, dim)
    result = [
        ttnn.from_torch(
            chunk, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for chunk in chunked
    ]
    return result


def check_hidden_dim_divisible_by_num_heads(hidden_dim, num_heads):
    if hidden_dim // ttnn.TILE_SIZE % num_heads != 0:
        pytest.skip("hidden_dim must be divisible by 32 * num_heads")


def run_distributed_fused_rmsnorm(
    device,
    num_simulated_devices,
    inp_shape,
    dtype,
    stats_dtype,
    num_heads_per_device=1,
    use_weight=True,
    use_rope=False,
    rope_dtype=None,
):
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    torch.manual_seed(1234)

    num_heads = num_heads_per_device * num_simulated_devices

    torch_input = torch.randn(inp_shape) * 4 - 1

    if use_weight:
        torch_weight = torch.rand(inp_shape[-1:]).unsqueeze(0)
        assert torch_weight.shape == (1, inp_shape[-1])

    tt_rope_cos = None
    tt_rope_sin = None
    tt_transformation_mat = None
    if use_rope:
        # Use rope_dtype if provided, otherwise default to dtype
        effective_rope_dtype = rope_dtype if rope_dtype is not None else dtype

        head_dim = inp_shape[-1] // num_heads
        rope_cos = torch.randn(1, 1, inp_shape[2], head_dim // 2)
        rope_cos = torch.stack([rope_cos, rope_cos], dim=-1).flatten(-2)
        rope_sin = torch.randn(1, 1, inp_shape[2], head_dim // 2)
        rope_sin = torch.stack([rope_sin, rope_sin], dim=-1).flatten(-2)

        transformation_mat = get_rot_transformation_mat()

        tt_rope_cos = ttnn.from_torch(
            rope_cos,
            dtype=effective_rope_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_rope_sin = ttnn.from_torch(
            rope_sin,
            dtype=effective_rope_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tt_transformation_mat = ttnn.from_torch(
            transformation_mat,
            dtype=dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    epsilon = 1e-5

    tt_inp = chunk_and_shard_tensor(torch_input, num_simulated_devices, device, -1, dtype)

    tt_weight = None
    if use_weight:
        tt_weight = chunk_and_shard_tensor(torch_weight, num_simulated_devices, device, -1, dtype)

    tt_stats = []
    for tt_inp_chunk in tt_inp:
        tt_stats.append(
            ttnn.experimental.wan_fused_rmsnorm_pre_allgather(
                tt_inp_chunk, compute_kernel_config=compute_kernel_config, dtype=stats_dtype
            )
        )

    # Pretend we're running on multi-device by concatenating the stats
    tt_stats_gathered = ttnn.concat(tt_stats, -1)

    tt_out = []
    for idx in range(len(tt_inp)):
        tt_out.append(
            ttnn.experimental.wan_fused_rmsnorm_post_allgather(
                tt_inp[idx],
                tt_stats_gathered,
                epsilon=epsilon,
                num_heads_per_device=num_heads_per_device,
                weight=tt_weight[idx] if tt_weight is not None else None,
                transformation_mat=tt_transformation_mat,
                rope_cos=tt_rope_cos,
                rope_sin=tt_rope_sin,
                compute_kernel_config=compute_kernel_config,
            )
        )

    # Concat the output on the "sharded" dimension
    if num_heads > 1:
        tt_out = ttnn.concat(tt_out, -3)
    else:
        tt_out = ttnn.concat(tt_out, -1)

    tt_out = ttnn.to_torch(
        tt_out,
    )

    # reference impl
    out_torch = torch.nn.functional.rms_norm(torch_input, normalized_shape=(inp_shape[-1],), eps=epsilon)
    if use_weight:
        out_torch = out_torch * torch_weight

    # create heads fusion
    out_torch = out_torch.reshape(inp_shape[0], inp_shape[2], num_heads, -1).permute(0, 2, 1, 3)

    if use_rope:
        out_torch = apply_rotary_emb(out_torch, rope_cos, rope_sin)

    passing, output_str = comp_allclose(tt_out, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"torch vs tt distributed fused rmsnorm = {output_str}")
    assert passing


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [2048, 2080],
    ids=["seqlen2048", "seqlen2080"],
)
@pytest.mark.parametrize("hidden_dim", [2048, 5120], ids=["hidden_dim2048", "hidden_dim5120"])
@pytest.mark.parametrize("num_heads_per_device", [1, 2], ids=["num_heads1", "num_heads2"])
@pytest.mark.parametrize("use_weight", [True, False], ids=["has_weight", "no_weight"])
@pytest.mark.parametrize("use_rope", [True, False], ids=["has_rope", "no_rope"])
@pytest.mark.parametrize("num_simulated_devices", [8], ids=["num_simulated_devices8"])
def test_distributed_fused_rmsnorm_sweep_fusions(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    num_heads_per_device,
    use_weight,
    use_rope,
    reset_seeds,
):
    num_heads = num_heads_per_device * num_simulated_devices
    check_hidden_dim_divisible_by_num_heads(hidden_dim, num_heads)
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_fused_rmsnorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, num_heads_per_device, use_weight, use_rope
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [128, 256, 8192],
    ids=["seqlen128", "seqlen256", "seqlen8192"],
)
@pytest.mark.parametrize("hidden_dim", [1024, 2048], ids=["hidden_dim1024", "hidden_dim2048"])
@pytest.mark.parametrize("num_heads_per_device", [1, 2], ids=["num_heads1", "num_heads2"])
@pytest.mark.parametrize("use_weight", [True], ids=["has_weight"])
@pytest.mark.parametrize("use_rope", [True], ids=["has_rope"])
@pytest.mark.parametrize("num_simulated_devices", [8], ids=["num_simulated_devices8"])
def test_distributed_fused_rmsnorm_sweep_shapes(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    num_heads_per_device,
    use_weight,
    use_rope,
    reset_seeds,
):
    num_heads = num_heads_per_device * num_simulated_devices
    check_hidden_dim_divisible_by_num_heads(hidden_dim, num_heads)
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_fused_rmsnorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, num_heads_per_device, use_weight, use_rope
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [32, 4096],
    ids=["seqlen32", "seqlen4096"],
)
@pytest.mark.parametrize("hidden_dim", [320, 2432], ids=["hidden_dim320", "hidden_dim2432"])
@pytest.mark.parametrize("num_heads_per_device", [1], ids=["num_heads1"])
@pytest.mark.parametrize("use_weight", [True], ids=["has_weight"])
@pytest.mark.parametrize("use_rope", [False], ids=["no_rope"])
@pytest.mark.parametrize("num_simulated_devices", [2], ids=["num_simulated_devices2"])
def test_distributed_fused_rmsnorm_odd_hidden_dim(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    num_heads_per_device,
    use_weight,
    use_rope,
    reset_seeds,
):
    """
    This test case ensures that the hidden dim tiles is a number not divisible by DST size,
    testing that CB sizes are correct in order to prevent hangs.
    """
    num_heads = num_heads_per_device * num_simulated_devices
    check_hidden_dim_divisible_by_num_heads(hidden_dim, num_heads)
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_fused_rmsnorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, num_heads_per_device, use_weight, use_rope
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [9472, 18944],
    ids=["seqlen9472", "seqlen18944"],
)
@pytest.mark.parametrize("hidden_dim", [5120], ids=["hidden_dim5120"])
@pytest.mark.parametrize("use_weight", [True], ids=["has_weight"])
@pytest.mark.parametrize("use_rope", [True], ids=["has_rope"])
@pytest.mark.parametrize("num_simulated_devices", [2, 4], ids=["num_simulated_devices2", "num_simulated_devices4"])
def test_distributed_fused_rmsnorm_wan_configs(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    use_weight,
    use_rope,
    reset_seeds,
):
    WAN_NUM_HEADS = 40
    num_heads_per_device = WAN_NUM_HEADS // num_simulated_devices
    num_heads = num_heads_per_device * num_simulated_devices
    check_hidden_dim_divisible_by_num_heads(hidden_dim, num_heads)
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_fused_rmsnorm(
        device, num_simulated_devices, inp_shape, dtype, stats_dtype, num_heads_per_device, use_weight, use_rope
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize("seqlen", [2048])
@pytest.mark.parametrize("hidden_dim", [8192], ids=["hidden_dim8192"])
@pytest.mark.parametrize("num_heads_per_device", [1], ids=["num_heads1"])
@pytest.mark.parametrize("use_weight", [True], ids=["has_weight"])
@pytest.mark.parametrize("use_rope", [True], ids=["has_rope"])
@pytest.mark.parametrize("num_simulated_devices", [8], ids=["num_simulated_devices8"])
def test_distributed_fused_rmsnorm_program_cache(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    num_heads_per_device,
    use_weight,
    use_rope,
    reset_seeds,
):
    num_heads = num_heads_per_device * num_simulated_devices
    check_hidden_dim_divisible_by_num_heads(hidden_dim, num_heads)
    inp_shape = (1, 1, seqlen, hidden_dim)
    dummy_tensors = []
    for i in range(2):
        run_distributed_fused_rmsnorm(
            device,
            num_simulated_devices,
            inp_shape,
            dtype,
            stats_dtype,
            num_heads_per_device,
            use_weight,
            use_rope,
        )
        dummy_tensors.append(
            ttnn.from_torch(
                torch.zeros(32, 32),
                dtype=dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.float32], ids=["FLOAT32_stats"])
@pytest.mark.parametrize("rope_dtype", [ttnn.float32], ids=["FLOAT32_rope"])
@pytest.mark.parametrize("seqlen", [2048], ids=["seqlen2048"])
@pytest.mark.parametrize("hidden_dim", [2048], ids=["hidden_dim2048"])
@pytest.mark.parametrize("num_heads_per_device", [1], ids=["num_heads1"])
@pytest.mark.parametrize("use_weight", [True], ids=["has_weight"])
@pytest.mark.parametrize("use_rope", [True], ids=["has_rope"])
@pytest.mark.parametrize("num_simulated_devices", [8], ids=["num_simulated_devices8"])
def test_distributed_fused_rmsnorm_fp32_rope(
    device,
    num_simulated_devices,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    rope_dtype,
    num_heads_per_device,
    use_weight,
    use_rope,
    reset_seeds,
):
    """Test that rope_cos and rope_sin can be FLOAT32 while input is BFLOAT16."""
    num_heads = num_heads_per_device * num_simulated_devices
    check_hidden_dim_divisible_by_num_heads(hidden_dim, num_heads)
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_fused_rmsnorm(
        device,
        num_simulated_devices,
        inp_shape,
        dtype,
        stats_dtype,
        num_heads_per_device,
        use_weight,
        use_rope,
        rope_dtype,
    )
