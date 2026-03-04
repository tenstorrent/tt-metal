# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from scripts.debug.save_tensor_csv import save_tensor_csv


def assert_quality(torch_output, tt_output, pcc_threshold=0.9995, rel_rmse_threshold=0.02):
    pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std if std > 0 else 0.0
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def rms_norm_golden(input_tensor, epsilon, weight, bias, activation):
    """
    Compute golden RMS output for an input tensor and weight tensor.
    Args:
        input_tensor: The input tensor to run the rms norm on.
        epsilon: The epsilon to use for the rms norm.
        weight: The weight tensor to use for the rms norm.
        bias: The bias tensor to use for the rms norm.
        activation: The activation to apply to the rms norm output.
    Returns:
        The output tensor as a torch tensor.
    """

    with torch.no_grad():
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        output = input_tensor * torch.rsqrt(variance + epsilon)

        if weight is not None:
            output = output * weight

        if bias is not None:
            output = output + bias

        if activation == "silu" or activation == ttnn.UnaryOpType.SILU:
            output = torch.nn.functional.silu(output)
        elif activation == "gelu" or activation == ttnn.UnaryOpType.GELU:
            output = torch.nn.functional.gelu(output)
        elif activation is not None:
            raise ValueError(
                f"Unsupported activation: {activation!r}. "
                "Supported: 'silu', 'gelu', ttnn.UnaryOpType.SILU, ttnn.UnaryOpType.GELU, or None."
            )

    return output


def run_dit_rms_norm_unary_fused_test(
    device,
    shape,
    epsilon=1e-5,
    use_weight=False,
    use_bias=False,
    activation=None,  # "silu", "gelu", ttnn.UnaryOpType, or None
    dtype=ttnn.bfloat16,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=False,
    # --- sharded path: shard_params = (num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt) or None ---
    shard_params=None,
    input_layout=ttnn.TILE_LAYOUT,
):
    """
    Test helper for dit_rms_norm_unary_fused.

    When sharded=False (default): uses the interleaved path (layernorm.cpp kernel).
    When sharded=True: wraps the input in a BLOCK_SHARDED memory config and passes a
    LayerNormShardedMultiCoreProgramConfig (layernorm_sharded.cpp kernel).
    shard_params must be (num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt) when sharded=True.

    Weight/bias tensors use shape (1, w) regardless of path.
    """
    torch.manual_seed(42)

    w = shape[-1]
    h = shape[-2]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.ones(shape[-1], dtype=torch.bfloat16) if use_weight else None
    torch_bias = torch.rand(shape[-1], dtype=torch.bfloat16) if use_bias else None

    memory_config = ttnn.DRAM_MEMORY_CONFIG
    program_config = None

    if shard_params is not None:
        assert (
            shard_params is not None and len(shard_params) == 5
        ), "shard_params must be (num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt) when sharded=True"
        num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = shard_params
        shard_height = h // num_cores_h
        shard_width = w // num_cores_w
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))}),
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=subblock_wt,
            use_welford=False,
            inplace=False,
        )
    torch_expected = rms_norm_golden(torch_input, epsilon, torch_weight, torch_bias, activation)

    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=input_layout, device=device, memory_config=memory_config
    )
    tt_weight = (
        ttnn.from_torch(torch_weight.unsqueeze(0), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if use_weight
        else None
    )
    tt_bias = (
        ttnn.from_torch(torch_bias.unsqueeze(0), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if use_bias
        else None
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )

    tt_output = ttnn.experimental.dit_rms_norm_unary_fused(
        tt_input,
        epsilon=epsilon,
        weight=tt_weight,
        bias=tt_bias,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_config,
        activation=activation,
    )

    assert (
        tt_output.layout == tt_input.layout
    ), f"tt_output.layout: {tt_output.layout} != tt_input.layout: {tt_input.layout}"

    tt_output_torch = ttnn.to_torch(tt_output)

    # save_tensor_csv(tt_output_torch, "output/tt_output_torch.csv")
    # save_tensor_csv(torch_expected, "output/torch_expected.csv")

    # print(f"tt_output_torch: \n{tt_output_torch}")
    # print(f"torch_expected: \n{torch_expected}")

    return assert_quality(torch_expected, tt_output_torch)


@pytest.mark.parametrize("activation", [None, "silu", ttnn.UnaryOpType.SILU], ids=["none", "silu", "UnaryOpType.SILU"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
def test_dit_rms_norm_unary_fused_silu_unary_op_type(device, dtype, activation):
    """Test with SiLU activation passed as ttnn.UnaryOpType."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=(1, 4096),
        activation=activation,
        dtype=dtype,
    )
    assert check_result["pcc"] > 0.9995, f"PCC too low: {check_result['pcc']}"
    assert check_result["relative_rmse"] < 0.03, f"Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize(
    "shape, name",
    [
        ((1, 256), "small"),
        ((1, 512), "medium"),
        ((38, 4096), "dit_norm_shape"),
    ],
    ids=["small", "medium", "dit_norm_shape"],
)
def test_dit_rms_norm_unary_fused_basic_shapes(device, shape, name):
    """Basic shapes test with SiLU activation."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=shape,
        activation="silu",
    )
    assert check_result["pcc"] > 0.9995, f"[{name}] PCC too low: {check_result['pcc']}"
    assert check_result["relative_rmse"] < 0.03, f"[{name}] Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize(
    "shape, config_name",
    [
        ((9472, 5120), "wan2.2_14b-720p-full"),
        ((2368, 5120), "wan2.2_14b-720p-single"),
        ((1, 384, 1, 90, 160), "wan.decoder.up_blocks.3.resnets.0.norm1"),
    ],
    ids=["wan2.2_14b-720p-full", "wan2.2_14b-720p-single", "wan.decoder.up_blocks.3.resnets.0.norm1"],
)
def test_dit_rms_norm_unary_fused_wan2_shapes(device, shape, config_name):
    """Test with actual Wan2.2 transformer shapes."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=shape,
        activation="silu",
        dtype=ttnn.bfloat16,
    )
    # assert check_result["pcc"] > 0.9995, f"[{config_name}] PCC too low: {check_result['pcc']}"
    # assert (
    #     check_result["relative_rmse"] < 0.04
    # ), f"[{config_name}] Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize(
    "use_weight, use_bias",
    [(False, True), (True, True)],
    ids=["bias_only", "weight_and_bias"],
)
@pytest.mark.parametrize(
    "activation",
    ["silu", None],
    ids=["silu", "no_activation"],
)
def test_dit_rms_norm_unary_fused_weight_bias(device, use_weight, use_bias, activation):
    """Weight/bias combinations for the interleaved (non-sharded) path."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=(1, 1024),
        use_weight=use_weight,
        use_bias=use_bias,
        activation=activation,
    )
    assert (
        check_result["pcc"] > 0.9995
    ), f"[w={use_weight},b={use_bias},{activation}] PCC too low: {check_result['pcc']}"
    assert (
        check_result["relative_rmse"] < 0.03
    ), f"[w={use_weight},b={use_bias},{activation}] Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize(
    "shape, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt",
    [((256, 320), 2, 5, 4, 2, 1)],
)
@pytest.mark.parametrize(
    "use_weight, use_bias",
    [(False, True), (True, True)],
    ids=["bias_only", "weight_and_bias"],
)
def test_dit_rms_norm_unary_fused_sharded_weight_bias(
    device, shape, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, use_weight, use_bias
):
    """
    Sharded path weight/bias combinations without activation.
    Activation must be None here: sharded + activation requires do_gamma=0, do_beta=0.
    """
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=shape,
        shard_params=(num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt),
        use_weight=use_weight,
        use_bias=use_bias,
        activation=None,
    )

    assert check_result["pcc"] > 0.9995, f"[sharded/w={use_weight},b={use_bias}] PCC too low: {check_result['pcc']}"

    assert (
        check_result["relative_rmse"] < 0.03
    ), f"[sharded/w={use_weight},b={use_bias}] Relative RMSE too high: {check_result['relative_rmse']}"


# ---------------------------------------------------------------------------
# ROW_MAJOR input tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, name",
    [
        # one_block: exactly 1 DPRINT-friendly block (Wt=blk=8, NCHt=1, no partial blocks, no padding).
        # Use this first when debugging — simplest possible case.
        ((32, 256), "one_block"),
        ((32, 64), "tiny"),
        ((64, 128), "small"),
        ((64, 512), "medium"),
        ((128, 1024), "large"),
        ((512, 4096), "larger"),
    ],
    ids=["one_block", "tiny", "small", "medium", "large", "larger"],
)
def test_dit_rms_norm_unary_fused_row_major_basic_shapes(device, shape, name):
    """ROW_MAJOR input, small and medium shapes, no gamma/bias."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=shape,
        activation=None,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    assert check_result["pcc"] > 0.9995, f"[{name}] PCC too low: {check_result['pcc']}"
    assert check_result["relative_rmse"] < 0.03, f"[{name}] Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize(
    "shape, config_name",
    [
        ((1584, 5120), "wan2.2_14b-480p-single"),
        ((6336, 5120), "wan2.2_14b-480p-full"),
        ((1, 2368, 5120), "wan2.2_14b-720p-single"),
        ((1, 9472, 5120), "wan2.2_14b-720p-full"),
        ((1, 384, 1, 90, 160), "wan.decoder.up_blocks.3.resnets.0.norm1"),
    ],
    ids=[
        "wan2.2_14b-480p-single",
        "wan2.2_14b-480p-full",
        "wan2.2_14b-720p-single",
        "wan2.2_14b-720p-full",
        "wan.decoder.up_blocks.3.resnets.0.norm1",
    ],
)
def test_dit_rms_norm_unary_fused_row_major_wan2_shapes(device, shape, config_name):
    """ROW_MAJOR input with Wan2.2 shapes (triggers large-tensor kernel path)."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=shape,
        activation="silu",
        dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    print(f"[{config_name}] PCC: {check_result['pcc']}, Relative RMSE: {check_result['relative_rmse']}")
    assert check_result["pcc"] > 0.9995, f"[{config_name}] PCC too low: {check_result['pcc']}"
    assert (
        check_result["relative_rmse"] < 0.04
    ), f"[{config_name}] Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize(
    "use_weight, use_bias",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["no_weight_no_bias", "weight_only", "bias_only", "weight_and_bias"],
)
@pytest.mark.parametrize(
    "activation",
    ["silu", None],
    ids=["silu", "no_activation"],
)
@pytest.mark.parametrize(
    "shape",
    [
        # (64, 512),
        (512, 4096)
    ],
    ids=[
        # "small",
        "large"
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bfloat16", "float32"])
def test_dit_rms_norm_unary_fused_row_major_with_tile_weights(device, use_weight, use_bias, activation, shape, dtype):
    """ROW_MAJOR input with TILE-layout gamma and/or beta."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=shape,
        use_weight=use_weight,
        use_bias=use_bias,
        activation=activation,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
    )
    assert (
        check_result["pcc"] > 0.9995
    ), f"[rm/w={use_weight},b={use_bias},{activation}] PCC too low: {check_result['pcc']}"
    assert (
        check_result["relative_rmse"] < 0.03
    ), f"[rm/w={use_weight},b={use_bias},{activation}] Relative RMSE too high: {check_result['relative_rmse']}"


def test_dit_rms_one_block_bug(device):
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        shape=(32, 256),
        use_weight=True,
        use_bias=False,
        activation=None,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    assert check_result["pcc"] > 0.9995, f"[rm/w=True,b=False,None] PCC too low: {check_result['pcc']}"
    assert (
        check_result["relative_rmse"] < 0.03
    ), f"[rm/w=True,b=False,None] Relative RMSE too high: {check_result['relative_rmse']}"


def test_dit_rms_norm_unary_fused_row_major_sharded_fatal(device):
    """ROW_MAJOR input + sharded memory config must raise a TT_FATAL / RuntimeError."""
    import re

    shape = (256, 320)
    num_cores_h, num_cores_w = 2, 5
    shard_height = shape[0] // num_cores_h
    shard_width = shape[1] // num_cores_w
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))}),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    with pytest.raises((RuntimeError, Exception), match=re.compile(r"sharded|ROW_MAJOR", re.IGNORECASE)):
        ttnn.experimental.dit_rms_norm_unary_fused(tt_input, epsilon=1e-5)
