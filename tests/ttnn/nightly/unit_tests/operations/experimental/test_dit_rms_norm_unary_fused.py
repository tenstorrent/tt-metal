# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc


def assert_quality(torch_output, tt_output, pcc_threshold=0.9995, rel_rmse_threshold=0.02):
    pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std if std > 0 else 0.0
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_dit_rms_norm_unary_fused_test(
    device,
    input_shape,
    hidden_dim,
    epsilon=1e-5,
    use_weight=True,
    activation=None,  # "silu", "gelu", ttnn.UnaryOpType, or None
    dtype=ttnn.bfloat16,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=False,
):
    """
    Test dit_rms_norm_unary_fused against reference: activation(rms_norm(x)).
    input_shape: tuple of leading dims, e.g. (1, 1, seq_len) — hidden_dim is appended.
    """
    torch.manual_seed(42)

    full_shape = input_shape + (hidden_dim,)

    torch_input = torch.randn(*full_shape, dtype=torch.bfloat16)
    torch_weight = torch.ones(hidden_dim, dtype=torch.bfloat16) if use_weight else None

    # Reference: rms_norm then activation
    with torch.no_grad():
        variance = torch_input.pow(2).mean(dim=-1, keepdim=True)
        torch_normed = torch_input * torch.rsqrt(variance + epsilon)
        if torch_weight is not None:
            torch_normed = torch_normed * torch_weight
        # Apply activation
        if activation == "silu" or activation == ttnn.UnaryOpType.SILU:
            torch_expected = torch.nn.functional.silu(torch_normed)
        elif activation == "gelu":
            torch_expected = torch.nn.functional.gelu(torch_normed)
        elif activation is None:
            torch_expected = torch_normed
        else:
            torch_expected = torch_normed

    # Convert to ttnn
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_weight = None
    if use_weight:
        torch_weight_2d = torch_weight.unsqueeze(0)
        tt_weight = ttnn.from_torch(torch_weight_2d, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

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
        compute_kernel_config=compute_config,
        activation=activation,
    )

    tt_output_torch = ttnn.to_torch(tt_output)

    return assert_quality(torch_expected, tt_output_torch)


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_weight", [True, False], ids=["with_weight", "no_weight"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
def test_dit_rms_norm_unary_fused_no_activation(device, use_weight, dtype):
    """Without activation: should match plain rms_norm."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        input_shape=(1, 1),
        hidden_dim=4096,
        use_weight=use_weight,
        activation=None,
        dtype=dtype,
    )
    assert check_result["pcc"] > 0.9995, f"PCC too low: {check_result['pcc']}"
    assert check_result["relative_rmse"] < 0.03, f"Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
def test_dit_rms_norm_unary_fused_silu_string(device, dtype):
    """Test with SiLU activation passed as a string."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        input_shape=(1, 1),
        hidden_dim=4096,
        activation="silu",
        dtype=dtype,
    )
    assert check_result["pcc"] > 0.9995, f"PCC too low: {check_result['pcc']}"
    assert check_result["relative_rmse"] < 0.03, f"Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
def test_dit_rms_norm_unary_fused_silu_unary_op_type(device, dtype):
    """Test with SiLU activation passed as ttnn.UnaryOpType."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        input_shape=(1, 1),
        hidden_dim=4096,
        activation=ttnn.UnaryOpType.SILU,
        dtype=dtype,
    )
    assert check_result["pcc"] > 0.9995, f"PCC too low: {check_result['pcc']}"
    assert check_result["relative_rmse"] < 0.03, f"Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize(
    "input_shape, hidden_dim, name",
    [
        ((1, 1), 256, "small"),
        ((1, 1), 512, "medium"),
        ((1, 38), 4096, "dit_norm_shape"),
    ],
    ids=["small", "medium", "dit_norm_shape"],
)
def test_dit_rms_norm_unary_fused_basic_shapes(device, input_shape, hidden_dim, name):
    """Basic shapes test with SiLU activation."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        input_shape=input_shape,
        hidden_dim=hidden_dim,
        activation="silu",
    )
    assert check_result["pcc"] > 0.9995, f"[{name}] PCC too low: {check_result['pcc']}"
    assert check_result["relative_rmse"] < 0.03, f"[{name}] Relative RMSE too high: {check_result['relative_rmse']}"


# ---------------------------------------------------------------------------
# Wan2.2 shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_len, hidden_dim, config_name",
    [
        (9472, 5120, "wan2.2_14b-720p-full"),
        (2368, 5120, "wan2.2_14b-720p-single"),
    ],
    ids=["wan2.2_14b-720p-full", "wan2.2_14b-720p-single"],
)
def test_dit_rms_norm_unary_fused_wan2_shapes(device, seq_len, hidden_dim, config_name):
    """Test with actual Wan2.2 transformer shapes."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        input_shape=(1, 1, seq_len),
        hidden_dim=hidden_dim,
        activation="silu",
        dtype=ttnn.bfloat16,
    )
    assert check_result["pcc"] > 0.9995, f"[{config_name}] PCC too low: {check_result['pcc']}"
    assert (
        check_result["relative_rmse"] < 0.04
    ), f"[{config_name}] Relative RMSE too high: {check_result['relative_rmse']}"


# ---------------------------------------------------------------------------
# Activation variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activation",
    ["silu", "gelu", None],
    ids=["silu", "gelu", "no_activation"],
)
def test_dit_rms_norm_unary_fused_activations(device, activation):
    """Test different activation functions."""
    check_result = run_dit_rms_norm_unary_fused_test(
        device=device,
        input_shape=(1, 1),
        hidden_dim=1024,
        activation=activation,
        use_weight=True,
    )
    assert check_result["pcc"] > 0.9995, f"[{activation}] PCC too low: {check_result['pcc']}"
    assert (
        check_result["relative_rmse"] < 0.02
    ), f"[{activation}] Relative RMSE too high: {check_result['relative_rmse']}"


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_dim",
    [
        (1, 32, 128),
        (2, 64, 256),
    ],
    ids=["batch1_seq32_dim128", "batch2_seq64_dim256"],
)
def test_dit_rms_norm_unary_fused_vs_ttnn_silu_rms_norm(device, batch_size, seq_len, hidden_dim):
    """Compare dit_rms_norm_unary_fused with ttnn.silu(ttnn.rms_norm) for consistency."""
    import torch
    import numpy as np
    import ttnn

    # Generate random input and weight tensors
    torch.manual_seed(123)
    input_torch = torch.rand((batch_size, seq_len, hidden_dim), dtype=torch.bfloat16)
    weight_torch = torch.rand((hidden_dim,), dtype=torch.bfloat16)

    # to device/tile layout
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT)
    weight_tt = ttnn.from_torch(weight_torch, device=device, layout=ttnn.TILE_LAYOUT)

    # Reference: ttnn.silu(ttnn.rms_norm)
    rms_norm_out = ttnn.rms_norm(input_tt, weight=weight_tt)
    silu_rms_norm_out = ttnn.silu(rms_norm_out)
    silu_rms_norm_out = ttnn.from_device(silu_rms_norm_out)
    silu_rms_norm_out = ttnn.to_torch(silu_rms_norm_out)

    math_fidelity = ttnn.MathFidelity.HiFi4
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
    )

    calculated = ttnn.experimental.dit_rms_norm_unary_fused(
        input_tt,
        epsilon=1e-5,
        weight=weight_tt,
        compute_kernel_config=compute_config,
        activation="silu",
    )

    torch_calculated = ttnn.to_torch(calculated)

    # Ensure shape match
    assert torch_calculated.shape == silu_rms_norm_out.shape

    # PCC and RMSE
    torch_calculated_fp32 = torch_calculated.to(torch.float32)
    golden_fp32 = silu_rms_norm_out.to(torch.float32)
    pcc = np.corrcoef(golden_fp32.flatten().numpy(), torch_calculated_fp32.flatten().numpy())[0, 1]
    rmse = torch.sqrt(torch.mean((golden_fp32 - torch_calculated_fp32) ** 2)).item()
    mean_abs = torch.mean(torch.abs(golden_fp32)).item()
    rel_rmse = rmse / (mean_abs + 1e-7)

    assert (
        pcc > 0.9995
    ), f"PCC too low: {pcc} (golden: {golden_fp32.flatten().numpy()}, calculated: {torch_calculated_fp32.flatten().numpy()})"
    assert (
        rel_rmse < 0.02
    ), f"Relative RMSE too high: {rel_rmse} (golden: {golden_fp32.flatten().numpy()}, calculated: {torch_calculated_fp32.flatten().numpy()})"
