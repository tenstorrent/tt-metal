# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.recurrent_deltanet import TTNNRecurrentDeltaNet
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_recurrent_deltanet(device):
    """Test recurrent deltanet with TTNN acceleration."""
    try:
        from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_deltanet import (
            gated_deltanet_forward,
        )
    except ImportError:
        pytest.skip("PyTorch reference implementation not available")

    batch_size = 1
    seq_len = 32
    num_heads = 4
    num_v_heads = 4
    head_k_dim = 64
    head_v_dim = 128
    hidden_size = num_heads * head_k_dim
    conv_kernel_size = 4
    use_gate = True

    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Create random weights matching expected shapes
    q_proj_weight = torch.randn(hidden_size, num_heads * head_k_dim, dtype=torch.float32) * 0.02
    k_proj_weight = torch.randn(hidden_size, num_heads * head_k_dim, dtype=torch.float32) * 0.02
    v_proj_weight = torch.randn(hidden_size, num_v_heads * head_v_dim, dtype=torch.float32) * 0.02
    a_proj_weight = torch.randn(hidden_size, num_v_heads, dtype=torch.float32) * 0.02
    b_proj_weight = torch.randn(hidden_size, num_v_heads, dtype=torch.float32) * 0.02
    o_proj_weight = torch.randn(num_v_heads * head_v_dim, hidden_size, dtype=torch.float32) * 0.02
    g_proj_weight = torch.randn(hidden_size, num_v_heads * head_v_dim, dtype=torch.float32) * 0.02

    q_conv_weight = torch.randn(num_heads * head_k_dim, 1, conv_kernel_size, dtype=torch.float32) * 0.02
    k_conv_weight = torch.randn(num_heads * head_k_dim, 1, conv_kernel_size, dtype=torch.float32) * 0.02
    v_conv_weight = torch.randn(num_v_heads * head_v_dim, 1, conv_kernel_size, dtype=torch.float32) * 0.02

    q_conv_bias = torch.randn(num_heads * head_k_dim, dtype=torch.float32) * 0.01
    k_conv_bias = torch.randn(num_heads * head_k_dim, dtype=torch.float32) * 0.01
    v_conv_bias = torch.randn(num_v_heads * head_v_dim, dtype=torch.float32) * 0.01

    A_log = torch.randn(num_v_heads, dtype=torch.float32) * 0.1 - 1.0
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32) * 0.1
    o_norm_weight = torch.ones(head_v_dim, dtype=torch.float32)

    # Run PyTorch reference implementation
    # PyTorch F.linear expects weights in [out_features, in_features] format
    # but we create them in [in_features, out_features] for TTNN, so transpose for PyTorch
    print(f"[DEBUG TEST] PyTorch input shape: {hidden_states.shape}")
    print(f"[DEBUG TEST] PyTorch input sample (first 5): {hidden_states.flatten()[:5].tolist()}")
    print(
        f"[DEBUG TEST] PyTorch input stats - min: {hidden_states.min().item():.6f}, max: {hidden_states.max().item():.6f}, mean: {hidden_states.mean().item():.6f}"
    )

    output_torch, cache_torch = gated_deltanet_forward(
        hidden_states=hidden_states,
        q_proj_weight=q_proj_weight.transpose(0, 1),
        k_proj_weight=k_proj_weight.transpose(0, 1),
        v_proj_weight=v_proj_weight.transpose(0, 1),
        a_proj_weight=a_proj_weight.transpose(0, 1),
        b_proj_weight=b_proj_weight.transpose(0, 1),
        o_proj_weight=o_proj_weight.transpose(0, 1),
        q_conv_weight=q_conv_weight,
        k_conv_weight=k_conv_weight,
        v_conv_weight=v_conv_weight,
        q_conv_bias=q_conv_bias,
        k_conv_bias=k_conv_bias,
        v_conv_bias=v_conv_bias,
        A_log=A_log,
        dt_bias=dt_bias,
        o_norm_weight=o_norm_weight,
        g_proj_weight=g_proj_weight.transpose(0, 1),
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        use_gate=use_gate,
        allow_neg_eigval=False,
        norm_eps=1e-5,
        mode="recurrent",
        chunk_size=64,
        recurrent_state=None,
        output_final_state=True,
    )

    final_state_torch = cache_torch["recurrent_state"] if cache_torch else None

    print(f"[DEBUG TEST] PyTorch output shape: {output_torch.shape}")
    print(f"[DEBUG TEST] PyTorch output sample (first 5): {output_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG TEST] PyTorch output stats - min: {output_torch.min().item():.6f}, max: {output_torch.max().item():.6f}, mean: {output_torch.mean().item():.6f}"
    )
    if final_state_torch is not None:
        print(f"[DEBUG TEST] PyTorch final_state shape: {final_state_torch.shape}")
        print(f"[DEBUG TEST] PyTorch final_state sample (first 5): {final_state_torch.flatten()[:5].tolist()}")
        print(
            f"[DEBUG TEST] PyTorch final_state stats - min: {final_state_torch.min().item():.6f}, max: {final_state_torch.max().item():.6f}, mean: {final_state_torch.mean().item():.6f}"
        )

    # Convert input to bfloat16 for TTNN
    hidden_states_ttnn = hidden_states.to(torch.bfloat16)

    # Create TTNN model
    ttnn_model = TTNNRecurrentDeltaNet(
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        use_gate=use_gate,
        allow_neg_eigval=False,
        norm_eps=1e-5,
        mode="recurrent",
        chunk_size=64,
    )
    set_device(ttnn_model, device)

    # Manually preprocess and set weights
    # TTNN linear expects weights in [in_features, out_features] format
    # Our weights are already in this format (PyTorch convention), so no transpose needed
    ttnn_model.q_proj_weight = ttnn.from_torch(
        q_proj_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.k_proj_weight = ttnn.from_torch(
        k_proj_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.v_proj_weight = ttnn.from_torch(
        v_proj_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.a_proj_weight = ttnn.from_torch(
        a_proj_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.b_proj_weight = ttnn.from_torch(
        b_proj_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.o_proj_weight = ttnn.from_torch(
        o_proj_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.g_proj_weight = ttnn.from_torch(
        g_proj_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Conv weights
    ttnn_model.q_conv_weight = ttnn.from_torch(
        q_conv_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn_model.k_conv_weight = ttnn.from_torch(
        k_conv_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn_model.v_conv_weight = ttnn.from_torch(
        v_conv_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Conv biases
    ttnn_model.q_conv_bias = ttnn.from_torch(
        q_conv_bias.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.k_conv_bias = ttnn.from_torch(
        k_conv_bias.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.v_conv_bias = ttnn.from_torch(
        v_conv_bias.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Other parameters
    ttnn_model.A_log = ttnn.from_torch(
        A_log.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.dt_bias = ttnn.from_torch(
        dt_bias.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model.o_norm_weight = ttnn.from_torch(
        o_norm_weight.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Wrap input as TorchTTNNTensor
    hidden_states_ttnn_tensor = TorchTTNNTensor(hidden_states_ttnn)

    # Run forward pass
    output_ttnn, final_state_ttnn = ttnn_model(hidden_states=hidden_states_ttnn_tensor)

    # Convert TTNN outputs to torch for comparison
    output_ttnn_torch = output_ttnn.to_torch if hasattr(output_ttnn, "to_torch") else ttnn.to_torch(output_ttnn)
    final_state_ttnn_torch = (
        final_state_ttnn.to_torch if hasattr(final_state_ttnn, "to_torch") else ttnn.to_torch(final_state_ttnn)
    )

    print(f"[DEBUG TEST] TTNN output shape: {output_ttnn.shape}")
    print(f"[DEBUG TEST] TTNN output sample (first 5): {output_ttnn_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG TEST] TTNN output stats - min: {output_ttnn_torch.min().item():.6f}, max: {output_ttnn_torch.max().item():.6f}, mean: {output_ttnn_torch.mean().item():.6f}"
    )
    if final_state_ttnn is not None:
        print(f"[DEBUG TEST] TTNN final_state shape: {final_state_ttnn.shape}")
        print(f"[DEBUG TEST] TTNN final_state sample (first 5): {final_state_ttnn_torch.flatten()[:5].tolist()}")
        print(
            f"[DEBUG TEST] TTNN final_state stats - min: {final_state_ttnn_torch.min().item():.6f}, max: {final_state_ttnn_torch.max().item():.6f}, mean: {final_state_ttnn_torch.mean().item():.6f}"
        )

    # Compute differences
    diff_output = torch.abs(output_torch - output_ttnn_torch.to(torch.float32))
    print(
        f"[DEBUG TEST] Output diff stats - min: {diff_output.min().item():.6f}, max: {diff_output.max().item():.6f}, mean: {diff_output.mean().item():.6f}"
    )
    if final_state_torch is not None:
        diff_state = torch.abs(final_state_torch - final_state_ttnn_torch.to(torch.float32))
        print(
            f"[DEBUG TEST] State diff stats - min: {diff_state.min().item():.6f}, max: {diff_state.max().item():.6f}, mean: {diff_state.mean().item():.6f}"
        )

    # Verify output shapes
    assert output_ttnn.shape == (
        batch_size,
        seq_len,
        hidden_size,
    ), f"Expected output shape {(batch_size, seq_len, hidden_size)}, got {output_ttnn.shape}"
    if final_state_torch is not None:
        assert final_state_ttnn.shape == (
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        ), f"Expected final_state shape {(batch_size, num_v_heads, head_k_dim, head_v_dim)}, got {final_state_ttnn.shape}"

    # Compare outputs using PCC (prints output PCC only, validates both output and state)
    if final_state_torch is not None:
        compare_fn_outputs(
            (TorchTTNNTensor(output_torch), TorchTTNNTensor(final_state_torch)),
            (output_ttnn, final_state_ttnn),
            "RecurrentDeltaNet",
        )
    else:
        compare_fn_outputs(
            TorchTTNNTensor(output_torch),
            output_ttnn,
            "RecurrentDeltaNet",
        )
