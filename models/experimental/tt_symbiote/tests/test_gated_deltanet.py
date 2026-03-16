# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.gated_deltanet import TTNNGatedDeltaNet
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_gated_deltanet_recurrent(device):
    """Test Gated DeltaNet in recurrent mode with TTNN acceleration."""
    try:
        from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_deltanet import (
            gated_deltanet_forward,
        )
        from models.experimental.gated_attention_gated_deltanet.tests.test_gated_deltanet import (
            make_gated_deltanet_params,
        )
    except ImportError:
        pytest.skip("PyTorch reference implementation not available")

    seq_len = 32
    params = make_gated_deltanet_params(seq_len=seq_len)

    # Run PyTorch reference implementation
    output_torch, _ = gated_deltanet_forward(**params, mode="fused_recurrent", output_final_state=False)

    # Convert input to bfloat16 for TTNN
    hidden_states_ttnn = params["hidden_states"].to(torch.bfloat16)

    # Create TTNN model
    ttnn_model = TTNNGatedDeltaNet(
        num_heads=params["num_heads"],
        num_v_heads=params["num_v_heads"],
        head_k_dim=params["head_k_dim"],
        head_v_dim=params["head_v_dim"],
        conv_kernel_size=params["conv_kernel_size"],
        use_gate=params["use_gate"],
        allow_neg_eigval=False,
        norm_eps=1e-5,
        mode="recurrent",
        chunk_size=64,
    )
    set_device(ttnn_model, device)

    ttnn_model.q_proj_weight = ttnn.from_torch(
        params["q_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.k_proj_weight = ttnn.from_torch(
        params["k_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.v_proj_weight = ttnn.from_torch(
        params["v_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.a_proj_weight = ttnn.from_torch(
        params["a_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.b_proj_weight = ttnn.from_torch(
        params["b_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.o_proj_weight = ttnn.from_torch(
        params["o_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if params["g_proj_weight"] is not None:
        ttnn_model.g_proj_weight = ttnn.from_torch(
            params["g_proj_weight"].T.contiguous().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    ttnn_model.q_conv_weight = ttnn.from_torch(
        params["q_conv_weight"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.k_conv_weight = ttnn.from_torch(
        params["k_conv_weight"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.v_conv_weight = ttnn.from_torch(
        params["v_conv_weight"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_model.q_conv_bias = (
        ttnn.from_torch(
            params["q_conv_bias"].to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if params["q_conv_bias"] is not None
        else None
    )
    ttnn_model.k_conv_bias = (
        ttnn.from_torch(
            params["k_conv_bias"].to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if params["k_conv_bias"] is not None
        else None
    )
    ttnn_model.v_conv_bias = (
        ttnn.from_torch(
            params["v_conv_bias"].to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if params["v_conv_bias"] is not None
        else None
    )

    ttnn_model.A_log = ttnn.from_torch(
        params["A_log"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.dt_bias = ttnn.from_torch(
        params["dt_bias"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.o_norm_weight = ttnn.from_torch(
        params["o_norm_weight"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Wrap input as TorchTTNNTensor
    hidden_states_ttnn_tensor = TorchTTNNTensor(hidden_states_ttnn)

    # Run forward pass
    output_ttnn = ttnn_model(hidden_states=hidden_states_ttnn_tensor)

    # Verify output shape
    assert output_ttnn.shape == (
        params["hidden_states"].shape[0],
        params["hidden_states"].shape[1],
        params["hidden_states"].shape[2],
    ), f"Expected output shape {params['hidden_states'].shape}, got {output_ttnn.shape}"

    # Compare outputs using PCC
    compare_fn_outputs(
        TorchTTNNTensor(output_torch),
        output_ttnn,
        "GatedDeltaNet",
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_gated_deltanet_chunked(device):
    """Test Gated DeltaNet in chunked mode with TTNN acceleration."""
    try:
        from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_deltanet import (
            gated_deltanet_forward,
        )
        from models.experimental.gated_attention_gated_deltanet.tests.test_gated_deltanet import (
            make_gated_deltanet_params,
        )
    except ImportError:
        pytest.skip("PyTorch reference implementation not available")

    seq_len = 128
    chunk_size = 64
    params = make_gated_deltanet_params(seq_len=seq_len)

    # Run PyTorch reference implementation
    output_torch, _ = gated_deltanet_forward(**params, mode="chunk", chunk_size=chunk_size, output_final_state=False)

    # Convert input to bfloat16 for TTNN
    hidden_states_ttnn = params["hidden_states"].to(torch.bfloat16)

    # Create TTNN model
    ttnn_model = TTNNGatedDeltaNet(
        num_heads=params["num_heads"],
        num_v_heads=params["num_v_heads"],
        head_k_dim=params["head_k_dim"],
        head_v_dim=params["head_v_dim"],
        conv_kernel_size=params["conv_kernel_size"],
        use_gate=params["use_gate"],
        allow_neg_eigval=False,
        norm_eps=1e-5,
        mode="chunk",
        chunk_size=chunk_size,
    )
    set_device(ttnn_model, device)

    ttnn_model.q_proj_weight = ttnn.from_torch(
        params["q_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.k_proj_weight = ttnn.from_torch(
        params["k_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.v_proj_weight = ttnn.from_torch(
        params["v_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.a_proj_weight = ttnn.from_torch(
        params["a_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.b_proj_weight = ttnn.from_torch(
        params["b_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.o_proj_weight = ttnn.from_torch(
        params["o_proj_weight"].T.contiguous().to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if params["g_proj_weight"] is not None:
        ttnn_model.g_proj_weight = ttnn.from_torch(
            params["g_proj_weight"].T.contiguous().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    ttnn_model.q_conv_weight = ttnn.from_torch(
        params["q_conv_weight"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.k_conv_weight = ttnn.from_torch(
        params["k_conv_weight"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.v_conv_weight = ttnn.from_torch(
        params["v_conv_weight"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_model.q_conv_bias = (
        ttnn.from_torch(
            params["q_conv_bias"].to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if params["q_conv_bias"] is not None
        else None
    )
    ttnn_model.k_conv_bias = (
        ttnn.from_torch(
            params["k_conv_bias"].to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if params["k_conv_bias"] is not None
        else None
    )
    ttnn_model.v_conv_bias = (
        ttnn.from_torch(
            params["v_conv_bias"].to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if params["v_conv_bias"] is not None
        else None
    )

    ttnn_model.A_log = ttnn.from_torch(
        params["A_log"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.dt_bias = ttnn.from_torch(
        params["dt_bias"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_model.o_norm_weight = ttnn.from_torch(
        params["o_norm_weight"].to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    hidden_states_ttnn_tensor = TorchTTNNTensor(hidden_states_ttnn)

    output_ttnn = ttnn_model(hidden_states=hidden_states_ttnn_tensor)

    assert output_ttnn.shape == (
        params["hidden_states"].shape[0],
        params["hidden_states"].shape[1],
        params["hidden_states"].shape[2],
    ), f"Expected output shape {params['hidden_states'].shape}, got {output_ttnn.shape}"

    compare_fn_outputs(
        TorchTTNNTensor(output_torch),
        output_ttnn,
        "GatedDeltaNet",
    )
