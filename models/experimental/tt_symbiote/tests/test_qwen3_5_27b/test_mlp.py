# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU MLP accuracy tests for Qwen3.5-27B.

Qwen3_5MLP: gate_proj, up_proj, down_proj with SiLU activation.
forward: down_proj(act_fn(gate_proj(x)) * up_proj(x))
"""

import pytest
import torch

from .conftest import (
    assert_with_pcc,
    get_config_attr,
    skip_no_ttnn,
    skip_no_transformers,
    skip_no_symbiote,
)

PCC_LINEAR_PROJ = 0.99
PCC_MLP_E2E = 0.97


# ──────────────────────────────────────────────────────────────────────
# Test: gate_proj individual linear
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_mlp_gate_proj_pcc(device, model_1_layer):
    """Test gate_proj as TTNNLinear. PCC >= 0.99."""
    import ttnn
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.utils.device_management import set_device

    model, config = model_1_layer
    torch_proj = model.model.layers[0].mlp.gate_proj

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_proj(x)

    ttnn_proj = TTNNLinear.from_torch(torch_proj)
    set_device(ttnn_proj, device)
    ttnn_proj.preprocess_weights()
    ttnn_proj.move_weights_to_device()

    ttnn_input = TorchTTNNTensor(x)
    ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
    ttnn_out = ttnn_proj(ttnn_input.ttnn_tensor)

    pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LINEAR_PROJ)
    print(f"  gate_proj PCC = {pcc:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Test: up_proj individual linear
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_mlp_up_proj_pcc(device, model_1_layer):
    """Test up_proj as TTNNLinear. PCC >= 0.99."""
    import ttnn
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.utils.device_management import set_device

    model, config = model_1_layer
    torch_proj = model.model.layers[0].mlp.up_proj

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_proj(x)

    ttnn_proj = TTNNLinear.from_torch(torch_proj)
    set_device(ttnn_proj, device)
    ttnn_proj.preprocess_weights()
    ttnn_proj.move_weights_to_device()

    ttnn_input = TorchTTNNTensor(x)
    ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
    ttnn_out = ttnn_proj(ttnn_input.ttnn_tensor)

    pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LINEAR_PROJ)
    print(f"  up_proj PCC = {pcc:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Test: down_proj with intermediate-sized input
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_mlp_down_proj_pcc(device, model_1_layer):
    """Test down_proj as TTNNLinear with intermediate-sized input. PCC >= 0.99."""
    import ttnn
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.utils.device_management import set_device

    model, config = model_1_layer
    torch_proj = model.model.layers[0].mlp.down_proj

    batch_size, seq_len = 1, 32
    intermediate_size = get_config_attr(config, "intermediate_size")
    x = torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_proj(x)

    ttnn_proj = TTNNLinear.from_torch(torch_proj)
    set_device(ttnn_proj, device)
    ttnn_proj.preprocess_weights()
    ttnn_proj.move_weights_to_device()

    ttnn_input = TorchTTNNTensor(x)
    ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
    ttnn_out = ttnn_proj(ttnn_input.ttnn_tensor)

    pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LINEAR_PROJ)
    print(f"  down_proj PCC = {pcc:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Test: PyTorch-only MLP forward sanity
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_mlp_pytorch_forward_sanity(model_1_layer):
    """PyTorch-only MLP forward: check shape, dtype, and no NaN."""
    model, config = model_1_layer
    mlp = model.model.layers[0].mlp

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        out = mlp(x)

    assert out.shape == (batch_size, seq_len, hidden_size), f"Unexpected shape: {out.shape}"
    assert out.dtype == torch.bfloat16, f"Unexpected dtype: {out.dtype}"
    assert not torch.isnan(out).any(), "NaN detected in MLP output"
    print(f"  MLP sanity: shape={out.shape}, dtype={out.dtype}, no NaN")


# ──────────────────────────────────────────────────────────────────────
# Test: End-to-end MLP with all 3 projections replaced by TTNNLinear
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_mlp_end_to_end_pcc(device, model_1_layer):
    """Replace all 3 MLP projections with TTNNLinear, run full MLP, compare. PCC >= 0.97."""
    import ttnn
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.utils.device_management import set_device

    model, config = model_1_layer
    mlp = model.model.layers[0].mlp

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # PyTorch reference: full MLP forward
    with torch.no_grad():
        torch_out = mlp(x)

    # Build TTNN versions of all 3 projections
    ttnn_gate = TTNNLinear.from_torch(mlp.gate_proj)
    ttnn_up = TTNNLinear.from_torch(mlp.up_proj)
    ttnn_down = TTNNLinear.from_torch(mlp.down_proj)

    for proj in [ttnn_gate, ttnn_up, ttnn_down]:
        set_device(proj, device)
        proj.preprocess_weights()
        proj.move_weights_to_device()

    # Convert input to TTNN
    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    # Run gate and up projections on TTNN
    gate_out_tt = ttnn_gate.forward(tt_input)
    up_out_tt = ttnn_up.forward(tt_input)

    # Convert back to torch for SiLU activation and element-wise multiply
    gate_out_torch = ttnn.to_torch(gate_out_tt).to(torch.bfloat16)
    up_out_torch = ttnn.to_torch(up_out_tt).to(torch.bfloat16)
    ttnn.deallocate(gate_out_tt)
    ttnn.deallocate(up_out_tt)

    import torch.nn.functional as F

    intermediate = F.silu(gate_out_torch) * up_out_torch

    # Run down_proj on TTNN
    tt_inter = ttnn.from_torch(intermediate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_inter = ttnn.to_device(tt_inter, device)
    down_out_tt = ttnn_down.forward(tt_inter)

    pcc = assert_with_pcc(torch_out, down_out_tt, PCC_MLP_E2E)
    print(f"  MLP end-to-end PCC = {pcc:.6f}")
