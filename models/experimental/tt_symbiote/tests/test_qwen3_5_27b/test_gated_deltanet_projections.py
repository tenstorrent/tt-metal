# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GatedDeltaNet in_proj_b and in_proj_a accuracy tests for Qwen3.5-27B.

These are the beta and alpha projections inside the GatedDeltaNet (linear
attention) layer. Both are small linear projections: hidden_size -> num_v_heads.
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


# ──────────────────────────────────────────────────────────────────────
# Test: in_proj_b (beta projection)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_in_proj_b_pcc(device, model_1_layer):
    """Compare PyTorch vs TTNNLinear for beta projection (in_proj_b). PCC >= 0.99."""
    import ttnn
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.utils.device_management import set_device

    model, config = model_1_layer
    torch_proj = model.model.layers[0].linear_attn.in_proj_b

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
    print(f"  in_proj_b PCC = {pcc:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Test: in_proj_a (alpha projection)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_in_proj_a_pcc(device, model_1_layer):
    """Compare PyTorch vs TTNNLinear for alpha projection (in_proj_a). PCC >= 0.99."""
    import ttnn
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.utils.device_management import set_device

    model, config = model_1_layer
    torch_proj = model.model.layers[0].linear_attn.in_proj_a

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
    print(f"  in_proj_a PCC = {pcc:.6f}")
