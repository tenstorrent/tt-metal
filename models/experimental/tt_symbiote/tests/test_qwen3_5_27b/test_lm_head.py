# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LM Head accuracy tests for Qwen3.5-27B.

Tests the final lm_head linear projection (hidden_size -> vocab_size)
using TTNNLinear against PyTorch reference.
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

PCC_LM_HEAD = 0.99


# ──────────────────────────────────────────────────────────────────────
# Test: LM head with single token input
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_lm_head_pcc(device, model_4_layers):
    """Test lm_head as TTNNLinear, single token input. PCC >= 0.99."""
    import ttnn
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.utils.device_management import set_device

    model, config = model_4_layers
    torch_lm_head = model.lm_head

    batch_size, seq_len = 1, 1
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_lm_head(x)

    ttnn_proj = TTNNLinear.from_torch(torch_lm_head)
    set_device(ttnn_proj, device)
    ttnn_proj.preprocess_weights()
    ttnn_proj.move_weights_to_device()

    ttnn_input = TorchTTNNTensor(x)
    ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
    ttnn_out = ttnn_proj(ttnn_input.ttnn_tensor)

    pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LM_HEAD)
    print(f"  lm_head (seq_len=1) PCC = {pcc:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Test: LM head with seq_len=32 input
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_lm_head_seq32_pcc(device, model_4_layers):
    """Test lm_head as TTNNLinear with seq_len=32. PCC >= 0.99."""
    import ttnn
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.utils.device_management import set_device

    model, config = model_4_layers
    torch_lm_head = model.lm_head

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_lm_head(x)

    ttnn_proj = TTNNLinear.from_torch(torch_lm_head)
    set_device(ttnn_proj, device)
    ttnn_proj.preprocess_weights()
    ttnn_proj.move_weights_to_device()

    ttnn_input = TorchTTNNTensor(x)
    ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
    ttnn_out = ttnn_proj(ttnn_input.ttnn_tensor)

    pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LM_HEAD)
    print(f"  lm_head (seq_len=32) PCC = {pcc:.6f}")
