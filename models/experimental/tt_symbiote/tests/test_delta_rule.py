# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.delta_rule import TTNNFusedChunkedDeltaRule
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_fused_chunked_delta_rule(device):
    """Test fused chunked delta rule with TTNN acceleration."""
    try:
        from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
            chunk_gated_delta_rule,
        )
    except ImportError:
        pytest.skip("PyTorch reference implementation not available")

    batch_size = 1
    seq_len = 64
    num_heads = 4
    head_k_dim = 64
    head_v_dim = 128
    chunk_size = 32

    # Create test inputs (use float32 for PyTorch reference, will be converted to bfloat16 for TTNN)
    q = torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_v_dim, dtype=torch.float32)
    beta = torch.rand(batch_size, seq_len, num_heads, dtype=torch.float32)
    g = -torch.rand(batch_size, seq_len, num_heads, dtype=torch.float32) * 2

    # Run PyTorch reference implementation
    outputs_torch, final_state_torch = chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=chunk_size, output_final_state=True, use_qk_l2norm=True
    )

    # Convert inputs to bfloat16 for TTNN
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    beta = beta.to(torch.bfloat16)
    g = g.to(torch.bfloat16)

    # Wrap inputs as TorchTTNNTensor
    q_ttnn = TorchTTNNTensor(q)
    k_ttnn = TorchTTNNTensor(k)
    v_ttnn = TorchTTNNTensor(v)
    beta_ttnn = TorchTTNNTensor(beta)
    g_ttnn = TorchTTNNTensor(g)

    # Create TTNN model
    ttnn_model = TTNNFusedChunkedDeltaRule(chunk_size=chunk_size)
    set_device(ttnn_model, device)

    # Run forward pass (module_run will handle conversion from TorchTTNNTensor to ttnn.Tensor)
    outputs_ttnn, final_state_ttnn = ttnn_model(
        q=q_ttnn,
        k=k_ttnn,
        v=v_ttnn,
        beta=beta_ttnn,
        g=g_ttnn,
    )

    # Verify output shapes
    assert outputs_ttnn.shape == (
        batch_size,
        seq_len,
        num_heads,
        head_v_dim,
    ), f"Expected output shape {(batch_size, seq_len, num_heads, head_v_dim)}, got {outputs_ttnn.shape}"
    assert final_state_ttnn.shape == (
        batch_size,
        num_heads,
        head_k_dim,
        head_v_dim,
    ), f"Expected final_state shape {(batch_size, num_heads, head_k_dim, head_v_dim)}, got {final_state_ttnn.shape}"

    # Compare outputs using PCC (prints output PCC only, validates both output and state)
    compare_fn_outputs(
        (TorchTTNNTensor(outputs_torch), TorchTTNNTensor(final_state_torch)),
        (outputs_ttnn, final_state_ttnn),
        "FusedChunkedDeltaRule",
    )
