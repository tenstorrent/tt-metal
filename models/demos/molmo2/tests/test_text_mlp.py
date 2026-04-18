# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Molmo2 Text Model MLP.

Validates SwiGLU MLP against PyTorch reference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc


def get_mlp_weights(model_id: str = "allenai/Molmo2-8B", layer_num: int = 0):
    """Load MLP weights from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    prefix = f"model.transformer.blocks.{layer_num}.mlp"
    keys = [
        f"{prefix}.ff_proj.weight",
        f"{prefix}.ff_out.weight",
    ]
    return load_state_dict_from_safetensors(model_id, keys)


class ReferenceSwiGLU(nn.Module):
    """PyTorch reference SwiGLU MLP."""

    def __init__(self, hidden_dim=4096, intermediate_dim=12288):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def test_text_mlp(device):
    """Test TextMLP against PyTorch reference."""
    from models.demos.molmo2.tt.text_mlp import TextMLP

    model_id = "allenai/Molmo2-8B"
    layer_num = 0
    hidden_dim = 4096
    intermediate_dim = 12288
    seq_len = 32

    # Load weights
    state_dict = get_mlp_weights(model_id, layer_num)
    prefix = f"model.transformer.blocks.{layer_num}.mlp"

    # Create reference model
    ref_model = ReferenceSwiGLU(hidden_dim, intermediate_dim)

    # Split fused ff_proj into gate and up
    ff_proj = state_dict[f"{prefix}.ff_proj.weight"]
    ref_model.gate_proj.weight.data = ff_proj[:intermediate_dim, :]
    ref_model.up_proj.weight.data = ff_proj[intermediate_dim:, :]
    ref_model.down_proj.weight.data = state_dict[f"{prefix}.ff_out.weight"]
    ref_model.eval()

    # Create random input
    torch.manual_seed(42)
    x_torch = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)

    # Reference forward
    with torch.no_grad():
        ref_output = ref_model(x_torch)

    # Create TTNN MLP
    tt_mlp = TextMLP(
        mesh_device=device,
        state_dict=state_dict,
        layer_num=layer_num,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        dtype=ttnn.bfloat8_b,
    )

    # Convert input to TTNN
    x_ttnn = ttnn.from_torch(
        x_torch.unsqueeze(0),  # [1, 1, seq_len, hidden_dim]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN forward
    tt_output = tt_mlp(x_ttnn)

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).squeeze(0)

    # Compare with PCC
    passing, pcc_msg = comp_pcc(ref_output, tt_output_torch, pcc=0.99)
    print(f"TextMLP (layer={layer_num}) PCC: {pcc_msg}")

    assert passing, f"TextMLP failed PCC check: {pcc_msg}"


if __name__ == "__main__":
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        test_text_mlp(device)
        print("Test passed!")
    finally:
        ttnn.close_device(device)
