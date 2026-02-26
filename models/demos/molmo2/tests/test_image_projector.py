# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Molmo2 Image Projector (SwiGLU).

Validates ImageProjector against PyTorch reference implementation.
"""


import pytest
import torch
import torch.nn as nn

import ttnn
from models.common.utility_functions import comp_pcc


def get_projector_weights(model_id: str = "allenai/Molmo2-8B"):
    """Load projector weights from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    prefix = "model.vision_backbone.image_projector"
    keys = [
        f"{prefix}.w1.weight",
        f"{prefix}.w2.weight",
        f"{prefix}.w3.weight",
    ]
    return load_state_dict_from_safetensors(model_id, keys)


class ReferenceSwiGLU(nn.Module):
    """PyTorch reference implementation of SwiGLU projector."""

    def __init__(self, input_dim=1152, hidden_dim=12288, output_dim=4096):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x))


@pytest.mark.parametrize("num_tokens", [64, 256, 729])
def test_image_projector(num_tokens, device):
    """
    Test ImageProjector against PyTorch reference.

    Args:
        num_tokens: Number of input tokens
        device: TTNN device fixture
    """
    from models.demos.molmo2.tt.image_projector import ImageProjector

    model_id = "allenai/Molmo2-8B"
    input_dim = 1152
    hidden_dim = 12288
    output_dim = 4096

    # Load weights
    state_dict = get_projector_weights(model_id)

    # Create reference model
    ref_model = ReferenceSwiGLU(input_dim, hidden_dim, output_dim)
    prefix = "model.vision_backbone.image_projector"
    ref_model.w1.weight.data = state_dict[f"{prefix}.w1.weight"]
    ref_model.w2.weight.data = state_dict[f"{prefix}.w2.weight"]
    ref_model.w3.weight.data = state_dict[f"{prefix}.w3.weight"]
    ref_model.eval()

    # Create random input
    torch.manual_seed(42)
    x_torch = torch.randn(1, num_tokens, input_dim, dtype=torch.float32)

    # Reference forward
    with torch.no_grad():
        ref_output = ref_model(x_torch)

    # Create TTNN projector
    tt_projector = ImageProjector(
        mesh_device=device,
        state_dict=state_dict,
        input_dim=input_dim,
        intermediate_dim=hidden_dim,
        output_dim=output_dim,
        dtype=ttnn.bfloat8_b,
    )

    # Convert input to TTNN
    x_ttnn = ttnn.from_torch(
        x_torch.unsqueeze(0),  # [1, 1, num_tokens, input_dim]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN forward
    tt_output = tt_projector(x_ttnn)

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0)

    # Compare with PCC
    passing, pcc_msg = comp_pcc(ref_output, tt_output_torch, pcc=0.99)
    print(f"ImageProjector (num_tokens={num_tokens}) PCC: {pcc_msg}")

    assert passing, f"ImageProjector failed PCC check: {pcc_msg}"


if __name__ == "__main__":
    # Quick standalone test
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        test_image_projector(256, device)
        print("Test passed!")
    finally:
        ttnn.close_device(device)
