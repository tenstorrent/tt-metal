# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from loguru import logger
import ttnn
from models.common.utility_functions import comp_pcc, comp_allclose
from models.experimental.tt_dit.models.transformers.sd35_med.mlp_sd35_medium import SD35MediumMlp


class Mlp(torch.nn.Module):
    """Reference PyTorch MLP matching MM-DiT implementation"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        dtype=torch.bfloat16,
        act_kwargs=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Add dtype=torch.bfloat16 to match input dtype
        self.fc1 = torch.nn.Linear(in_features, hidden_features, bias=bias, dtype=torch.bfloat16)
        # Handle act_layer: if it's already an instance, use it; otherwise instantiate with kwargs
        if isinstance(act_layer, torch.nn.Module):
            self.act = act_layer
        else:
            act_kwargs = act_kwargs or {}
            self.act = act_layer(**act_kwargs)
        self.fc2 = torch.nn.Linear(hidden_features, out_features, bias=bias, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "in_features, hidden_features, batch_size, seq_len",
    [
        (1536, 6144, 1, 1024),  # SD3.5 Medium typical dimensions
        (1536, 6144, 1, 4096),  # Larger sequence
        (1536, 6144, 2, 512),  # Batch size > 1
    ],
    ids=["sd35_med_1k", "sd35_med_4k", "sd35_med_batch2"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_mlp(device, dtype, in_features, hidden_features, batch_size, seq_len, reset_seeds):
    """
    Test SD3.5 Medium MLP layer.
    Validates that the TTNN implementation matches PyTorch reference.
    """
    torch.manual_seed(1234)

    out_features = in_features

    # Create reference model
    reference_model = Mlp(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        bias=True,
    )
    reference_model.eval()

    # Create TTNN model
    tt_model = SD35MediumMlp(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        bias=True,
        mesh_device=device,
    )

    # Load weights from reference model
    state_dict = {
        "fc1.weight": reference_model.fc1.weight,
        "fc1.bias": reference_model.fc1.bias,
        "fc2.weight": reference_model.fc2.weight,
        "fc2.bias": reference_model.fc2.bias,
    }
    tt_model.load_torch_state_dict(state_dict)

    # Create random input
    torch_input = torch.randn(1, batch_size, seq_len, in_features, dtype=torch.bfloat16)

    # Reference forward pass
    with torch.no_grad():
        reference_output = reference_model(torch_input.squeeze(0))

    # Convert to TTNN tensor
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN forward pass
    tt_output = tt_model(tt_input)

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch[0, :batch_size, :seq_len, :out_features]

    # Compare outputs
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("SD3.5 Medium MLP Passed!")
    else:
        logger.warning("SD3.5 Medium MLP Failed!")

    assert passing, f"MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
