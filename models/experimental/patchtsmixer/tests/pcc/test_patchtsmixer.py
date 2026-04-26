# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixer
from models.experimental.patchtsmixer.reference.patchtsmixer import PatchTSMixer as TorchPatchTSMixer


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_patchtsmixer_inference(device, reset_seeds):
    # Model parameters
    input_length = 96
    patch_length = 16
    stride = 8
    num_input_channels = 7
    d_model = 128
    num_blocks = 2
    expansion_factor = 2.0
    dropout = 0.1
    num_output_steps = 48

    # Create TT model
    tt_model = TtPatchTSMixer(
        device=device,
        input_length=input_length,
        patch_length=patch_length,
        stride=stride,
        num_input_channels=num_input_channels,
        d_model=d_model,
        num_blocks=num_blocks,
        expansion_factor=expansion_factor,
        dropout=dropout,
        num_output_steps=num_output_steps,
        norm_type="batch",
        channel_mixing=True,
        use_gated_attention=True,
        enable_reconciliation=True,
    )

    # Create torch model for comparison
    torch_model = TorchPatchTSMixer(
        input_length=input_length,
        patch_length=patch_length,
        stride=stride,
        num_input_channels=num_input_channels,
        d_model=d_model,
        num_blocks=num_blocks,
        expansion_factor=expansion_factor,
        dropout=dropout,
        num_output_steps=num_output_steps,
        norm_type="batch",
        channel_mixing=True,
        use_gated_attention=True,
        enable_reconciliation=True,
    )
    torch_model.eval()

    # Generate input
    batch_size = 2
    torch_input = torch.randn(batch_size, num_input_channels, input_length)
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run inference
    tt_output = tt_model(tt_input)
    torch_output = torch_model(torch_input)

    # Convert TT output to torch
    tt_output_torch = ttnn.to_torch(tt_output)

    # Normalize outputs for comparison
    tt_output_torch = tt_output_torch.detach().numpy()
    torch_output = torch_output.detach().numpy()

    # Check shape
    assert tt_output_torch.shape == torch_output.shape

    # Check values (PCC > 0.95)
    atol, rtol = 1e-2, 1e-2
    passing_pcc = torch.allclose(
        torch.from_numpy(tt_output_torch),
        torch.from_numpy(torch_output),
        atol=atol,
        rtol=rtol,
    )
    assert passing_pcc, f"PCC failed: got {passing_pcc}"