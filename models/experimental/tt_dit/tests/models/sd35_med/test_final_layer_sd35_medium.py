# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.experimental.tt_dit.models.transformers.sd35_med.final_layer_sd35_medium import SD35MediumFinalLayer


def modulate(x, shift, scale):
    """Reference modulation function matching MM-DiT"""
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(torch.nn.Module):
    """Reference PyTorch implementation matching MM-DiT FinalLayer"""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        total_out_channels: int = None,
        dtype=torch.bfloat16,
        device=None,
    ):
        super().__init__()
        self.norm_final = torch.nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )

        if total_out_channels is None:
            self.linear = torch.nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                dtype=dtype,
                device=device,
            )
        else:
            self.linear = torch.nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)

        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "hidden_size, patch_size, out_channels, seq_len, batch_size",
    [
        (1536, 2, 16, 1024, 1),  # SD3.5 Medium final layer
        (1536, 2, 16, 512, 1),  # SD3.5 Medium final layer (shorter seq)
    ],
    ids=["sd35_med_1k", "sd35_med_512"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_final_layer(
    device, dtype, hidden_size, patch_size, out_channels, seq_len, batch_size, reset_seeds
):
    """Test SD3.5 Medium FinalLayer forward pass"""
    torch.manual_seed(1234)

    # Create reference model
    reference_model = FinalLayer(
        hidden_size=hidden_size,
        patch_size=patch_size,
        out_channels=out_channels,
        dtype=torch.bfloat16,
    )
    reference_model.eval()

    # Create parallel config
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=None),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=None),
    )

    # Create TTNN model
    tt_model = SD35MediumFinalLayer(
        hidden_size=hidden_size,
        patch_size=patch_size,
        out_channels=out_channels,
        mesh_device=device,
        parallel_config=parallel_config,
    )

    # Load weights
    state_dict = reference_model.state_dict()
    tt_model.load_state_dict(state_dict)

    # Create inputs
    x_input = torch.randn(1, batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    c_input = torch.randn(1, batch_size, hidden_size, dtype=torch.bfloat16)

    # Reference forward
    with torch.no_grad():
        ref_output = reference_model(x_input.squeeze(0), c_input.squeeze(0))

    # TTNN forward
    tt_x_input = ttnn.from_torch(x_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_c_input = ttnn.from_torch(c_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = tt_model(tt_x_input, tt_c_input)

    # Convert back and compare
    tt_output_torch = ttnn.to_torch(tt_output)[0, :batch_size, :seq_len, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_required)

    logger.info(f"Final Layer PCC: {pcc_message}")
    assert passing, f"Final Layer output does not meet PCC requirement {pcc_required}."

    logger.info("SD3.5 Medium final layer test passed!")
