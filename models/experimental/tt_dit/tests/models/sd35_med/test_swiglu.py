# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from models.common.utility_functions import comp_pcc, comp_allclose
from models.experimental.tt_dit.models.transformers.sd35_med.swiglu_sd35_medium import SD35MediumSwiGLU


class SwiGLUFeedForward(torch.nn.Module):
    """Reference PyTorch SwiGLU matching the reference implementation"""

    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None, bias=False, dtype=torch.bfloat16):
        super().__init__()

        # Same hidden dim calculation as reference
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=bias, dtype=torch.bfloat16)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=bias, dtype=torch.bfloat16)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=bias, dtype=torch.bfloat16)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "dim, hidden_dim, multiple_of, batch_size, seq_len",
    [
        (1536, 4096, 256, 1, 1024),  # SD3.5 Medium typical
        (1536, 4096, 256, 1, 4096),  # Larger sequence
        (1536, 4096, 256, 2, 512),  # Batch size > 1
    ],
    ids=["sd35_med_1k", "sd35_med_4k", "sd35_med_batch2"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_swiglu(device, dtype, dim, hidden_dim, multiple_of, batch_size, seq_len, reset_seeds):
    """
    Test SD3.5 Medium SwiGLU layer.
    Validates that the TTNN implementation matches PyTorch reference.
    """
    torch.manual_seed(1234)

    # Create reference model
    reference_model = SwiGLUFeedForward(
        dim=dim,
        hidden_dim=hidden_dim,
        multiple_of=multiple_of,
        bias=False,
    )
    reference_model.eval()

    # Create TTNN model
    tt_model = SD35MediumSwiGLU(
        dim=dim,
        hidden_dim=hidden_dim,
        multiple_of=multiple_of,
        bias=False,
        mesh_device=device,
    )

    # Load weights from reference model
    state_dict = {
        "w1.weight": reference_model.w1.weight,
        "w2.weight": reference_model.w2.weight,
        "w3.weight": reference_model.w3.weight,
    }
    tt_model.load_torch_state_dict(state_dict)

    # Create random input
    torch_input = torch.randn(1, batch_size, seq_len, dim, dtype=torch.bfloat16)

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
    tt_output_torch = tt_output_torch[0, :batch_size, :seq_len, :dim]

    # Compare outputs
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("SD3.5 Medium SwiGLU Passed!")
    else:
        logger.warning("SD3.5 Medium SwiGLU Failed!")

    assert passing, f"SwiGLU output does not meet PCC requirement {pcc_required}: {pcc_message}."
