# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import (
    HIDDEN_SIZE,
    MAX_MEL_POS,
    MAX_TEXT_POS,
    reference_gpt_block,
)
from models.experimental.xtts.tt.xtts_gpt_block import TtXttsGptBlock


@pytest.mark.parametrize("layer_idx", [0])
@pytest.mark.parametrize(
    "seq_len",
    [
        MAX_TEXT_POS,
        MAX_MEL_POS,
    ],
)
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_gpt_block(device, xtts_state_dict, layer_idx, seq_len, pcc, reset_seeds):
    """Compare a single TTNN GPT block prefill to the PyTorch reference via PCC."""
    reference = reference_gpt_block(xtts_state_dict, layer_idx=layer_idx)
    logger.info(f"XTTS reference GPT block (layer {layer_idx}):\n{reference}")

    torch_input = torch.randn(1, seq_len, HIDDEN_SIZE) * 0.1
    with torch.no_grad():
        reference_output = reference(torch_input)

    tt_block = TtXttsGptBlock(xtts_state_dict, device, layer_idx=layer_idx)
    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    tt_output = ttnn.to_torch(tt_block.forward_prefill(tt_input)[0]).float()[:, :seq_len, :]

    does_pass, pcc_message = comp_pcc(reference_output, tt_output, pcc)
    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"seq_len={seq_len} layer={layer_idx}: {pcc_message}")

    assert does_pass, f"XTTS GPT block PCC below {pcc}: {pcc_message}"
