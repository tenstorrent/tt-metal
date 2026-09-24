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
    NUM_LAYERS,
)
from models.experimental.xtts.reference.xtts_gpt_stack import reference_gpt_stack
from models.experimental.xtts.tt.xtts_gpt_stack import TtXttsGptStack


@pytest.mark.parametrize(
    "seq_len",
    [
        MAX_TEXT_POS,
        MAX_MEL_POS,
    ],
)
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_gpt_stack(device, xtts_state_dict, seq_len, pcc, reset_seeds):
    """Compare the full TTNN GPT stack to the PyTorch reference via PCC."""
    reference = reference_gpt_stack(xtts_state_dict, num_layers=NUM_LAYERS)
    logger.info(f"XTTS reference GPT decoder stack ({NUM_LAYERS} blocks + ln_f):\n{reference}")

    torch_input = torch.randn(1, seq_len, HIDDEN_SIZE) * 0.1
    with torch.no_grad():
        reference_output = reference(torch_input)

    tt_stack = TtXttsGptStack(xtts_state_dict, device, num_layers=NUM_LAYERS)
    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    tt_output = ttnn.to_torch(tt_stack(tt_input)).float()[:, :seq_len, :]

    does_pass, pcc_message = comp_pcc(reference_output, tt_output, pcc)
    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"seq_len={seq_len} stack({NUM_LAYERS})+ln_f: {pcc_message}")

    assert does_pass, f"XTTS GPT stack PCC below {pcc}: {pcc_message}"
