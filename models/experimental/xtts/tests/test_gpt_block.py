# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for a single XTTS-v2 GPT decoder block.

XTTS-v2's autoregressive core is a 30-layer GPT-2 stack of identical blocks.
This validates the TTNN port of one of those repeating blocks against the
pure-PyTorch HuggingFace reference, using the *real* weights from the upstream
checkpoint at https://huggingface.co/coqui/XTTS-v2 (``model.pth``).

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    # first run downloads ~1.9 GB of XTTS-v2 weights to the HF cache
    pytest models/experimental/xtts/tests/test_gpt_block.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import (
    HIDDEN_SIZE,
    MAX_GPT_SEQ_LEN,
    MAX_MEL_POS,
    MAX_TEXT_POS,
    load_xtts_state_dict,
    reference_gpt_block,
)
from models.experimental.xtts.tt.xtts_gpt_block import TtXttsGptBlock


@pytest.fixture(scope="module")
def xtts_state_dict():
    """Load the real XTTS-v2 checkpoint state dict once for the whole module."""
    return load_xtts_state_dict()


@pytest.mark.parametrize("layer_idx", [0])
@pytest.mark.parametrize(
    "seq_len",
    [
        MAX_TEXT_POS,  # 404 — max text stream length
        MAX_MEL_POS,  # 608 — max mel/audio stream length
        MAX_GPT_SEQ_LEN,  # 1012 — full concatenated [text]+[mel] GPT context
    ],
)
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_gpt_block(device, xtts_state_dict, layer_idx, seq_len, pcc, reset_seeds):
    # Reference: one repeating GPT block with real weights (causal self-attention).
    reference = reference_gpt_block(xtts_state_dict, layer_idx=layer_idx)
    logger.info(f"XTTS reference GPT block (layer {layer_idx}):\n{reference}")

    torch_input = torch.randn(1, seq_len, HIDDEN_SIZE) * 0.1
    with torch.no_grad():
        reference_output = reference(torch_input)

    # TTNN port of the same block.
    tt_block = TtXttsGptBlock(xtts_state_dict, device, layer_idx=layer_idx)
    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    # forward_prefill is the block's full causal pass (also returns the prompt K/V, which we drop).
    tt_output = ttnn.to_torch(tt_block.forward_prefill(tt_input)[0]).float()[:, :seq_len, :]

    does_pass, pcc_message = comp_pcc(reference_output, tt_output, pcc)
    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"seq_len={seq_len} layer={layer_idx}: {pcc_message}")

    assert does_pass, f"XTTS GPT block PCC below {pcc}: {pcc_message}"
