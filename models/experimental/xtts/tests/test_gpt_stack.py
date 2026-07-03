# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the full XTTS-v2 GPT decoder stack (30 blocks + final LayerNorm).

Validates the TTNN port of the whole GPT-2 decoder — the 30 repeating blocks
followed by ``ln_f`` — against the pure-PyTorch HuggingFace reference, using the
*real* weights from the upstream checkpoint at
https://huggingface.co/coqui/XTTS-v2 (``model.pth``).

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    # first run downloads ~1.9 GB of XTTS-v2 weights to the HF cache
    pytest models/experimental/xtts/tests/test_gpt_stack.py
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
    NUM_LAYERS,
    load_xtts_state_dict,
)
from models.experimental.xtts.reference.xtts_gpt_stack import reference_gpt_stack
from models.experimental.xtts.tt.xtts_gpt_stack import TtXttsGptStack


@pytest.fixture(scope="module")
def xtts_state_dict():
    """Load the real XTTS-v2 checkpoint state dict once for the whole module."""
    return load_xtts_state_dict()


@pytest.mark.parametrize(
    "seq_len",
    [
        MAX_TEXT_POS,  # 404 — max text stream length
        MAX_MEL_POS,  # 608 — max mel/audio stream length
        MAX_GPT_SEQ_LEN,  # 1012 — full concatenated [text]+[mel] GPT context
    ],
)
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_gpt_stack(device, xtts_state_dict, seq_len, pcc, reset_seeds):
    # Reference: all 30 GPT decoder blocks + final LayerNorm, with real weights.
    reference = reference_gpt_stack(xtts_state_dict, num_layers=NUM_LAYERS)
    logger.info(f"XTTS reference GPT decoder stack ({NUM_LAYERS} blocks + ln_f):\n{reference}")

    torch_input = torch.randn(1, seq_len, HIDDEN_SIZE) * 0.1
    with torch.no_grad():
        reference_output = reference(torch_input)

    # TTNN port of the same 30-block stack.
    tt_stack = TtXttsGptStack(xtts_state_dict, device, num_layers=NUM_LAYERS)
    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    tt_output = ttnn.to_torch(tt_stack(tt_input)).float()[:, :seq_len, :]

    does_pass, pcc_message = comp_pcc(reference_output, tt_output, pcc)
    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"seq_len={seq_len} stack({NUM_LAYERS})+ln_f: {pcc_message}")

    assert does_pass, f"XTTS GPT stack PCC below {pcc}: {pcc_message}"
