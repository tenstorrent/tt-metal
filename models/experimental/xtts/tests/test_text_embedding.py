# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 text input path — text ids -> input of the GPT decoder.

Validates the TTNN port of the text branch (text token embedding + learned
position embedding, i.e. the text portion of the ``[text] + [mel]`` stream that
feeds the GPT-2 decoder) against the pure-PyTorch HuggingFace reference, using
the *real* weights from https://huggingface.co/coqui/XTTS-v2 (``model.pth``).

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    # first run downloads ~1.9 GB of XTTS-v2 weights to the HF cache
    pytest models/experimental/xtts/tests/test_text_embedding.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text, reference_text_embedding
from models.experimental.xtts.tt.xtts_text_embedding import TtXttsTextEmbedding


@pytest.fixture(scope="module")
def xtts_state_dict():
    """Load the real XTTS-v2 checkpoint state dict once for the whole module."""
    return load_xtts_state_dict()


@pytest.mark.parametrize(
    "input_text",
    [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "text to speech synthesis on tenstorrent hardware",
    ],
)
@pytest.mark.parametrize("pcc", [0.99])
def test_xtts_text_embedding(device, xtts_state_dict, input_text, pcc, reset_seeds):
    # Preprocess: raw text -> XTTS BPE token ids (the real input to the text path).
    text_ids = preprocess_text(input_text, lang="en")
    text_len = text_ids.shape[1]
    logger.info(f"input text: {input_text!r} -> {text_len} tokens: {text_ids.tolist()[0]}")

    # Reference: text token embedding + learned position embedding, real weights.
    reference = reference_text_embedding(xtts_state_dict)
    logger.info(f"XTTS reference text embedding:\n{reference}")

    with torch.no_grad():
        reference_output = reference(text_ids)

    # TTNN port of the same text path.
    tt_text_embedding = TtXttsTextEmbedding(xtts_state_dict, device)
    logger.info(f"XTTS TTNN text embedding:\n{tt_text_embedding}")

    tt_output = ttnn.to_torch(tt_text_embedding(text_ids)).float()[:, :text_len, :]

    does_pass, pcc_message = comp_pcc(reference_output, tt_output, pcc)
    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"text_embedding (text_len={text_len}): {pcc_message}")

    assert does_pass, f"XTTS text embedding PCC below {pcc}: {pcc_message}"
