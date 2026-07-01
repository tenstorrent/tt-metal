# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder-layer decode-mode PCC vs HuggingFace Qwen2 layer 0 (Devstral-style).

Feeds random hidden states one token at a time (batch 1) while advancing KV-cache
position from an empty cache. Runs ``DECODE_GENERATION_LENGTH`` decode steps (10) at
positions 0 … 9 and asserts PCC ≥ 0.99 on every step.

Unlike ``test_lm_decode_pcc.py``, this does **not** run a prefill or the full 28-layer LM —
it isolates layer 0 so decode SDPA can be validated at arbitrary cache depths without
re-running prefill PCC.

Run: ``pytest …/test_decoder_layer_pcc.py -v -s``
"""

import pytest

from models.experimental.vibevoice.tests.pcc.decoder_layer_pcc_common import (
    run_decoder_layer_decode_pcc_sweep,
)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_decoder_layer_decode_pcc(mesh_device, vv_config, lm_state):
    """10 decode steps at positions 0–9, batch 1, empty KV cache, random hiddens."""
    run_decoder_layer_decode_pcc_sweep(mesh_device, lm_state, vv_config)
    print("PASS")
