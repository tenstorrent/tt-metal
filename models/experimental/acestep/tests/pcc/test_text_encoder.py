# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: TT AceStepTextEncoder vs the genuine Qwen3-Embedding-0.6B text encoder (real weights).

The text encoder is the prompt front-end: tokenized text -> text_hidden_states [1, L, 1024] via a
28-layer CAUSAL Qwen3 model. Validated against `text_encoder(input_ids).last_hidden_state`.

Reuses AceStepEncoderLayer + RMSNorm1D (causal via an additive mask). Skipped if the pipeline
bundle (which contains Qwen3-Embedding-0.6B) isn't downloaded.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.common.utility_functions import comp_pcc
from models.experimental.acestep.reference.weight_utils import have_pipeline
from models.experimental.acestep.tt.model_config import build_text_encoder
from models.experimental.acestep.tests.test_utils import require_single_device, to_torch

HEAD_DIM = 128
HIDDEN = 1024

# Prompt-length token sequences (tile-friendly).
SEQ_LENS = [32, 128]


@pytest.mark.slow
@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline bundle (Qwen3 text encoder) not downloaded")
@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_text_encoder(device, seq_len):
    require_single_device(device)

    tt_te, hf_te = build_text_encoder(device)

    torch.manual_seed(seq_len)
    input_ids = torch.randint(0, hf_te.config.vocab_size, (1, seq_len))

    with torch.no_grad():
        ref = hf_te(input_ids=input_ids).last_hidden_state  # [1, L, 1024]

    # RoPE tables (theta=1e6, head_dim=128) via the reference rotary embedding.
    rope = Qwen3RotaryEmbedding(hf_te.config)
    pos = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, seq_len, HEAD_DIM), pos)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt_te.forward(input_ids, cos_tt, sin_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, HIDDEN)).reshape(1, seq_len, HIDDEN)

    passing, msg = comp_pcc(ref, out, 0.97)
    print(f"TEXT_ENCODER_PCC seq={seq_len}: {msg}")
    assert passing, f"text encoder PCC {msg} < 0.97"
