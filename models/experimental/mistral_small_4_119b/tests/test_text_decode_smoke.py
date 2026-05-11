# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill + decode smoke test for Mistral-Small-4 text model.

Runs a short prefill then steps through N decode iterations.
Checks: finite logits, correct shape at every step.

Run manually::

    export MISTRAL4_DECODE_SMOKE=1
    export MISTRAL4_DECODE_N_LAYERS=2        # optional; default 2
    export MISTRAL4_DECODE_PREFILL_LEN=4     # optional; default 4
    export MISTRAL4_DECODE_N_STEPS=3         # optional; default 3
    export MESH_DEVICE=P150x4                # optional
    pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_smoke.py -v -s --timeout=0
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_VOCAB_SIZE,
    HF_MODEL_ID,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")

_N_LAYERS = int(os.environ.get("MISTRAL4_DECODE_N_LAYERS", "2"))
_PREFILL_LEN = int(os.environ.get("MISTRAL4_DECODE_PREFILL_LEN", "4"))
_N_STEPS = int(os.environ.get("MISTRAL4_DECODE_N_STEPS", "3"))
_MAX_SEQ_LEN = _PREFILL_LEN + _N_STEPS + 64  # small cache for smoke


def _state_dict_prefixes(n_layers: int) -> tuple:
    p = ["language_model.model.embed_tokens."]
    for i in range(n_layers):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    return tuple(p)


def _mesh_params():
    shape = mesh_device_request_param()
    base = {"trace_region_size": 30000000, "num_command_queues": 1}
    fabric = ttnn.FabricConfig.DISABLED if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D
    return [pytest.param(shape, {**base, "fabric_config": fabric}, id=f"mesh{shape[0]}x{shape[1]}")]


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_DECODE_SMOKE") != "1",
    reason="Set MISTRAL4_DECODE_SMOKE=1 to run the prefill+decode smoke test.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_text_decode_smoke(reset_seeds, mesh_device):
    """Prefill N tokens, greedily decode M steps, check shapes and finiteness."""
    from transformers import AutoConfig
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    text = cfg.text_config
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(text, attr):
            setattr(text, attr, "eager")

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_N_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    vocab = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY].shape[0]
    assert vocab == EXPECTED_VOCAB_SIZE

    embed_w = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY].to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)

    # ── Build model ──────────────────────────────────────────────────────
    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers, max_seq={_MAX_SEQ_LEN})...")
    try:
        model = TtMistral4TextModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            text_config=text,
            num_decoder_layers=_N_LAYERS,
            max_seq_len=_MAX_SEQ_LEN,
        )
    except Exception as exc:
        pytest.fail(f"TtMistral4TextModel init failed: {exc}")

    # ── Prefill ──────────────────────────────────────────────────────────
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, _PREFILL_LEN), dtype=torch.long)
    position_ids = torch.arange(_PREFILL_LEN, dtype=torch.long).unsqueeze(0)

    hidden0 = F.embedding(input_ids, embed_w).to(torch.bfloat16)
    pos_emb_prefill = rotary(hidden0, position_ids)

    logger.info(f"Running prefill (seq_len={_PREFILL_LEN})...")
    prefill_logits = model.prefill(input_ids, pos_emb_prefill)

    assert prefill_logits.shape == (
        1,
        _PREFILL_LEN,
        vocab,
    ), f"Prefill logits shape {tuple(prefill_logits.shape)} != (1, {_PREFILL_LEN}, {vocab})"
    assert torch.isfinite(prefill_logits.to(torch.float32)).all(), "Non-finite values in prefill logits"
    logger.info(f"Prefill OK: shape={tuple(prefill_logits.shape)}")

    # ── Greedy decode loop ───────────────────────────────────────────────
    next_token = prefill_logits[0, -1, :].argmax().item()
    logger.info(f"First decode token (greedy from prefill): {next_token}")

    for step in range(_N_STEPS):
        current_pos = _PREFILL_LEN + step
        tok_tensor = torch.tensor([[next_token]], dtype=torch.long)
        dec_pos_ids = torch.tensor([[current_pos]], dtype=torch.long)

        dec_hidden = F.embedding(tok_tensor, embed_w).to(torch.bfloat16)
        pos_emb_decode = rotary(dec_hidden, dec_pos_ids)

        decode_logits = model.decode(tok_tensor, pos_emb_decode, current_pos)

        assert decode_logits.shape == (
            1,
            1,
            vocab,
        ), f"Step {step}: decode logits shape {tuple(decode_logits.shape)} != (1, 1, {vocab})"
        assert torch.isfinite(decode_logits.to(torch.float32)).all(), f"Step {step}: non-finite values in decode logits"

        next_token = decode_logits[0, 0, :].argmax().item()
        logger.info(f"Decode step {step} (pos={current_pos}): OK, next token={next_token}")

    logger.info(f"{_N_LAYERS}-layer prefill({_PREFILL_LEN})+decode({_N_STEPS}) smoke: PASSED")
