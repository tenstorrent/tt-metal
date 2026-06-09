# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 — Language Model PCC test.

Loads real LM weights, runs reference Qwen2 forward and TT forward for a
short sequence (S=32), asserts PCC >= 0.99 on last_hidden_state.
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
    remap_lm_keys_to_tt_transformers,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import (
    preprocess_lm_weights,
    TTVibeVoiceLM,
    create_kv_cache,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEQ_LEN = 32


@pytest.fixture(scope="module")
def vv_config():
    return load_vibevoice_model_config(MODEL_PATH)


@pytest.fixture(scope="module")
def lm_state():
    full_sd = load_vibevoice_state_dict(MODEL_PATH)
    sub = split_submodule_weights(full_sd)
    return remap_lm_keys_to_tt_transformers(sub["lm"])


def _reference_lm_forward(lm_state: dict, input_ids: torch.Tensor, vv_config) -> torch.Tensor:
    """Run reference Qwen2 forward using transformers."""
    from transformers import Qwen2Model, Qwen2Config

    cfg_dec = vv_config.decoder
    hf_cfg = Qwen2Config(
        hidden_size=cfg_dec.hidden_size,
        num_hidden_layers=cfg_dec.num_hidden_layers,
        num_attention_heads=cfg_dec.num_attention_heads,
        num_key_value_heads=cfg_dec.num_key_value_heads,
        intermediate_size=cfg_dec.intermediate_size,
        vocab_size=cfg_dec.vocab_size,
        rope_theta=cfg_dec.rope_theta,
        rms_norm_eps=cfg_dec.rms_norm_eps,
        max_position_embeddings=cfg_dec.max_position_embeddings,
    )
    model = Qwen2Model(hf_cfg)

    # Remap our tt-style keys back to HF keys for reference loading
    hf_state = {}
    for k, v in lm_state.items():
        hf_k = k
        hf_k = hf_k.replace("tok_embeddings.", "embed_tokens.")
        hf_k = hf_k.replace(".attention.wq", ".self_attn.q_proj")
        hf_k = hf_k.replace(".attention.wk", ".self_attn.k_proj")
        hf_k = hf_k.replace(".attention.wv", ".self_attn.v_proj")
        hf_k = hf_k.replace(".attention.wo", ".self_attn.o_proj")
        hf_k = hf_k.replace(".feed_forward.w1", ".mlp.gate_proj")
        hf_k = hf_k.replace(".feed_forward.w3", ".mlp.up_proj")
        hf_k = hf_k.replace(".feed_forward.w2", ".mlp.down_proj")
        hf_k = hf_k.replace(".attention_norm", ".input_layernorm")
        hf_k = hf_k.replace(".ffn_norm", ".post_attention_layernorm")
        hf_k = hf_k.replace("norm.weight", "norm.weight")
        hf_state[hf_k] = v

    model.load_state_dict(hf_state, strict=False)
    model.eval()
    with torch.no_grad():
        out = model(input_ids)
    return out.last_hidden_state  # [B, S, hidden]


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_hidden_state_pcc(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder

    # Short sequence
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), dtype=torch.long)

    # 1) Reference
    ref_hidden = _reference_lm_forward(lm_state, input_ids, vv_config)  # [1, S, hidden]

    # 2) TT forward
    weights = preprocess_lm_weights(lm_state, mesh_device, cfg)
    lm_tt = TTVibeVoiceLM(weights, mesh_device)

    kv_cache = create_kv_cache(cfg.num_hidden_layers)
    _, tt_hidden = lm_tt.prefill(input_ids, kv_cache=kv_cache, return_last_hidden=True)

    tt_hidden_torch = ttnn.to_torch(tt_hidden).to(torch.float32).squeeze(1)  # [1, S, hidden]

    passed, pcc_val = comp_pcc(ref_hidden.to(torch.float32), tt_hidden_torch, pcc=0.99)
    assert passed, f"LM last_hidden_state PCC {pcc_val:.6f} < 0.99"
