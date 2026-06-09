# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 — Semantic Tokenizer PCC test.

Loads real semantic_tokenizer weights, runs reference PyTorch encoder
and TT encoder on a fixed audio segment (24000 samples), asserts PCC >= 0.99.
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
    fold_weight_norm,
)
from models.experimental.vibevoice.tt.ttnn_semantic_tokenizer import (
    preprocess_semantic_tokenizer_weights,
    TTSemanticTokenizer,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

AUDIO_LEN = 24000  # 1 second at 24kHz


@pytest.fixture(scope="module")
def vv_config():
    return load_vibevoice_model_config(MODEL_PATH)


@pytest.fixture(scope="module")
def sem_tok_state():
    sd = load_vibevoice_state_dict(MODEL_PATH)
    sub = split_submodule_weights(sd)
    return fold_weight_norm(sub["semantic_tokenizer"])


def _reference_semantic_encode(hf_state: dict, audio: torch.Tensor, vv_config) -> torch.Tensor:
    """Run reference semantic tokenizer encode."""
    from vibevoice.modular.configuration_vibevoice import VibeVoiceSemanticTokenizerConfig
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceSemanticTokenizerModel

    cfg = vv_config.semantic_tokenizer
    tok_cfg = VibeVoiceSemanticTokenizerConfig(
        channels=1,
        causal=cfg.causal,
        vae_dim=cfg.vae_dim,
        encoder_n_filters=cfg.encoder_n_filters,
        encoder_ratios=cfg.encoder_ratios,
        encoder_depths=cfg.encoder_depths,
        layernorm=cfg.layernorm,
        layernorm_eps=cfg.layernorm_eps,
        conv_bias=cfg.conv_bias,
        mixer_layer=cfg.mixer_layer,
    )
    model = VibeVoiceSemanticTokenizerModel(tok_cfg)
    model.load_state_dict(hf_state, strict=False)
    model.eval()
    with torch.no_grad():
        # audio: [1, 1, T] (BCT)
        out = model.encoder(audio)  # [1, vae_dim, T_enc] BCT
    return out  # [1, vae_dim, T_enc]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_semantic_tokenizer_pcc(mesh_device, vv_config, sem_tok_state):
    torch.manual_seed(0)

    # Fixed audio segment
    audio = torch.randn(1, 1, AUDIO_LEN, dtype=torch.bfloat16)  # [B=1, C=1, T]

    # 1) Reference
    ref_out = _reference_semantic_encode(sem_tok_state, audio.float(), vv_config)
    # ref_out: [1, vae_dim, T_enc] BCT format

    # 2) TT forward
    cfg = vv_config.semantic_tokenizer
    weights = preprocess_semantic_tokenizer_weights(sem_tok_state, mesh_device, cfg)
    tokenizer_tt = TTSemanticTokenizer(weights, mesh_device)

    # Input: [1, 1, 1, T] for TT (B, 1, H=1, T)
    audio_4d = audio.unsqueeze(0)  # [1, 1, 1, T]
    audio_tt = ttnn.as_tensor(
        audio_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = tokenizer_tt(audio_tt)  # [1, 1, T_enc, vae_dim]
    tt_out_torch = ttnn.to_torch(tt_out).to(torch.float32).squeeze(1)  # [1, T_enc, vae_dim]

    # Compare: ref [1, vae_dim, T_enc] vs tt [1, T_enc, vae_dim]
    ref_compare = ref_out.to(torch.float32).permute(0, 2, 1)  # [1, T_enc, vae_dim]
    T_min = min(ref_compare.shape[1], tt_out_torch.shape[1])

    passed, pcc_val = comp_pcc(
        ref_compare[:, :T_min, :],
        tt_out_torch[:, :T_min, :],
        pcc=0.99,
    )
    assert passed, f"Semantic tokenizer encoder PCC {pcc_val:.6f} < 0.99"
