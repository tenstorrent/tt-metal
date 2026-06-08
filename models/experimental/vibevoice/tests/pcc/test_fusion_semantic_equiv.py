# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Proof that the fusion's semantic branch matches HuggingFace.

The reference produces the per-step semantic feature with *streaming* tokenizers
(a per-layer cache gives every chunk its causal history).  The TT generator
instead decodes/encodes the whole *cumulative* latent history and keeps the last
frame.  This test runs BOTH on identical latents (CPU, fp32 reference modules) and
asserts the resulting per-frame semantic features are the same — i.e. the
cumulative approach is equivalent to HF's streaming cache.

Together with test_connector_pcc.py (which validates fc1->RMSNorm->fc2 at PCC>=0.99)
and the trivial elementwise add, this proves the TT fusion block matches HF.

CPU-only (no device): the equivalence is a property of the causal tokenizers, not
of TTNN, so it is independent of bf16 precision.
"""

import sys
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
    fold_weight_norm,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(not Path(MODEL_PATH).is_dir(), reason="VIBEVOICE_MODEL_PATH weights missing")

N_FRAMES = 6  # number of diffusion latent frames to simulate


def _build_reference_tokenizers(vv_config, ac_state, sem_state):
    from vibevoice.modular.configuration_vibevoice import (
        VibeVoiceAcousticTokenizerConfig,
        VibeVoiceSemanticTokenizerConfig,
    )
    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceAcousticTokenizerModel,
        VibeVoiceSemanticTokenizerModel,
    )

    ac = vv_config.acoustic_tokenizer
    ac_model = VibeVoiceAcousticTokenizerModel(
        VibeVoiceAcousticTokenizerConfig(
            channels=1,
            causal=ac.causal,
            vae_dim=ac.vae_dim,
            encoder_n_filters=ac.encoder_n_filters,
            encoder_ratios=ac.encoder_ratios,
            encoder_depths=ac.encoder_depths,
            decoder_ratios=ac.decoder_ratios,
            decoder_n_filters=ac.decoder_n_filters,
            layernorm=ac.layernorm,
            layernorm_eps=ac.layernorm_eps,
            conv_bias=ac.conv_bias,
        )
    )
    ac_model.load_state_dict(ac_state, strict=False)
    ac_model.eval()

    sem = vv_config.semantic_tokenizer
    sem_model = VibeVoiceSemanticTokenizerModel(
        VibeVoiceSemanticTokenizerConfig(
            channels=1,
            causal=sem.causal,
            vae_dim=sem.vae_dim,
            encoder_n_filters=sem.encoder_n_filters,
            encoder_ratios=sem.encoder_ratios,
            encoder_depths=sem.encoder_depths,
            layernorm=sem.layernorm,
            layernorm_eps=sem.layernorm_eps,
            conv_bias=sem.conv_bias,
            mixer_layer=sem.mixer_layer,
        )
    )
    sem_model.load_state_dict(sem_state, strict=False)
    sem_model.eval()
    return ac_model, sem_model


def test_fusion_semantic_branch_matches_hf_streaming():
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

    vv_config = load_vibevoice_model_config(MODEL_PATH)
    sd = load_vibevoice_state_dict(MODEL_PATH)
    sub = split_submodule_weights(sd)
    ac_state = fold_weight_norm(sub["acoustic_tokenizer"])
    sem_state = fold_weight_norm(sub["semantic_tokenizer"])

    ac_model, sem_model = _build_reference_tokenizers(vv_config, ac_state, sem_state)

    vae_dim = vv_config.acoustic_tokenizer.vae_dim
    torch.manual_seed(0)
    # Simulated diffusion latents, channels-first [1, vae_dim, N_FRAMES].
    latents = torch.randn(1, vae_dim, N_FRAMES, dtype=torch.float32)
    sample_indices = torch.tensor([0])

    # ── HuggingFace path: streaming caches, frame by frame ──────────────────
    ac_cache = VibeVoiceTokenizerStreamingCache()
    sem_cache = VibeVoiceTokenizerStreamingCache()
    stream_feats = []
    with torch.no_grad():
        for n in range(N_FRAMES):
            lat_n = latents[:, :, n : n + 1]  # [1, vae_dim, 1]
            audio_n = ac_model.decode(lat_n, cache=ac_cache, sample_indices=sample_indices, use_cache=True)
            sem_n = sem_model.encode(audio_n, cache=sem_cache, sample_indices=sample_indices, use_cache=True).mean
            stream_feats.append(sem_n[:, -1, :])  # last (new) frame, [1, sem_dim]
    stream_feats = torch.cat(stream_feats, dim=0)  # [N, sem_dim]

    # ── TT path: cumulative full-context, slice last frame (no cache) ───────
    cumulative_feats = []
    with torch.no_grad():
        for n in range(N_FRAMES):
            lat_prefix = latents[:, :, : n + 1]  # [1, vae_dim, n+1]
            audio = ac_model.decode(lat_prefix)  # non-streaming, full causal context
            sem = sem_model.encode(audio).mean  # [1, n+1, sem_dim]
            cumulative_feats.append(sem[:, -1, :])  # last frame
    cumulative_feats = torch.cat(cumulative_feats, dim=0)  # [N, sem_dim]

    passed, pcc_val = comp_pcc(stream_feats, cumulative_feats, pcc=0.999)
    print(f"[fusion semantic] streaming-vs-cumulative PCC = {pcc_val:.6f}")
    assert passed, (
        f"Cumulative semantic features differ from HF streaming (PCC {pcc_val:.6f} < 0.999) — "
        "the TT fusion semantic branch would NOT match HF."
    )
