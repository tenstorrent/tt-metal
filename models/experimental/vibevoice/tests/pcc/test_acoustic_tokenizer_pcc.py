# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 — Acoustic Tokenizer PCC tests.

Three tests:
  1. test_acoustic_tokenizer_encode_pcc: encode PCC >= 0.99
  2. test_acoustic_tokenizer_decode_pcc: decode scaled-random latents PCC >= 0.99
  3. test_acoustic_tokenizer_decode_real_latents_pcc: decode encoder latents PCC >= 0.99
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
from models.experimental.vibevoice.tt.ttnn_acoustic_tokenizer import (
    preprocess_acoustic_tokenizer_weights,
    TTAcousticTokenizer,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

AUDIO_LEN = 24000


@pytest.fixture(scope="module")
def vv_config():
    return load_vibevoice_model_config(MODEL_PATH)


@pytest.fixture(scope="module")
def ac_tok_state():
    sd = load_vibevoice_state_dict(MODEL_PATH)
    sub = split_submodule_weights(sd)
    return fold_weight_norm(sub["acoustic_tokenizer"])


@pytest.fixture(scope="function")
def ac_tokenizer_tt(mesh_device, ac_tok_state, vv_config):
    cfg = vv_config.acoustic_tokenizer
    weights = preprocess_acoustic_tokenizer_weights(ac_tok_state, mesh_device, cfg)
    return TTAcousticTokenizer(weights, mesh_device)


def _reference_acoustic_encode(hf_state, audio, vv_config):
    from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel

    cfg = vv_config.acoustic_tokenizer
    tok_cfg = VibeVoiceAcousticTokenizerConfig(
        channels=1,
        causal=cfg.causal,
        vae_dim=cfg.vae_dim,
        encoder_n_filters=cfg.encoder_n_filters,
        encoder_ratios=cfg.encoder_ratios,
        encoder_depths=cfg.encoder_depths,
        decoder_ratios=cfg.decoder_ratios,
        decoder_n_filters=cfg.decoder_n_filters,
        layernorm=cfg.layernorm,
        layernorm_eps=cfg.layernorm_eps,
        conv_bias=cfg.conv_bias,
    )
    model = VibeVoiceAcousticTokenizerModel(tok_cfg)
    model.load_state_dict(hf_state, strict=False)
    model.eval()
    with torch.no_grad():
        out = model.encoder(audio)
    return out  # [1, vae_dim, T_enc]


def _reference_acoustic_decode(hf_state, latents, vv_config):
    from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel

    cfg = vv_config.acoustic_tokenizer
    tok_cfg = VibeVoiceAcousticTokenizerConfig(
        channels=1,
        causal=cfg.causal,
        vae_dim=cfg.vae_dim,
        encoder_n_filters=cfg.encoder_n_filters,
        encoder_ratios=cfg.encoder_ratios,
        encoder_depths=cfg.encoder_depths,
        decoder_ratios=cfg.decoder_ratios,
        decoder_n_filters=cfg.decoder_n_filters,
        layernorm=cfg.layernorm,
        layernorm_eps=cfg.layernorm_eps,
        conv_bias=cfg.conv_bias,
    )
    model = VibeVoiceAcousticTokenizerModel(tok_cfg)
    model.load_state_dict(hf_state, strict=False)
    model.eval()
    with torch.no_grad():
        out = model.decoder(latents)
    return out  # [1, 1, T_audio]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_acoustic_tokenizer_encode_pcc(mesh_device, ac_tok_state, vv_config, ac_tokenizer_tt):
    torch.manual_seed(0)

    audio = torch.randn(1, 1, AUDIO_LEN, dtype=torch.bfloat16)

    # 1) Reference
    ref_enc = _reference_acoustic_encode(ac_tok_state, audio.float(), vv_config)
    # [1, vae_dim, T_enc]

    # 2) TT
    audio_4d = audio.unsqueeze(0)  # [1, 1, 1, T]
    audio_tt = ttnn.as_tensor(
        audio_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_enc = ac_tokenizer_tt.encode(audio_tt)  # [1, 1, T_enc, vae_dim]
    tt_enc_torch = ttnn.to_torch(tt_enc).to(torch.float32).squeeze(1)  # [1, T_enc, vae_dim]

    ref_compare = ref_enc.to(torch.float32).permute(0, 2, 1)  # [1, T_enc, vae_dim]
    T_min = min(ref_compare.shape[1], tt_enc_torch.shape[1])

    passed, pcc_val = comp_pcc(ref_compare[:, :T_min], tt_enc_torch[:, :T_min], pcc=0.99)
    assert passed, f"Acoustic tokenizer encode PCC {pcc_val:.6f} < 0.99"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_acoustic_tokenizer_decode_pcc(mesh_device, ac_tok_state, vv_config, ac_tokenizer_tt):
    torch.manual_seed(1)

    VAE_DIM = vv_config.acoustic_tokenizer.vae_dim
    T_ENC = 32  # short latent chunk

    # Plain N(0,1) latents decode to near-silence, so bf16 noise floors PCC at ~0.99.
    # Scale to ~encoder latent std (~23–24) so magnitude matches the decoder's normal
    # operating range; that clears 0.99 without needing a real encode step.
    LATENT_STD = 24.0
    latents = (torch.randn(1, VAE_DIM, T_ENC) * LATENT_STD).to(torch.bfloat16)

    # 1) Reference
    ref_dec = _reference_acoustic_decode(ac_tok_state, latents.float(), vv_config)
    # [1, 1, T_audio]

    # 2) TT: latents as [1, 1, T_enc, vae_dim]
    lat_4d = latents.permute(0, 2, 1).unsqueeze(1)  # [1, 1, T_enc, vae_dim]
    lat_tt = ttnn.as_tensor(
        lat_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_dec = ac_tokenizer_tt.decode(lat_tt)  # [1, 1, 1, T_audio]
    tt_dec_torch = ttnn.to_torch(tt_dec).to(torch.float32).squeeze()  # [T_audio]

    ref_compare = ref_dec.to(torch.float32).squeeze()  # [T_audio]
    T_min = min(ref_compare.shape[-1], tt_dec_torch.shape[-1])

    passed, pcc_val = comp_pcc(ref_compare[:T_min], tt_dec_torch[:T_min], pcc=0.99)
    assert passed, f"Acoustic tokenizer decode PCC {pcc_val:.6f} < 0.99"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_acoustic_tokenizer_decode_real_latents_pcc(mesh_device, ac_tok_state, vv_config, ac_tokenizer_tt):
    """Decode PCC on *real* encoder latents instead of scaled random noise.

    Same decoder workload as test_acoustic_tokenizer_decode_pcc (T_enc=32) but the
    latents come from the fp32 reference encoder on audio.  Complements the
    scaled-random decode test with structured encoder-distribution inputs.
    """
    torch.manual_seed(0)

    # 102400 = 3200 (product of encoder ratios 8*5*5*4*2*2) * 32 -> T_enc == 32,
    # matching the random-latent test's decoder workload exactly.
    AUDIO_LEN_RT = 102400
    audio = torch.randn(1, 1, AUDIO_LEN_RT, dtype=torch.bfloat16)

    # Real latents from the fp32 reference encoder: [1, vae_dim, T_enc]
    ref_latents = _reference_acoustic_encode(ac_tok_state, audio.float(), vv_config)

    # Feed the SAME latents to both decoders (isolates decoder precision on real input)
    ref_dec = _reference_acoustic_decode(ac_tok_state, ref_latents.float(), vv_config)  # [1, 1, T_audio]

    lat_4d = ref_latents.to(torch.bfloat16).permute(0, 2, 1).unsqueeze(1)  # [1, 1, T_enc, vae_dim]
    lat_tt = ttnn.as_tensor(
        lat_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_dec = ac_tokenizer_tt.decode(lat_tt)  # [1, 1, 1, T_audio]
    tt_dec_torch = ttnn.to_torch(tt_dec).to(torch.float32).squeeze()  # [T_audio]

    ref_compare = ref_dec.to(torch.float32).squeeze()  # [T_audio]
    T_min = min(ref_compare.shape[-1], tt_dec_torch.shape[-1])

    passed, pcc_val = comp_pcc(ref_compare[:T_min], tt_dec_torch[:T_min], pcc=0.99)
    print(f"[decode real-latents] PCC = {pcc_val:.6f}")
    assert passed, f"Acoustic tokenizer decode (real latents) PCC {pcc_val:.6f} < 0.99"
