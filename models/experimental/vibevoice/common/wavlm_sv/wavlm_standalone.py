# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch-only loader for the WavLM-large speaker-verification model used by the VibeVoice
technical report for Speaker Similarity (SIM) — the UniSpeech ``wavlm_large_finetune.pth``
(WavLM-large backbone + ECAPA-TDNN x-vector head), loaded without fairseq/s3prl.

The vendored ``WavLM.py``/``modules.py`` are from microsoft/unilm (MIT); ``models/ecapa_tdnn.py``
is from microsoft/UniSpeech (lightly patched: optional torchaudio/fairseq imports + a torch-only
WavLM feature extractor). See README.md. The checkpoint is fetched from a HF mirror; the canonical
source is microsoft/UniSpeech ``downstreams/speaker_verification`` (wavlm_large_finetune.pth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .WavLM import WavLM, WavLMConfig

CKPT_REPO = "subatomicseer/wavlm-large-sv-ckpts"
CKPT_FILE = "wavlm_large_finetune.pth"

# WavLM-large architecture (validated: 488/488 backbone keys match the finetune checkpoint).
WAVLM_LARGE_CFG = {
    "extractor_mode": "layer_norm",
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "activation_fn": "gelu",
    "layer_norm_first": True,
    "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    "conv_bias": False,
    "normalize": True,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
}


class StandaloneWavLM(nn.Module):
    """Drop-in for the s3prl upstream: holds ``.model`` (WavLM) so checkpoint keys
    ``feature_extract.model.*`` load, and returns ``{"hidden_states": (25 tensors)}``
    reproducing s3prl's ordering — input-to-each-layer plus the final-LN encoder output —
    plus the s3prl input-waveform normalization, so the trained ``feature_weight`` applies
    to the exact representations it was trained on."""

    def __init__(self, cfg_dict=None):
        super().__init__()
        cfg_dict = cfg_dict or WAVLM_LARGE_CFG
        self.model = WavLM(WavLMConfig(cfg_dict))
        self._n_layers = cfg_dict["encoder_layers"]
        self._normalize = cfg_dict.get("normalize", False)

    def forward(self, wavs):
        wav = torch.stack([w.reshape(-1) for w in wavs], dim=0)
        if self._normalize:  # s3prl UpstreamExpert layer-norms the input waveform
            wav = torch.stack([F.layer_norm(w, w.shape) for w in wav], dim=0)
        # output_layer=n_layers → tgt_layer=n_layers-1 → layer_results = [pre, out0..out_{L-1}] (L+1=25)
        (_, layer_results), _pad = self.model.extract_features(wav, output_layer=self._n_layers, ret_layer_results=True)
        hs = [lr[0].transpose(0, 1) for lr in layer_results]  # each (B,T,C); raw residual-stream
        hs[-1] = self.model.encoder.layer_norm(hs[-1])  # s3prl encoder output carries the final LN
        return {"hidden_states": tuple(hs), "default": hs[-1]}


def _resolve_checkpoint(checkpoint_path=None):
    if checkpoint_path is not None:
        return checkpoint_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(CKPT_REPO, CKPT_FILE)


def init_model(checkpoint_path=None, device="cpu"):
    """Build the ECAPA-TDNN(WavLM-large) SV model and load the fine-tuned weights."""
    from .models.ecapa_tdnn import ECAPA_TDNN_SMALL

    model = ECAPA_TDNN_SMALL(feat_dim=1024, config_path="__standalone__")
    sd = torch.load(_resolve_checkpoint(checkpoint_path), map_location="cpu")["model"]
    # Non-strict: the checkpoint carries a training-only loss head (loss_calculator.*) we don't need.
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def embed(model, wav_16k, device="cpu"):
    """L2-normalized speaker embedding for a 1-D 16 kHz waveform."""
    wav = wav_16k.to(torch.float32).to(device).reshape(1, -1)
    return F.normalize(model(wav), dim=-1)
