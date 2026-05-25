# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS reference implementation wrapper.
Loads the HuggingFace model and provides utilities to dump intermediate tensors
for PCC comparison against the TT implementation.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Qwen3TTSConfig:
    """Mirror of the HuggingFace Qwen3-TTS config for the 1.7B model."""

    # Talker
    talker_hidden_size: int = 2048
    talker_num_layers: int = 28
    talker_num_heads: int = 16
    talker_num_kv_heads: int = 8
    talker_head_dim: int = 128
    talker_intermediate_size: int = 6144
    talker_vocab_size: int = 3072
    talker_text_vocab_size: int = 151936
    talker_text_hidden_size: int = 2048
    talker_num_code_groups: int = 16
    talker_rope_theta: float = 1000000.0
    talker_mrope_sections: list = field(default_factory=lambda: [24, 20, 20])

    # Code Predictor
    cp_hidden_size: int = 1024
    cp_num_layers: int = 5
    cp_num_heads: int = 16
    cp_num_kv_heads: int = 8
    cp_head_dim: int = 128
    cp_intermediate_size: int = 3072
    cp_vocab_size: int = 2048
    cp_num_code_groups: int = 16
    cp_rope_theta: float = 1000000.0

    # Speaker Encoder
    spk_enc_dim: int = 2048
    spk_mel_dim: int = 128
    spk_sample_rate: int = 24000
    spk_channels: list = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    spk_kernel_sizes: list = field(default_factory=lambda: [5, 3, 3, 3, 1])
    spk_dilations: list = field(default_factory=lambda: [1, 2, 3, 4, 1])

    # Special token IDs
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    tts_pad_token_id: int = 151671
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_pad_id: int = 2148
    codec_language_ids: dict = field(
        default_factory=lambda: {
            "chinese": 2055,
            "english": 2050,
            "german": 2053,
            "italian": 2070,
            "portuguese": 2071,
            "spanish": 2054,
            "japanese": 2058,
            "korean": 2064,
            "french": 2061,
            "russian": 2069,
        }
    )

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3TTSConfig":
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return cls()
        with open(config_path) as f:
            raw = json.load(f)
        tc = raw.get("talker_config", {})
        cp = tc.get("code_predictor_config", {})
        se = raw.get("speaker_encoder_config", {})
        return cls(
            talker_hidden_size=tc.get("hidden_size", 2048),
            talker_num_layers=tc.get("num_hidden_layers", 28),
            talker_num_heads=tc.get("num_attention_heads", 16),
            talker_num_kv_heads=tc.get("num_key_value_heads", 8),
            talker_head_dim=tc.get("head_dim", 128),
            talker_intermediate_size=tc.get("intermediate_size", 6144),
            talker_vocab_size=tc.get("vocab_size", 3072),
            talker_text_vocab_size=tc.get("text_vocab_size", 151936),
            talker_text_hidden_size=tc.get("text_hidden_size", 2048),
            talker_num_code_groups=tc.get("num_code_groups", 16),
            talker_rope_theta=tc.get("rope_theta", 1000000.0),
            talker_mrope_sections=tc.get("rope_scaling", {}).get("mrope_section", [24, 20, 20]),
            cp_hidden_size=cp.get("hidden_size", 1024),
            cp_num_layers=cp.get("num_hidden_layers", 5),
            cp_num_heads=cp.get("num_attention_heads", 16),
            cp_num_kv_heads=cp.get("num_key_value_heads", 8),
            cp_head_dim=cp.get("head_dim", 128),
            cp_intermediate_size=cp.get("intermediate_size", 3072),
            cp_vocab_size=cp.get("vocab_size", 2048),
            cp_num_code_groups=cp.get("num_code_groups", 16),
            cp_rope_theta=cp.get("rope_theta", 1000000.0),
            spk_enc_dim=se.get("enc_dim", 2048),
            spk_sample_rate=se.get("sample_rate", 24000),
            codec_language_ids=tc.get("codec_language_id", cls.codec_language_ids),
        )


class ActivationCapture:
    """Hook-based activation capture for PCC comparison."""

    def __init__(self):
        self.activations = {}
        self._hooks = []

    def register(self, module: torch.nn.Module, name: str):
        def hook_fn(mod, inp, out, name=name):
            if isinstance(out, tuple):
                self.activations[name] = out[0].detach().cpu()
            else:
                self.activations[name] = out.detach().cpu()

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for name, tensor in self.activations.items():
            safe_name = name.replace(".", "_")
            torch.save(tensor, os.path.join(output_dir, f"{safe_name}.pt"))

    @staticmethod
    def load(output_dir: str) -> dict:
        activations = {}
        for f in Path(output_dir).glob("*.pt"):
            activations[f.stem] = torch.load(f, map_location="cpu", weights_only=True)
        return activations


def load_reference_model(model_path: str, device: str = "cuda:0", dtype=torch.bfloat16):
    """Load the Qwen3-TTS model from HuggingFace."""
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
    )
    return model


def register_talker_hooks(model, capture: ActivationCapture):
    """Register hooks on all Talker layers for activation capture."""
    talker = model.talker
    capture.register(talker.embed_tokens, "talker.embed_tokens")

    for i, layer in enumerate(talker.layers):
        capture.register(layer.self_attn, f"talker.layers.{i}.self_attn")
        capture.register(layer.mlp, f"talker.layers.{i}.mlp")
        capture.register(layer, f"talker.layers.{i}")

    if hasattr(talker, "norm"):
        capture.register(talker.norm, "talker.norm")


def register_code_predictor_hooks(model, capture: ActivationCapture):
    """Register hooks on Code Predictor layers."""
    cp = model.talker.code_predictor
    for i, layer in enumerate(cp.layers):
        capture.register(layer.self_attn, f"code_predictor.layers.{i}.self_attn")
        capture.register(layer.mlp, f"code_predictor.layers.{i}.mlp")
        capture.register(layer, f"code_predictor.layers.{i}")


def register_all_hooks(model, capture: ActivationCapture):
    """Register hooks on all model components."""
    register_talker_hooks(model, capture)
    register_code_predictor_hooks(model, capture)


def generate_reference(
    model,
    text: str,
    language: str = "japanese",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    capture: Optional[ActivationCapture] = None,
):
    """Run reference inference and optionally capture activations."""
    if capture is not None:
        register_all_hooks(model, capture)

    gen_kwargs = dict(
        text=text,
        language=language,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    if ref_audio is not None:
        wavs, sr = model.generate_voice_clone(
            ref_audio=ref_audio,
            ref_text=ref_text or "",
            **gen_kwargs,
        )
    else:
        wavs, sr = model.generate(
            **gen_kwargs,
        )

    if capture is not None:
        capture.remove_hooks()

    return wavs, sr
