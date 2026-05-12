# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Seamless M4T v2 — TTNN text-to-text demo (Tenstorrent device).

Mirrors ``torch_demo.py`` intent: same English prompt and Hindi target language, but runs
``TTSeamlessM4Tv2Model.generate`` with ``generate_speech=False`` and prints decoded token ids.

Run from repo root (requires local checkpoint — see ``scripts/download_weights.py``):

  python models/experimental/seamless_m4t_v2_large/demo/ttnn_demo.py

Optional: ``SEAMLESS_M4T_V2_WEIGHTS=/path/to/seamless-m4t-v2-large`` if not using the default tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)


def _weights_dir() -> Path:
    env = os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return ensure_seamless_m4t_v2_large_weights()


def torch_ids_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def make_tt_model(device: ttnn.Device, model: torch.nn.Module, cfg, t2u_cfg) -> TTSeamlessM4Tv2Model:
    params = create_seamless_m4t_v2_model_parameters(model, device=device)
    return TTSeamlessM4Tv2Model(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        pad_token_id=cfg.pad_token_id,
        decoder_start_token_id=cfg.decoder_start_token_id,
        vocab_size=cfg.vocab_size,
        adaptor_kernel_size=cfg.adaptor_kernel_size,
        adaptor_stride=cfg.adaptor_stride,
        t2u_eos_token_id=cfg.t2u_eos_token_id,
        t2u_pad_token_id=t2u_cfg.pad_token_id,
        vocoder_offset=cfg.vocoder_offset,
        t2u_layer_norm_eps=t2u_cfg.layer_norm_eps,
        t2u_encoder_layers=t2u_cfg.encoder_layers,
        t2u_encoder_attention_heads=t2u_cfg.encoder_attention_heads,
        t2u_decoder_layers=t2u_cfg.decoder_layers,
        t2u_decoder_attention_heads=t2u_cfg.decoder_attention_heads,
        variance_predictor_embed_dim=t2u_cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=t2u_cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=t2u_cfg.variance_predictor_kernel_size,
        vocoder_config=cfg,
        generation_config=model.generation_config,
        hf_config=cfg,
    )


def main() -> None:
    weights_dir = _weights_dir()
    path = os.fspath(weights_dir)

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    text_inputs = processor(text="Hello, my dog is cute", src_lang="eng", return_tensors="pt")
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs["attention_mask"]

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config

    gen_kw = dict(
        generate_speech=False,
        max_new_tokens=48,
        do_sample=False,
        num_beams=1,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=cfg.eos_token_id,
        tgt_lang="hin",
    )

    original_default = ttnn.GetDefaultDevice()
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(device)
    try:
        tt_model = make_tt_model(device, model, cfg, t2u_cfg)
        tt_out = tt_model.generate(
            input_ids=torch_ids_to_ttnn(device, input_ids),
            attention_mask=torch_ids_to_ttnn(device, attention_mask),
            **gen_kw,
        )
        if not isinstance(tt_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"Expected TTSeamlessM4Tv2GreedySearchOutput, got {type(tt_out)}")

        seq_cpu = ttnn.to_torch(ttnn.from_device(tt_out.sequences)).to(torch.int64).cpu()
        decoded = tokenizer.batch_decode(seq_cpu, skip_special_tokens=False)[0]

        print("ok")
        print("  input_ids shape:", tuple(input_ids.shape))
        print("  generated shape:", tuple(seq_cpu.shape))
        print("  decoded:", decoded)
    finally:
        ttnn.SetDefaultDevice(original_default)
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
