#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture ONE Whisper forward (encoder + one decoder prefill step) under ttnn graph capture, dump
JSON for raw_hazard_analyzer.py. Single Wormhole chip (1x1 mesh).

Whisper is an ENCODER-DECODER speech transformer -- the point is the encoder->decoder CROSS-ATTENTION
dependency structure (decoder attends to the fixed encoder output). RANDOM weights via WhisperModel(config)
(config-only, no checkpoint download; hazard structure is weight-value independent). Smallest sensible
variant (whisper-base). Mirrors models/demos/audio/whisper/tests/test_whisper_modules.py::test_ttnn_whisper
(use_kv_cache=False path). Random mel input instead of the feature extractor (shapes are what matter).
"""
import json
import sys

import torch
import ttnn
from transformers import WhisperConfig, WhisperModel

from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.audio.whisper.tt import ttnn_optimized_functional_whisper as ttnn_model
from models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE

MODEL = "openai/whisper-base"  # config only; op-dependency structure is size-independent
BATCH = 1
DECODER_SEQ = 32  # prefill-style decoder step
OUT = "/tmp/whisper_capture.json"


def main():
    config = WhisperConfig.from_pretrained(MODEL)
    object.__setattr__(config, "_attn_implementation", "eager")
    model = WhisperModel(config).eval().float()  # RANDOM weights
    for m in model.modules():  # HF whisper indexes ALL_ATTENTION_FUNCTIONS[_attn_implementation]
        cfg = getattr(m, "config", None)
        if cfg is not None and getattr(cfg, "_attn_implementation", None) is None:
            object.__setattr__(cfg, "_attn_implementation", "eager")
    print(
        f"WhisperModel {MODEL}: d_model={config.d_model} enc_layers={config.encoder_layers} "
        f"dec_layers={config.decoder_layers} heads={config.encoder_attention_heads} (random weights)"
    )

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=WHISPER_L1_SMALL_SIZE)
    try:
        input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            convert_to_ttnn=ttnn_model.convert_to_ttnn,
            custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
            device=mesh_device,
        )

        input_features = torch.randn(BATCH, 80, 3000)  # mel spectrogram (random)
        decoder_input_ids = torch.ones(BATCH, DECODER_SEQ).to(torch.int32) * config.decoder_start_token_id

        input_embeds, decoder_hidden_states, decoder_attention_mask = ttnn_model.preprocess_inputs(
            config=config,
            input_features=input_features.unsqueeze(1),
            input_ids=decoder_input_ids,
            attention_mask=None,
            parameters=parameters,
            device=mesh_device,
            input_mesh_mapper=input_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = ttnn_model.whisper(
            config,
            input_embeds,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            kv_cache=None,
            cross_attn_cache=None,
            current_decode_pos=None,
            parameters=parameters,
        )
        ttnn.synchronize_device(mesh_device)
        captured = ttnn.graph.end_graph_capture()

        json.dump(captured, open(OUT, "w"))
        print(f"captured {len(captured)} nodes -> {OUT}")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    sys.exit(main())
