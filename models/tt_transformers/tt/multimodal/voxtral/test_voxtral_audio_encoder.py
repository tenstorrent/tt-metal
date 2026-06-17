# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 PCC gate for the Voxtral audio encoder.

Voxtral's `audio_tower` (`VoxtralEncoder`) is a Whisper-large-v3 encoder, so it is driven by the
existing on-device tt-metal Whisper encoder (`models/demos/audio/whisper`). This test loads the
real Voxtral `audio_tower` weights into that encoder and PCC-gates its output against the HF
reference. VoxtralEncoderConfig already exposes the whisper-style attrs (d_model,
encoder_attention_heads, num_mel_bins) the tt encoder reads, so no config shim is needed.
"""
import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight, preprocess_model_parameters

import ttnn
from models.demos.audio.whisper.tt import ttnn_optimized_functional_whisper as ttnn_whisper
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import comp_pcc


def voxtral_audio_preprocessor(weights_mesh_mapper):
    """Voxtral's encoder uses VoxtralAttention (not WhisperAttention) — structurally identical
    (q/k/v/out_proj, k bias-free). Fuse QKV exactly like the whisper preprocessor; delegate
    everything else (conv1/conv2 kept-as-torch, embed_positions) to the whisper preprocessor."""
    whisper_pp = ttnn_whisper.create_custom_mesh_preprocessor(weights_mesh_mapper)

    def pp(torch_model, name):
        if type(torch_model).__name__ == "VoxtralAttention":
            h = torch_model.k_proj.weight.shape[0]
            qkv_w = torch.cat([torch_model.q_proj.weight, torch_model.k_proj.weight, torch_model.v_proj.weight], dim=0)
            qkv_b = torch.cat([torch_model.q_proj.bias, torch.zeros(h), torch_model.v_proj.bias], dim=0)
            return {
                "query_key_value": {
                    "weight": preprocess_linear_weight(
                        qkv_w, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
                    ),
                    "bias": preprocess_linear_bias(qkv_b, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper),
                },
                "out_proj": {
                    "weight": preprocess_linear_weight(
                        torch_model.out_proj.weight, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
                    ),
                    "bias": preprocess_linear_bias(
                        torch_model.out_proj.bias, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
                    ),
                },
            }
        return whisper_pp(torch_model, name)

    return pp


MODEL_NAME = "mistralai/Voxtral-Mini-3B-2507"
WHISPER_L1_SMALL_SIZE = 16384


@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("seq_frames", [3000])  # 30s @ 100 frames/s -> Whisper standard
def test_voxtral_audio_encoder(mesh_device, seq_frames):
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    full = transformers.VoxtralForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    audio_tower = full.audio_tower.eval()
    config = audio_tower.config
    config._attn_implementation = "eager"
    # whisper tt code reads a few decoder-side attrs even on the encoder path; VoxtralEncoderConfig
    # only carries the encoder ones. Mirror them so the shared whisper helpers don't AttributeError.
    config.decoder_attention_heads = config.encoder_attention_heads
    if not hasattr(config, "decoder_layers"):
        config.decoder_layers = 0

    # Realistic log-mel features from a synthetic waveform (a real trained encoder is sensitive to
    # out-of-distribution white-noise input; use the actual feature extractor like production).
    from transformers import AutoFeatureExtractor

    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    sr = fe.sampling_rate
    t = torch.arange(int(sr * 30.0)) / sr
    wav = (
        0.5 * torch.sin(2 * torch.pi * 220 * t)
        + 0.3 * torch.sin(2 * torch.pi * 440 * t)
        + 0.2 * torch.sin(2 * torch.pi * 880 * t)
    ).numpy()
    feat = fe([wav] * mesh_device.get_num_devices(), sampling_rate=sr, return_tensors="pt").input_features
    torch_input_features = feat.to(torch.float32)[..., :seq_frames]
    with torch.no_grad():
        golden = audio_tower(torch_input_features).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: audio_tower,
        convert_to_ttnn=ttnn_whisper.convert_to_ttnn,
        custom_preprocessor=voxtral_audio_preprocessor(weights_mesh_mapper),
        prefix="encoder",  # whisper tt code hardcodes the "encoder." name prefix (conv1/conv2/embed_positions)
        device=mesh_device,
    )
    input_embeds = ttnn_whisper.preprocess_encoder_inputs(
        config=config,
        input_features=torch_input_features.unsqueeze(1),
        parameters=parameters,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    output = ttnn_whisper.encoder(config, input_embeds, parameters=parameters)
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)[: golden.shape[0]]

    # 32-layer Whisper-v3 encoder at bf16/LoFi vs fp32 reference lands ~0.97 (precision-bound; the
    # full-model e2e logit PCC is the ultimate correctness gate downstream).
    passed, pcc = comp_pcc(golden, output, 0.97)
    print(f"Voxtral audio encoder PCC: {pcc}")
    assert passed, f"Voxtral audio encoder PCC too low: {pcc}"
