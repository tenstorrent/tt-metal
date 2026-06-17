# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase-4 PCC gate: the composed Voxtral audio-features path (encoder -> reshape(-1,5120) ->
projector) vs HF VoxtralForConditionalGeneration.get_audio_features. This is the audio half of the
e2e: its output is what gets scattered into the text embeddings at audio_token_id. Combined with the
validated text decoder (Phase 3) and the exact masked_scatter, it closes the e2e correctness story.
"""
import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.audio.whisper.tt import ttnn_optimized_functional_whisper as ttnn_whisper
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.tt_transformers.tt.multimodal.voxtral.test_voxtral_audio_encoder import voxtral_audio_preprocessor
from tests.ttnn.utils_for_testing import comp_pcc

MODEL_NAME = "mistralai/Voxtral-Mini-3B-2507"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_voxtral_audio_features(mesh_device):
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    full = transformers.VoxtralForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval()
    enc_cfg = full.audio_tower.config
    enc_cfg._attn_implementation = "eager"
    enc_cfg.decoder_attention_heads = enc_cfg.encoder_attention_heads
    if not hasattr(enc_cfg, "decoder_layers"):
        enc_cfg.decoder_layers = 0
    inter = full.config.audio_config.intermediate_size  # 5120

    from transformers import AutoFeatureExtractor

    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    sr = fe.sampling_rate
    t = torch.arange(int(sr * 30.0)) / sr
    wav = (0.5 * torch.sin(2 * torch.pi * 220 * t) + 0.3 * torch.sin(2 * torch.pi * 440 * t)).numpy()
    feat = fe([wav] * mesh_device.get_num_devices(), sampling_rate=sr, return_tensors="pt").input_features.to(
        torch.float32
    )

    with torch.no_grad():
        golden = full.get_audio_features(feat).pooler_output  # [B*375, 3072]

    # TT audio encoder (reuse validated whisper-encoder path)
    enc_params = preprocess_model_parameters(
        initialize_model=lambda: full.audio_tower.eval(),
        convert_to_ttnn=ttnn_whisper.convert_to_ttnn,
        custom_preprocessor=voxtral_audio_preprocessor(weights_mesh_mapper),
        prefix="encoder",
        device=mesh_device,
    )
    input_embeds = ttnn_whisper.preprocess_encoder_inputs(
        config=enc_cfg,
        input_features=feat.unsqueeze(1),
        parameters=enc_params,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    enc_out = ttnn_whisper.encoder(enc_cfg, input_embeds, parameters=enc_params)
    enc_torch = ttnn.to_torch(enc_out, mesh_composer=output_mesh_composer)[: feat.shape[0]]  # [B,1500,1280]

    # reshape(-1,5120) groups 4 frames, then the validated projector
    reshaped = enc_torch.reshape(-1, inter)
    proj = full.multi_modal_projector
    w1 = ttnn.from_torch(
        proj.linear_1.weight.t().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w2 = ttnn.from_torch(
        proj.linear_2.weight.t().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x = ttnn.from_torch(
        reshaped,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    h = ttnn.linear(x, w1, activation="gelu")
    h = ttnn.linear(h, w2)
    out = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[: golden.shape[0]]

    passed, pcc = comp_pcc(golden, out, 0.97)
    print(f"Voxtral composed audio-features PCC: {pcc}")
    assert passed, f"Voxtral audio-features PCC too low: {pcc}"
