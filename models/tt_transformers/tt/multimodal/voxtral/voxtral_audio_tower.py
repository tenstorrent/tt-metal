# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral audio tower: Whisper-v3 encoder (reused from models/demos/audio/whisper) + multimodal
projector, producing audio embeddings in the text-decoder space. Mirrors the role of
TtMistralVisionTransformer in the mistral_24b VLM e2e, with the Pixtral tower swapped for the
Whisper audio encoder."""
import torch
import transformers
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_model_parameters

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.audio.whisper.tt import ttnn_optimized_functional_whisper as ttnn_whisper


def voxtral_audio_preprocessor(weights_mesh_mapper):
    """Fuse QKV for VoxtralAttention (== WhisperAttention structurally, k bias-free); delegate
    conv1/conv2 + embed_positions to the whisper preprocessor."""
    from ttnn.model_preprocessing import preprocess_linear_bias

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


class TtVoxtralAudioTower(LightweightModule):
    def __init__(
        self, mesh_device, state_dict, hf_config, weights_mesh_mapper, input_mesh_mapper, output_mesh_composer
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.input_mesh_mapper = input_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.output_mesh_composer = output_mesh_composer
        self.intermediate_size = hf_config.audio_config.intermediate_size

        # Reconstruct the HF VoxtralEncoder from config + the audio_tower.* weights so we can reuse the
        # whisper preprocess pipeline. The encoder config carries whisper-style attrs; mirror the
        # decoder-side ones the shared whisper helpers read.
        enc_cfg = transformers.models.voxtral.configuration_voxtral.VoxtralEncoderConfig(
            **hf_config.audio_config.to_dict()
        )
        enc_cfg._attn_implementation = "eager"
        enc_cfg.decoder_attention_heads = enc_cfg.encoder_attention_heads
        if not hasattr(enc_cfg, "decoder_layers"):
            enc_cfg.decoder_layers = 0
        self.enc_cfg = enc_cfg
        audio_tower = transformers.models.voxtral.modeling_voxtral.VoxtralEncoder(enc_cfg).eval()
        at_sd = {k[len("audio_tower.") :]: v for k, v in state_dict.items() if k.startswith("audio_tower.")}
        audio_tower.load_state_dict(at_sd, strict=False)

        self.params = preprocess_model_parameters(
            initialize_model=lambda: audio_tower,
            convert_to_ttnn=ttnn_whisper.convert_to_ttnn,
            custom_preprocessor=voxtral_audio_preprocessor(weights_mesh_mapper),
            prefix="encoder",
            device=mesh_device,
        )
        # Projector weights on device (transpose for ttnn.linear), matching the validated projector test.
        self.w1 = ttnn.from_torch(
            state_dict["multi_modal_projector.linear_1.weight"].t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.w2 = ttnn.from_torch(
            state_dict["multi_modal_projector.linear_2.weight"].t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, input_features):
        """input_features: [B, num_mel_bins, frames] torch -> audio embeds [B*frames//4, hidden] ttnn."""
        input_embeds = ttnn_whisper.preprocess_encoder_inputs(
            config=self.enc_cfg,
            input_features=input_features.unsqueeze(1),
            parameters=self.params,
            device=self.mesh_device,
            input_mesh_mapper=self.input_mesh_mapper,
            weights_mesh_mapper=self.weights_mesh_mapper,
        )
        enc = ttnn_whisper.encoder(self.enc_cfg, input_embeds, parameters=self.params)
        enc_t = ttnn.to_torch(enc, mesh_composer=self.output_mesh_composer)[: input_features.shape[0]]
        reshaped = enc_t.reshape(-1, self.intermediate_size)  # group 4 frames
        x = ttnn.from_torch(
            reshaped,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        h = ttnn.linear(x, self.w1, activation="gelu")
        h = ttnn.linear(h, self.w2)
        return h
