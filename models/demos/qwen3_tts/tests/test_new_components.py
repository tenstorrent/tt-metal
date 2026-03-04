# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test new TTNN components: text_projection and speaker_encoder.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    pytest models/demos/qwen3_tts/tests/test_new_components.py -v
"""

from pathlib import Path

import pytest
import torch

import ttnn


def load_weights():
    """Load model weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))
    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        if "speech_tokenizer" not in str(f):
            state_dict.update(load_file(f))
    return state_dict


class TestTextProjection:
    """Test text_projection in TTNN Talker."""

    @pytest.fixture
    def setup(self):
        """Setup device and model."""
        device = ttnn.open_device(device_id=0)
        state_dict = load_weights()
        yield device, state_dict
        ttnn.close_device(device)

    def test_text_projection_loaded(self, setup):
        """Test that text_projection weights are loaded."""
        device, state_dict = setup

        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
        from models.demos.qwen3_tts.tt.talker import Talker

        config = Qwen3TTSTalkerConfig()
        talker = Talker(device=device, config=config, state_dict=state_dict)

        assert talker.has_text_projection, "text_projection should be loaded"
        assert talker.text_proj_fc1 is not None
        assert talker.text_proj_fc2 is not None

    def test_text_projection_forward(self, setup):
        """Test text_projection forward pass."""
        device, state_dict = setup

        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
        from models.demos.qwen3_tts.tt.talker import Talker

        config = Qwen3TTSTalkerConfig()
        talker = Talker(device=device, config=config, state_dict=state_dict)

        # Create dummy input: [batch, 1, seq_len, hidden_size]
        batch, seq_len, hidden = 1, 10, 2048
        text_embeds = torch.randn(batch, 1, seq_len, hidden)
        text_embeds_tt = ttnn.from_torch(
            text_embeds,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run projection
        output_tt = talker.project_text(text_embeds_tt)
        output = ttnn.to_torch(output_tt)

        # Check output shape
        assert output.shape == (
            batch,
            1,
            seq_len,
            hidden,
        ), f"Expected shape {(batch, 1, seq_len, hidden)}, got {output.shape}"
        print(f"text_projection output shape: {output.shape}")

    def test_text_projection_pcc(self, setup):
        """Test text_projection PCC against reference."""
        device, state_dict = setup

        import torch.nn.functional as F

        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
        from models.demos.qwen3_tts.tt.talker import Talker

        config = Qwen3TTSTalkerConfig()
        talker = Talker(device=device, config=config, state_dict=state_dict)

        # Reference implementation
        fc1_w = state_dict["talker.text_projection.linear_fc1.weight"]
        fc1_b = state_dict["talker.text_projection.linear_fc1.bias"]
        fc2_w = state_dict["talker.text_projection.linear_fc2.weight"]
        fc2_b = state_dict["talker.text_projection.linear_fc2.bias"]

        def ref_project_text(x):
            h = F.linear(x, fc1_w, fc1_b)
            h = F.silu(h)
            return F.linear(h, fc2_w, fc2_b)

        # Create input
        batch, seq_len, hidden = 1, 10, 2048
        text_embeds = torch.randn(batch, seq_len, hidden)

        # Reference output
        ref_output = ref_project_text(text_embeds)

        # TTNN output
        text_embeds_4d = text_embeds.unsqueeze(1)  # [batch, 1, seq_len, hidden]
        text_embeds_tt = ttnn.from_torch(
            text_embeds_4d,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_tt = talker.project_text(text_embeds_tt)
        ttnn_output = ttnn.to_torch(output_tt).squeeze(1).float()

        # Compute PCC
        pcc = torch.corrcoef(torch.stack([ref_output.flatten(), ttnn_output.flatten()]))[0, 1].item()
        print(f"text_projection PCC: {pcc:.6f}")

        assert pcc > 0.99, f"PCC {pcc} is below threshold 0.99"


class TestSpeakerEncoder:
    """Test speaker_encoder in TTNN."""

    @pytest.fixture
    def setup(self):
        """Setup device and model."""
        device = ttnn.open_device(device_id=0)
        state_dict = load_weights()
        yield device, state_dict
        ttnn.close_device(device)

    def test_speaker_encoder_loaded(self, setup):
        """Test that speaker_encoder weights are loaded."""
        device, state_dict = setup

        from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

        speaker_enc = SpeakerEncoder(device=device, state_dict=state_dict)

        assert len(speaker_enc.pytorch_weights) > 0, "Speaker encoder weights should be loaded"
        assert "blocks.0.conv.weight" in speaker_enc.pytorch_weights
        assert "fc.weight" in speaker_enc.pytorch_weights

    def test_speaker_encoder_forward(self, setup):
        """Test speaker_encoder forward pass."""
        device, state_dict = setup

        from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

        speaker_enc = SpeakerEncoder(device=device, state_dict=state_dict)

        # Create dummy mel spectrogram: [batch, n_mels, time]
        batch, n_mels, time = 1, 128, 100
        mel = torch.randn(batch, n_mels, time)

        # Run forward
        output = speaker_enc.forward(mel)

        # Check output shape
        assert output.shape == (batch, 2048), f"Expected shape {(batch, 2048)}, got {output.shape}"
        print(f"speaker_encoder output shape: {output.shape}")

    def test_speaker_encoder_from_audio(self, setup):
        """Test speaker_encoder forward_from_audio."""
        device, state_dict = setup

        from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

        speaker_enc = SpeakerEncoder(device=device, state_dict=state_dict)

        # Create dummy audio: 1 second at 24kHz
        audio = torch.randn(24000)

        # Run forward
        output = speaker_enc.forward_from_audio(audio)

        # Check output shape
        assert output.shape == (1, 2048), f"Expected shape (1, 2048), got {output.shape}"
        print(f"speaker_encoder from audio output shape: {output.shape}")

    def test_speaker_encoder_pcc(self, setup):
        """Test speaker_encoder PCC against reference."""
        device, state_dict = setup

        from models.demos.qwen3_tts.reference.functional import extract_speaker_encoder_weights, speaker_encoder_forward
        from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

        speaker_enc = SpeakerEncoder(device=device, state_dict=state_dict)

        # Create dummy mel spectrogram
        batch, n_mels, time = 1, 128, 100
        mel = torch.randn(batch, n_mels, time)

        # Reference output
        ref_weights = extract_speaker_encoder_weights(state_dict)
        ref_output = speaker_encoder_forward(mel, ref_weights)

        # TTNN output
        ttnn_output = speaker_enc.forward(mel)

        # Compute PCC
        pcc = torch.corrcoef(torch.stack([ref_output.flatten(), ttnn_output.flatten()]))[0, 1].item()
        print(f"speaker_encoder PCC: {pcc:.6f}")

        assert pcc > 0.99, f"PCC {pcc} is below threshold 0.99"


class TestFullModel:
    """Test full Qwen3TTS model with new components."""

    @pytest.fixture
    def setup(self):
        """Setup device and model."""
        device = ttnn.open_device(device_id=0)
        state_dict = load_weights()
        yield device, state_dict
        ttnn.close_device(device)

    def test_model_components(self, setup):
        """Test that all components are accessible."""
        device, state_dict = setup

        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        model = Qwen3TTS(device=device, state_dict=state_dict)

        # Check all components exist
        assert model.talker is not None
        assert model.code_predictor is not None
        assert model.speaker_encoder is not None

        # Check delegated methods work
        assert model.talker.has_text_projection
        assert hasattr(model, "get_text_embedding")
        assert hasattr(model, "get_codec_embedding")
        assert hasattr(model, "project_text")
        assert hasattr(model, "extract_speaker_embedding")

        print("All model components loaded successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
