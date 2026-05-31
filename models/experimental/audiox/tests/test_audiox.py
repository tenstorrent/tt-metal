import pytest
import torch
import ttnn

from models.experimental.audiox.reference.audiox_model import AudioXModel
from models.experimental.audiox.tt.ttnn_audiox import TtnnAudioXModel


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [64, 128])
def test_audiox_diffusion_transformer(batch_size, seq_len, device):
    ref_model = AudioXModel(dim=256, depth=4, heads=4)
    ref_model.eval()
    ttnn_model = TtnnAudioXModel(ref_model, device)
    x = torch.randn(batch_size, seq_len, 80)
    t = torch.rand(batch_size, 1)
    out = ttnn_model.diffusion_forward(x, t)
    assert out.shape == (batch_size, seq_len, 80)


def test_audiox_text_to_audio(device):
    ref_model = AudioXModel(dim=256, depth=4, heads=4)
    ref_model.eval()
    ttnn_model = TtnnAudioXModel(ref_model, device)
    audio = ttnn_model.generate(num_steps=5, length=32)
    assert audio.shape[-1] == 32


def test_audiox_multimodal_fusion(device):
    ref_model = AudioXModel(dim=256, depth=2, heads=4)
    ref_model.eval()
    ttnn_model = TtnnAudioXModel(ref_model, device)
    text_embeds = torch.randn(1, 77, 512)
    audio_embeds = torch.randn(1, 32, 80)
    text_ctx = ttnn_model.encode_text(text_embeds)
    audio_ctx = ttnn_model.encode_audio(audio_embeds)
    fused = ttnn_model.fusion_forward(text_ctx, audio_ctx)
    assert fused is not None


def test_audiox_text_encoder(device):
    ref_model = AudioXModel(dim=256, depth=2, heads=4)
    ref_model.eval()
    ttnn_model = TtnnAudioXModel(ref_model, device)
    text_embeds = torch.randn(1, 77, 512)
    out = ttnn_model.encode_text(text_embeds)
    assert out is not None


def test_audiox_generation_pcc(device):
    torch.manual_seed(42)
    ref_model = AudioXModel(dim=256, depth=2, heads=4)
    ref_model.eval()
    ttnn_model = TtnnAudioXModel(ref_model, device)
    audio_ttnn = ttnn_model.generate(num_steps=5, length=32)
    torch.manual_seed(42)
    audio_ref = ref_model.generate(num_steps=5, length=32)
    diff = (audio_ttnn - audio_ref).abs().mean().item()
    assert diff < 1.0, f"Diff too high: {diff}"
