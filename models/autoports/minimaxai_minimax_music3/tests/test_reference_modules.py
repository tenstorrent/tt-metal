"""Vendored torch modules == the diffusers model classes on the real weights (reference venv; CPU fp32)."""
import pytest
import torch

pytest.importorskip("diffusers")


@pytest.mark.parametrize("num_layers", [2])
def test_dit_matches_diffusers(snapshot, num_layers):
    from diffusers import MiniMaxMusic3Transformer1DModel

    from models.autoports.minimaxai_minimax_music3.reference.dit import Music3DiT

    ours = Music3DiT.load(snapshot, num_layers=num_layers)
    ref = MiniMaxMusic3Transformer1DModel.from_pretrained(
        str(snapshot), subfolder="transformer", torch_dtype=torch.float32
    )
    ref.transformer_blocks = ref.transformer_blocks[:num_layers]
    ref.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 128, 128, generator=g)
    cond = torch.randn(2, 128, 2048, generator=g)
    t = torch.tensor([0.5, 0.0])
    with torch.no_grad():
        a = ours(x, t, cond)
        b = ref(hidden_states=x, timestep=t, encoder_hidden_states=cond, return_dict=False)[0]
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


def test_depth_decoder_matches_diffusers(snapshot):
    from diffusers import MiniMaxMusic3RVQDepthDecoder

    from models.autoports.minimaxai_minimax_music3.reference.depth_decoder import Music3DepthDecoder

    ours = Music3DepthDecoder.load(snapshot)
    ref = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        str(snapshot), subfolder="rvq_depth_decoder", torch_dtype=torch.float32
    ).eval()
    e = torch.randn(2, 5, 4096) * 0.02
    with torch.no_grad():
        a, b = ours(e), ref(e)
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
        for i in range(7):
            assert torch.allclose(ours.audio_heads[i](a[:, -1]), ref.audio_heads[i](b[:, -1]), atol=1e-4, rtol=1e-4)
        assert torch.equal(ours.audio_embeddings.weight, ref.audio_embeddings.weight)


def test_condition_encoder_and_vocoder_match_diffusers(snapshot):
    from diffusers import MiniMaxMusic3ConditionEncoder, MiniMaxMusic3Vocoder

    from models.autoports.minimaxai_minimax_music3.reference.condition_encoder import Music3ConditionEncoder
    from models.autoports.minimaxai_minimax_music3.reference.vocoder import Music3Vocoder

    ce, ce_ref = (
        Music3ConditionEncoder.load(snapshot),
        MiniMaxMusic3ConditionEncoder.from_pretrained(
            str(snapshot), subfolder="condition_encoder", torch_dtype=torch.float32
        ).eval(),
    )
    fh = torch.randn(1, 37, 8 * 4096)
    with torch.no_grad():
        a, b = ce(fh), ce_ref(fh)
    assert a.shape == b.shape == (1, 127, 2048) and torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    vc, vc_ref = (
        Music3Vocoder.load(snapshot),
        MiniMaxMusic3Vocoder.from_pretrained(str(snapshot), subfolder="vocoder", torch_dtype=torch.float32).eval(),
    )
    lat = torch.randn(1, 128, 40)
    with torch.no_grad():
        a, b = vc(lat), vc_ref(lat)
    assert a.shape == b.shape == (1, 2, 40 * 512) and torch.allclose(a, b, atol=1e-5, rtol=1e-5)
