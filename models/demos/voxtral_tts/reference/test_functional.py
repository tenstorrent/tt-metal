"""
Reference verification tests for Voxtral-4B-TTS-2603.

Each test exercises one block of the model against itself (sanity checks)
and, when weights are available, verifies the full pipeline produces valid audio.

Run:
  cd tt-metal
  export PYTHONPATH=$(pwd):$(pwd)/models
  source python_env/bin/activate
  pytest models/demos/voxtral_tts/reference/test_functional.py -v -s
"""

import os
from pathlib import Path

import pytest
import torch

from models.demos.voxtral_tts.reference.functional import (
    acoustic_transformer_step,
    build_alibi_bias,
    build_rope_cache,
    causal_conv1d,
    causal_conv_transpose1d,
    codec_attention,
    codec_decoder_forward,
    ode_solve,
    rms_norm,
    text_decoder_forward,
    text_decoder_layer,
    time_sinusoidal_embedding,
)
from models.demos.voxtral_tts.tt.load_checkpoint import (
    get_acoustic_transformer_state,
    get_codec_decoder_state,
    get_semantic_codebook,
    get_text_decoder_state,
    load_state_dict,
    load_voice_embeddings,
)

MODEL_DIR = Path(os.environ.get("VOXTRAL_MODEL_DIR", "/tmp/voxtral_tts_weights"))
WEIGHTS_PATH = MODEL_DIR / "consolidated.safetensors"
WEIGHTS_AVAILABLE = WEIGHTS_PATH.exists()

PCC_THRESHOLD = 0.99
P99_THRESHOLD = 0.02


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def p99_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).abs().flatten()
    k = max(1, int(0.99 * diff.numel()))
    return diff.kthvalue(k).values.item()


# ─────────────────────────── unit tests (no weights needed) ───────────────


class TestHelpers:
    def test_rms_norm_shape(self):
        x = torch.randn(2, 16, 128)
        w = torch.ones(128)
        out = rms_norm(x, w)
        assert out.shape == x.shape

    def test_rms_norm_unit_weight(self):
        x = torch.randn(4, 64)
        w = torch.ones(64)
        out = rms_norm(x, w)
        # after rms_norm with unit weight, each vector should have rms ≈ 1
        rms_vals = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms_vals, torch.ones_like(rms_vals), atol=1e-4)

    def test_rope_cache_shape(self):
        cos, sin = build_rope_cache(64, 128, 1e6, "cpu")
        assert cos.shape == (64, 128)
        assert sin.shape == (64, 128)

    def test_rope_cos_sin_identity(self):
        cos, sin = build_rope_cache(32, 128, 1e6, "cpu")
        # cos² + sin² should be 1 per dimension
        assert torch.allclose(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-5)

    def test_alibi_bias_causal_shape(self):
        bias = build_alibi_bias(seq_len=16, n_heads=8, window_size=4, device="cpu")
        assert bias.shape == (1, 8, 16, 16)

    def test_alibi_bias_causal_upper_tri_is_neg_inf(self):
        bias = build_alibi_bias(seq_len=8, n_heads=4, window_size=16, device="cpu")
        for h in range(4):
            for i in range(8):
                for j in range(i + 1, 8):
                    assert bias[0, h, i, j] == float("-inf"), f"Position ({i},{j}) should be masked"

    def test_alibi_bias_window_masks_far_positions(self):
        bias = build_alibi_bias(seq_len=8, n_heads=4, window_size=2, device="cpu")
        # Position (5, 0): distance 5 > window 2 → should be -inf
        assert bias[0, 0, 5, 0] == float("-inf")
        # Position (5, 4): distance 1 ≤ window 2 → should be finite
        assert bias[0, 0, 5, 4].item() != float("-inf")

    def test_causal_conv1d_output_shape(self):
        B, L, C = 1, 32, 64
        x = torch.randn(B, L, C)
        w = torch.randn(128, C, 3)
        out = causal_conv1d(x, w)
        assert out.shape == (B, L, 128)  # stride=1, same length

    def test_causal_conv1d_causal_property(self):
        """Output at position t should not depend on inputs at position > t."""
        B, L, C = 1, 10, 4
        x1 = torch.randn(B, L, C)
        x2 = x1.clone()
        x2[0, 5:, :] = torch.randn(5, C)  # change positions 5..9

        w = torch.randn(4, C, 3)
        out1 = causal_conv1d(x1, w)
        out2 = causal_conv1d(x2, w)

        # Outputs at positions 0..4 should be identical (causal)
        assert torch.allclose(
            out1[0, :5], out2[0, :5], atol=1e-5
        ), "Causal property violated: future inputs affected past outputs"

    def test_conv_transpose1d_output_length(self):
        B, L, C = 1, 16, 64
        x = torch.randn(B, L, C)
        # ConvTranspose weight: [in, out, k] for F.conv_transpose1d
        w = torch.randn(C, C, 4)
        out = causal_conv_transpose1d(x, w, stride=2)
        assert out.shape[1] == L * 2, f"Expected L*2={L*2}, got {out.shape[1]}"

    def test_time_sinusoidal_embedding_shape(self):
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        emb = time_sinusoidal_embedding(t, dim=3072)
        assert emb.shape == (5, 3072)

    def test_time_embedding_different_timesteps_are_different(self):
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])
        e1 = time_sinusoidal_embedding(t1, dim=128)
        e2 = time_sinusoidal_embedding(t2, dim=128)
        assert not torch.allclose(e1, e2), "t=0 and t=1 embeddings should differ"


class TestReferenceBlocksWithRandomWeights:
    """Smoke tests: verify correct I/O shapes and gradient flow using random weights."""

    @pytest.fixture
    def mock_text_sd(self):
        # Use real model dims but only 2 layers; B=1, S=4 for speed
        D, Dh, n_h, n_kv, hd = 3072, 9216, 32, 8, 128
        sd = {}
        sd["mm_audio_embeddings.tok_embeddings.weight"] = torch.randn(1000, D) * 0.01
        sd["norm.weight"] = torch.ones(D)
        for i in range(2):
            sd[f"layers.{i}.attention_norm.weight"] = torch.ones(D)
            sd[f"layers.{i}.ffn_norm.weight"] = torch.ones(D)
            sd[f"layers.{i}.attention.wq.weight"] = torch.randn(n_h * hd, D) * 0.01
            sd[f"layers.{i}.attention.wk.weight"] = torch.randn(n_kv * hd, D) * 0.01
            sd[f"layers.{i}.attention.wv.weight"] = torch.randn(n_kv * hd, D) * 0.01
            sd[f"layers.{i}.attention.wo.weight"] = torch.randn(D, n_h * hd) * 0.01
            sd[f"layers.{i}.feed_forward.w1.weight"] = torch.randn(Dh, D) * 0.01
            sd[f"layers.{i}.feed_forward.w2.weight"] = torch.randn(D, Dh) * 0.01
            sd[f"layers.{i}.feed_forward.w3.weight"] = torch.randn(Dh, D) * 0.01
        return sd

    def test_text_decoder_forward_shape(self, mock_text_sd):
        B, S, D = 1, 4, 3072
        input_ids = torch.randint(0, 1000, (B, S))
        out, _ = text_decoder_forward(input_ids, mock_text_sd, n_layers=2)
        assert out.shape == (B, S, D)

    def test_text_decoder_with_capture(self, mock_text_sd):
        input_ids = torch.randint(0, 1000, (1, 4))
        out, caps = text_decoder_forward(
            input_ids,
            mock_text_sd,
            n_layers=2,
            capture_intermediates=True,
        )
        assert "embed" in caps
        assert "final_norm" in caps
        assert "layer_0" in caps

    def test_text_decoder_deterministic(self, mock_text_sd):
        input_ids = torch.randint(0, 1000, (1, 4))
        out1, _ = text_decoder_forward(input_ids, mock_text_sd, n_layers=2)
        out2, _ = text_decoder_forward(input_ids, mock_text_sd, n_layers=2)
        assert torch.allclose(out1, out2), "Decoder should be deterministic"

    @pytest.fixture
    def mock_codec_sd(self):
        C = 64  # small for testing
        sd = {}
        # Initial conv (fused weight_norm)
        sd["decoder_blocks.0.conv.weight"] = torch.randn(C, 16, 3) * 0.01  # [out, in, k]
        # Block 1: 2 transformer layers
        for li in range(2):
            pfx = f"decoder_blocks.1.layers.{li}"
            sd[f"{pfx}.attention_norm.weight"] = torch.ones(C)
            sd[f"{pfx}.ffn_norm.weight"] = torch.ones(C)
            sd[f"{pfx}.attention.q_norm.weight"] = torch.ones(C)
            sd[f"{pfx}.attention.k_norm.weight"] = torch.ones(C)
            sd[f"{pfx}.attention.wq.weight"] = torch.randn(C, C) * 0.01
            sd[f"{pfx}.attention.wk.weight"] = torch.randn(C, C) * 0.01
            sd[f"{pfx}.attention.wv.weight"] = torch.randn(C, C) * 0.01
            sd[f"{pfx}.attention.wo.weight"] = torch.randn(C, C) * 0.01
            sd[f"{pfx}.feed_forward.w1.weight"] = torch.randn(C * 4, C) * 0.01
            sd[f"{pfx}.feed_forward.w2.weight"] = torch.randn(C, C * 4) * 0.01
            sd[f"{pfx}.feed_forward.w3.weight"] = torch.randn(C * 4, C) * 0.01
            sd[f"{pfx}.attention_scale"] = torch.ones(C) * 0.01
            sd[f"{pfx}.ffn_scale"] = torch.ones(C) * 0.01
        # Block 2: ConvTranspose
        sd["decoder_blocks.2.conv.weight"] = torch.randn(C, C, 4) * 0.01
        return sd

    def test_causal_alibi_attention_shape(self, mock_codec_sd):
        B, L, C = 1, 8, 64
        x = torch.randn(B, L, C)
        out, _ = codec_attention(x, mock_codec_sd, block_idx=1, layer_idx=0, window_size=4, n_heads=2, head_dim=32)
        assert out.shape == (B, L, C)


# ─────────────────────────── weight-dependent tests ───────────────────


@pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason=f"Model weights not found at {WEIGHTS_PATH}")
class TestWithRealWeights:
    @pytest.fixture(scope="class")
    def state_dicts(self):
        sd = load_state_dict(WEIGHTS_PATH)
        return {
            "full": sd,
            "text": get_text_decoder_state(sd),
            "acoustic": get_acoustic_transformer_state(sd),
            "codec": get_codec_decoder_state(sd),
        }

    def test_weight_norm_fusing_produces_correct_shapes(self, state_dicts):
        sd = state_dicts["codec"]
        # After fusing, parametrizations.* keys should be gone
        for k in sd:
            assert "parametrizations" not in k, f"Unfused weight_norm key found: {k}"
        # Initial conv weight should have shape [1024, 292, 3]
        assert sd["decoder_blocks.0.conv.weight"].shape == (1024, 292, 3)

    def test_semantic_codebook_shape(self, state_dicts):
        sd = state_dicts["full"]
        codebook = get_semantic_codebook(sd)
        assert codebook.shape == (8192, 256)

    def test_text_decoder_single_layer_pcc(self, state_dicts):
        sd = state_dicts["text"]
        B, S, D = 1, 32, 3072
        x = torch.randn(B, S, D, dtype=torch.bfloat16)
        cos, sin = build_rope_cache(S, 128, 1e6, "cpu")

        out1, caps1 = text_decoder_layer(x, sd, 0, cos, sin, capture_intermediates=True)
        out2, _ = text_decoder_layer(x, sd, 0, cos, sin)

        p = pcc(out1, out2)
        assert p > PCC_THRESHOLD, f"Layer forward not deterministic: PCC={p}"

    def test_text_decoder_all_layers_pcc(self, state_dicts):
        sd = state_dicts["text"]
        input_ids = torch.randint(0, 131072, (1, 16))
        h1, _ = text_decoder_forward(input_ids, sd)
        h2, _ = text_decoder_forward(input_ids, sd)
        p = pcc(h1, h2)
        assert p > PCC_THRESHOLD, f"Text decoder not deterministic: PCC={p}"

    def test_acoustic_transformer_velocity_shape(self, state_dicts):
        sd = state_dicts["acoustic"]
        B, N = 1, 25
        h = torch.randn(B, N, 3072, dtype=torch.bfloat16)
        x_t = torch.randn(B, N, 36, dtype=torch.bfloat16)
        v, sem_logits, _ = acoustic_transformer_step(h, x_t, t=0.5, sd=sd)
        assert v.shape == (B, N, 36)
        assert sem_logits.shape == (B, N, 8320)

    def test_ode_solve_output_range(self, state_dicts):
        sd = state_dicts["acoustic"]
        B, N = 1, 10
        h = torch.zeros(B, N, 3072, dtype=torch.bfloat16)
        codes, x_continuous, _ = ode_solve(h, sd, n_steps=8)
        assert codes.shape == (B, N, 36)
        # FSQ quantized values must be in [0, 20]
        assert codes.min() >= 0
        assert codes.max() <= 20

    def test_codec_decoder_output_shape(self, state_dicts):
        sd = state_dicts["codec"]
        B, N = 1, 25
        semantic_codes = torch.randint(0, 8192, (B, N))
        acoustic_codes = torch.randint(0, 21, (B, N, 36))
        waveform, _ = codec_decoder_forward(semantic_codes, acoustic_codes, sd)
        # Expected: B × (N * 8 * 240) samples = B × N * 1920
        assert waveform.shape == (B, N * 1920)

    def test_codec_decoder_waveform_amplitude(self, state_dicts):
        sd = state_dicts["codec"]
        B, N = 1, 100
        semantic_codes = torch.randint(0, 8192, (B, N))
        acoustic_codes = torch.randint(0, 21, (B, N, 36))
        waveform, _ = codec_decoder_forward(semantic_codes, acoustic_codes, sd)
        # Audio should be in a reasonable amplitude range
        assert waveform.abs().max() < 10.0, f"Waveform amplitude too large: {waveform.abs().max()}"

    def test_end_to_end_produces_audio(self, state_dicts):
        """Smoke test: does the pipeline produce audio of the right length?"""
        sd_full = state_dicts["full"]
        sd_text = state_dicts["text"]
        sd_acoustic = state_dicts["acoustic"]
        sd_codec = state_dicts["codec"]

        voice_dir = MODEL_DIR / "voice_embedding"
        if not voice_dir.exists():
            pytest.skip("Voice embeddings not found")

        voices = load_voice_embeddings(MODEL_DIR)
        voice_name = "casual_male"
        voice_emb = voices[voice_name].unsqueeze(0)  # [1, V_frames, 3072]

        # Short test text: "Hello."
        from mistral_common.protocol.instruct.chunk import TextChunk
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tok = MistralTokenizer.from_file(str(MODEL_DIR / "tekken.json"))
        req = ChatCompletionRequest(messages=[UserMessage(content=[TextChunk(text="Hello.")])])
        text_ids = torch.tensor([tok.encode_chat_completion(req).tokens])

        from models.demos.voxtral_tts.reference.functional import tts_generate

        waveform, semantic_codes, acoustic_codes, _ = tts_generate(
            text_ids=text_ids,
            voice_emb=voice_emb.to(torch.bfloat16),
            sd_text=sd_text,
            sd_acoustic=sd_acoustic,
            sd_codec=sd_codec,
            max_audio_frames=50,
        )

        assert waveform.shape[0] == 1
        assert waveform.shape[1] > 0
        print(f"\nGenerated waveform: {waveform.shape[1]} samples = {waveform.shape[1]/24000:.2f}s")
        print(f"Semantic codes shape: {semantic_codes.shape}")
        print(f"Acoustic codes shape: {acoustic_codes.shape}")
        print(f"Waveform amplitude: mean={waveform.abs().mean():.4f}, max={waveform.abs().max():.4f}")
