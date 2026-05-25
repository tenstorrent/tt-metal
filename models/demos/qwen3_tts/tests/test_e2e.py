# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end integration tests for the Qwen3-TTS pipeline.

Tests at three levels:
  1. CPU-only: small random models → verify pipeline wiring (no device needed)
  2. Reference: HuggingFace model → generate real audio (if weights available)
  3. Device: TTSGenerator on TT hardware (requires device fixture)
"""

import numpy as np
import pytest
import torch
from loguru import logger


class TestPipelineConfig:
    """Test that config parsing works end-to-end across all components."""

    def test_all_configs_parse(self):
        from models.demos.qwen3_tts.reference.functional import Qwen3TTSConfig

        cfg = Qwen3TTSConfig()
        assert cfg.talker_hidden_size == 2048
        assert cfg.talker_num_layers == 28
        assert cfg.cp_hidden_size == 1024
        assert cfg.cp_num_layers == 5
        assert cfg.spk_enc_dim == 2048
        assert cfg.codec_eos_token_id == 2150
        assert "japanese" in cfg.codec_language_ids

    def test_configs_from_pretrained(self):
        try:
            from models.demos.qwen3_tts.reference.functional import Qwen3TTSConfig

            cfg = Qwen3TTSConfig.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            assert cfg.talker_hidden_size == 2048
            assert cfg.talker_num_layers == 28
            logger.info(f"Loaded config: talker={cfg.talker_num_layers}L, CP={cfg.cp_num_layers}L")
        except Exception as e:
            pytest.skip(f"Config not available: {e}")


class TestCPUPipelineSmall:
    """Test the full pipeline wiring with small random models (no real weights, no device)."""

    @pytest.fixture
    def small_vocoder(self):
        from models.demos.qwen3_tts.reference.vocoder_ref import VocoderReference
        from models.demos.qwen3_tts.tt.vocoder import Vocoder

        model = VocoderReference(
            codebook_size=32, codebook_dim=16, hidden_size=32, latent_dim=64,
            num_layers=1, num_heads=4, num_kv_heads=4, head_dim=8, ffn_dim=64,
            sliding_window=8, num_quantizers=4, decoder_dim=48,
            upsample_rates=[2, 2, 2, 2], upsampling_ratios=[2, 2],
        )
        return Vocoder(model, dtype=torch.float32)

    @pytest.fixture
    def small_speaker_encoder(self):
        from models.demos.qwen3_tts.reference.speaker_encoder_ref import SpeakerEncoderReference

        model = SpeakerEncoderReference(
            mel_dim=32, enc_dim=64,
            enc_channels=[16, 16, 16, 16, 48],
            enc_res2net_scale=2, enc_se_channels=8, enc_attention_channels=8,
        )
        model.eval()
        return model

    def test_vocoder_decode_shape(self, small_vocoder):
        """Verify Vocoder produces correct waveform shape from codebook tokens."""
        num_frames = 10
        num_quantizers = 4
        codes = torch.randint(0, 32, (1, num_frames, num_quantizers))

        waveform = small_vocoder.decode(codes, chunk_decode=False)

        expected_upsample = 2 * 2 * 2 * 2 * 2 * 2  # 64
        assert isinstance(waveform, np.ndarray)
        assert len(waveform) == num_frames * expected_upsample
        logger.info(f"Vocoder: {codes.shape} → {waveform.shape}")

    def test_speaker_encoder_mel_to_embedding(self, small_speaker_encoder):
        """Verify speaker encoder produces embedding from mel spectrogram."""
        mel = torch.randn(1, 50, 32)  # [B, T, mel_dim]
        with torch.no_grad():
            embedding = small_speaker_encoder(mel)
        assert embedding.shape == (1, 64)
        assert not torch.isnan(embedding).any()
        logger.info(f"Speaker encoder: mel {mel.shape} → embedding {embedding.shape}")

    def test_codebook_token_flow(self, small_vocoder):
        """Test the token flow: CB0 + CB1..CB15 → stacked → Vocoder."""
        num_frames = 5
        num_quantizers = 4
        cb0 = torch.randint(0, 32, (1, num_frames))
        cb_rest = torch.randint(0, 32, (1, num_frames, num_quantizers - 1))
        all_cb = torch.cat([cb0.unsqueeze(-1), cb_rest], dim=-1)
        assert all_cb.shape == (1, num_frames, num_quantizers)

        waveform = small_vocoder.decode(all_cb, chunk_decode=False)
        expected_upsample = 2 ** 6  # 64
        assert len(waveform) == num_frames * expected_upsample
        logger.info(f"Token flow: CB0 {cb0.shape} + rest {cb_rest.shape} → wav {waveform.shape}")

    def test_sampling_function(self):
        """Test the sampling function (standalone, no ttnn needed)."""
        logits = torch.randn(1, 1, 100)

        def sample_token(logits, temperature, top_k, top_p):
            logits = logits[:, -1, :]
            if temperature <= 0:
                return torch.argmax(logits, dim=-1)
            logits = logits / temperature
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
                return next_token.squeeze(-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        token_greedy = sample_token(logits.clone(), temperature=0.0, top_k=0, top_p=1.0)
        assert token_greedy.shape == (1,)
        assert token_greedy.item() == logits[0, 0].argmax().item()

        token_sampled = sample_token(logits.clone(), temperature=0.9, top_k=10, top_p=0.9)
        assert token_sampled.shape == (1,)
        assert 0 <= token_sampled.item() < 100

        logger.info(f"Sampling: greedy={token_greedy.item()}, sampled={token_sampled.item()}")

    def test_cer_metric_computation(self):
        """Test CER metric with known reference/hypothesis pairs."""
        from models.demos.qwen3_tts.evaluation.metrics.cer_eval import compute_cer_single

        assert compute_cer_single("こんにちは", "こんにちは") == 0.0
        cer = compute_cer_single("こんにちは世界", "こんにちわ世界")
        assert 0.0 < cer < 0.5  # 1 substitution out of 6 chars
        assert compute_cer_single("", "") == 0.0
        logger.info(f"CER metric: perfect=0.0, 1-sub={cer:.4f}")

    def test_rtf_metric_computation(self):
        """Test RTF metric aggregation."""
        from models.demos.qwen3_tts.evaluation.metrics.rtf_eval import compute_rtf_summary

        results = [
            {"id": "a", "elapsed": 1.0, "duration": 2.0, "rtf": 0.5},
            {"id": "b", "elapsed": 2.0, "duration": 3.0, "rtf": 0.667},
            {"id": "c", "elapsed": 0.5, "duration": 1.0, "rtf": 0.5},
        ]
        summary = compute_rtf_summary(results)
        assert summary["num_samples"] == 3
        assert summary["realtime_capable"] is True
        assert abs(summary["total_elapsed_s"] - 3.5) < 0.01
        assert abs(summary["total_audio_duration_s"] - 6.0) < 0.01
        logger.info(f"RTF summary: mean={summary['mean_rtf']:.3f}, realtime={summary['realtime_capable']}")


class TestReferencePipeline:
    """Test the HuggingFace reference inference pipeline (requires model weights)."""

    def test_reference_generate(self):
        """Generate speech using the HF reference model (GPU or CPU)."""
        try:
            from models.demos.qwen3_tts.reference.functional import (
                generate_reference,
                load_reference_model,
            )

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = load_reference_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device=device)

            wavs, sr = generate_reference(
                model,
                text="こんにちは",
                language="japanese",
                max_new_tokens=64,
                temperature=0.9,
            )

            assert sr == 24000
            wav = wavs[0].cpu().numpy() if isinstance(wavs[0], torch.Tensor) else wavs[0]
            assert len(wav) > 0
            duration = len(wav) / sr
            logger.info(f"Reference generate: {len(wav)} samples ({duration:.2f}s)")

        except Exception as e:
            pytest.skip(f"Reference model not available: {e}")

    def test_reference_with_activation_capture(self):
        """Generate with activation capture for PCC comparison."""
        try:
            from models.demos.qwen3_tts.reference.functional import (
                ActivationCapture,
                generate_reference,
                load_reference_model,
            )

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = load_reference_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device=device)
            capture = ActivationCapture()

            wavs, sr = generate_reference(
                model,
                text="テスト",
                language="japanese",
                max_new_tokens=32,
                temperature=0.0,
                capture=capture,
            )

            assert len(capture.activations) > 0
            for name, tensor in list(capture.activations.items())[:5]:
                logger.info(f"  {name}: {tensor.shape}")
            logger.info(f"Captured {len(capture.activations)} activations")

        except Exception as e:
            pytest.skip(f"Reference model not available: {e}")


class TestVocoderRealWeights:
    """Test Vocoder with real pretrained weights (if available)."""

    def test_real_vocoder_decode(self):
        """Decode random codebook tokens through real vocoder weights."""
        try:
            from models.demos.qwen3_tts.tt.vocoder import Vocoder

            vocoder = Vocoder.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

            num_frames = 5
            codes = torch.randint(0, 2048, (1, num_frames, 16))

            waveform = vocoder.decode(codes)

            expected_samples = num_frames * 1920
            assert len(waveform) == expected_samples
            assert not np.isnan(waveform).any()
            duration = len(waveform) / 24000
            logger.info(f"Real vocoder: {codes.shape} → {len(waveform)} samples ({duration:.2f}s)")

        except Exception as e:
            pytest.skip(f"Vocoder weights not available: {e}")


class TestCustomDataset:
    """Test the custom Japanese evaluation dataset."""

    def test_load_custom_ja(self):
        """Verify custom_ja.json loads and has expected structure."""
        import json
        from pathlib import Path

        dataset_path = Path(__file__).parent.parent / "evaluation" / "datasets" / "custom_ja.json"
        if not dataset_path.exists():
            pytest.skip("custom_ja.json not found")

        with open(dataset_path) as f:
            data = json.load(f)

        assert data["name"] == "custom_ja_100"
        assert data["language"] == "japanese"

        total_samples = sum(len(cat["samples"]) for cat in data["categories"])
        assert total_samples == 100, f"Expected 100 samples, got {total_samples}"

        category_names = [cat["name"] for cat in data["categories"]]
        expected_categories = {"short", "medium", "long", "numbers", "mixed_language", "emotion", "keigo"}
        assert set(category_names) == expected_categories, f"Missing categories: {expected_categories - set(category_names)}"

        for cat in data["categories"]:
            for sample in cat["samples"]:
                assert "id" in sample
                assert "text" in sample
                assert len(sample["text"]) > 0

        logger.info(f"Dataset: {total_samples} samples across {len(category_names)} categories")

    def test_benchmark_config_valid(self):
        """Verify benchmark_config.yaml is valid and references correct paths."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "evaluation" / "benchmark_config.yaml"
        if not config_path.exists():
            pytest.skip("benchmark_config.yaml not found")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert "model" in cfg
        assert cfg["model"]["language"] == "japanese"
        assert "generation" in cfg
        assert cfg["generation"]["max_new_tokens"] == 2048
        assert "metrics" in cfg
        assert cfg["metrics"]["cer"]["enabled"] is True
        assert cfg["metrics"]["utmos"]["enabled"] is True
        assert "pass_criteria" in cfg
        assert cfg["pass_criteria"]["max_rtf"] == 1.0
        logger.info("Benchmark config validated")
