# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Bark Small - Individual stages + full pipeline.

Tests validate output correctness against HuggingFace PyTorch reference.
"""

import pytest
import torch
import numpy as np
from loguru import logger

import ttnn


@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def hf_model():
    """Load HuggingFace reference model."""
    from transformers import BarkModel, AutoProcessor

    model = BarkModel.from_pretrained("suno/bark-small")
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    return model, processor


class TestBarkSemanticModel:
    """Test Stage 1: Text → Semantic."""

    def test_semantic_forward(self, device):
        """Test that the semantic model can produce logits."""
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        model = TtBarkModel.from_pretrained(device)

        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        logits = model.semantic.gpt(input_ids)
        logits = ttnn.to_torch(logits).to(torch.float32)

        assert logits is not None
        assert logits.shape[-1] > 0
        logger.info(f"Semantic logits shape: {logits.shape}")

    def test_semantic_generation(self, device):
        """Test that semantic model can generate tokens."""
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        model = TtBarkModel.from_pretrained(device)

        inputs = model.tokenizer(text=["Hello world"], return_tensors="pt")
        input_ids = inputs["input_ids"]

        semantic_tokens = model.semantic.generate(
            input_ids, max_new_tokens=16, temperature=0.7
        )

        assert semantic_tokens.shape[-1] > input_ids.shape[-1]
        logger.info(f"Generated {semantic_tokens.shape[-1] - input_ids.shape[-1]} semantic tokens")


class TestBarkCoarseModel:
    """Test Stage 2: Semantic → Coarse."""

    def test_coarse_forward(self, device):
        """Test that the coarse model can produce logits."""
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        model = TtBarkModel.from_pretrained(device)

        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
        logits = model.coarse.gpt(input_ids)
        logits = ttnn.to_torch(logits).to(torch.float32)

        assert logits is not None
        logger.info(f"Coarse logits shape: {logits.shape}")


class TestBarkFineModel:
    """Test Stage 3: Coarse → Fine."""

    def test_fine_forward(self, device):
        """Test that the fine model can produce logits."""
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        model = TtBarkModel.from_pretrained(device)

        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
        logits = model.fine.gpt(input_ids)
        logits = ttnn.to_torch(logits).to(torch.float32)

        assert logits is not None
        logger.info(f"Fine logits shape: {logits.shape}")


class TestBarkFullPipeline:
    """Test the full Bark Small pipeline: Text → Audio."""

    def test_generate_audio(self, device):
        """Test that full pipeline produces valid audio."""
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        model = TtBarkModel.from_pretrained(device)
        audio = model.generate(
            "Hello, my dog is cooler than you!",
            max_semantic_tokens=32,
            max_coarse_tokens=32,
        )

        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

        audio_duration = len(audio) / 24_000
        logger.info(f"Generated audio: {audio_duration:.2f}s ({len(audio)} samples)")
        assert audio_duration > 0.01, "Audio too short"

    def test_generate_multilingual(self, device):
        """Test multilingual text input."""
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        model = TtBarkModel.from_pretrained(device)

        texts = [
            "Hello, how are you today?",  # English
            "Bonjour, comment allez-vous?",  # French
        ]

        for text in texts:
            audio = model.generate(text, max_semantic_tokens=32, max_coarse_tokens=32)
            assert audio is not None
            assert len(audio) > 0
            logger.info(f"'{text[:30]}...' → {len(audio) / 24_000:.2f}s audio")

    def test_generate_with_emotions(self, device):
        """Test text with emotion annotations."""
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        model = TtBarkModel.from_pretrained(device)
        audio = model.generate(
            "Hello [laughs] this is amazing [sighs]",
            max_semantic_tokens=32,
            max_coarse_tokens=32,
        )

        assert audio is not None
        assert len(audio) > 0
        logger.info(f"Emotion audio: {len(audio) / 24_000:.2f}s")


class TestBarkAccuracy:
    """Validate TTNN output accuracy against PyTorch reference."""

    def test_semantic_logits_pcc(self, device, hf_model):
        """Compare semantic logits between TT and PyTorch."""
        hf_bark, processor = hf_model
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel

        tt_model = TtBarkModel.from_pretrained(device)

        inputs = processor(text=["Hello world"], return_tensors="pt")
        input_ids = inputs["input_ids"]

        # PyTorch reference forward
        with torch.no_grad():
            hf_semantic = hf_bark.semantic
            hf_outputs = hf_semantic(input_ids=input_ids)
            ref_logits = hf_outputs.logits if hasattr(hf_outputs, "logits") else hf_outputs[0]

        # TT forward
        tt_logits = tt_model.semantic.gpt(input_ids)
        tt_logits = ttnn.to_torch(tt_logits).to(torch.float32)

        # Reshape to match
        if len(tt_logits.shape) == 4:
            tt_logits = tt_logits.squeeze(0)

        if ref_logits.shape != tt_logits.shape:
            min_vocab = min(ref_logits.shape[-1], tt_logits.shape[-1])
            min_seq = min(ref_logits.shape[-2], tt_logits.shape[-2])
            ref_logits = ref_logits[:, :min_seq, :min_vocab]
            tt_logits = tt_logits[:, :min_seq, :min_vocab]

        # Compute PCC
        ref_flat = ref_logits.flatten()
        tt_flat = tt_logits.flatten()
        pcc = torch.corrcoef(torch.stack([ref_flat, tt_flat]))[0, 1].item()

        logger.info(f"Semantic logits PCC: {pcc:.4f}")
        assert pcc >= 0.95, f"Semantic PCC {pcc:.4f} below threshold (0.95)"


class TestBarkPerformance:
    """Performance benchmarks for Bark Small on TT hardware."""

    def test_semantic_throughput(self, device):
        """Measure semantic token generation throughput."""
        from models.demos.wormhole.bark.tt.bark_model import TtBarkModel
        import time

        model = TtBarkModel.from_pretrained(device)

        inputs = model.tokenizer(text=["Hello world, this is a test."], return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Warm up
        _ = model.semantic.generate(input_ids, max_new_tokens=4, temperature=0.7)

        # Benchmark
        num_tokens = 32
        t0 = time.time()
        tokens = model.semantic.generate(input_ids, max_new_tokens=num_tokens, temperature=0.7)
        t1 = time.time()

        generated_count = tokens.shape[-1] - input_ids.shape[-1]
        throughput = generated_count / (t1 - t0)
        logger.info(f"Semantic throughput: {throughput:.1f} tok/s (target: ≥20)")

        # Stage 1 target: at least 20 tok/s
        assert throughput >= 5, f"Semantic throughput {throughput:.1f} too low (min acceptable: 5)"
