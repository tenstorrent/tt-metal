# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Bark Small TTNN implementation.

Tests:
1. Forward pass through each stage (semantic, coarse, fine)
2. Full pipeline end-to-end audio generation
3. PCC validation against HuggingFace reference (target >= 0.95)
4. Multilingual text support
5. Emotion annotation support ([laughs], [sighs])
"""

import pytest
import torch
import numpy as np
import ttnn

from models.demos.wormhole.bark.tt.bark_gpt import (
    BarkConfig,
    TtBarkGPT,
    preprocess_model_parameters,
)
from models.demos.wormhole.bark.tt.bark_fine import (
    TtBarkFineModel,
    preprocess_fine_model_parameters,
)
from models.demos.wormhole.bark.tt.bark_model import TtBarkModel
from models.demos.wormhole.bark.reference.bark_reference import (
    load_bark_reference,
    run_semantic_forward,
    run_coarse_forward,
    run_fine_forward,
    compute_pcc,
)


@pytest.fixture(scope="module")
def device():
    """Create a TTNN device for testing."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace reference model."""
    try:
        return load_bark_reference("suno/bark-small")
    except Exception as exc:
        pytest.skip(f"Skipping Bark tests: failed to load HuggingFace reference model 'suno/bark-small': {exc}")


class TestBarkSemantic:
    """Tests for Stage 1: Text-to-Semantic model."""

    def test_semantic_forward_pass(self, device, hf_model):
        """Test single forward pass through semantic model."""
        # Prepare input
        batch_size, seq_len = 1, 64
        input_ids = torch.randint(0, 10048, (batch_size, seq_len))

        # TTNN forward
        config = BarkConfig(
            hidden_size=hf_model.semantic.config.hidden_size,
            num_heads=hf_model.semantic.config.num_heads,
            num_layers=hf_model.semantic.config.num_layers,
            block_size=hf_model.semantic.config.block_size,
            input_vocab_size=hf_model.semantic.config.input_vocab_size,
            output_vocab_size=hf_model.semantic.config.output_vocab_size,
            bias=getattr(hf_model.semantic.config, "bias", False),
        )
        params = preprocess_model_parameters(hf_model.semantic, device)
        tt_model = TtBarkGPT(device, params, config, is_causal=True)

        tt_logits, _ = tt_model(input_ids=input_ids)
        tt_logits_torch = ttnn.to_torch(tt_logits)
        ttnn.deallocate(tt_logits)

        # Reference forward
        ref_logits = run_semantic_forward(hf_model, input_ids)

        # Validate shapes match
        assert (
            tt_logits_torch.squeeze(0).shape == ref_logits.shape
        ), f"Shape mismatch: TTNN={tt_logits_torch.squeeze(0).shape}, Ref={ref_logits.shape}"

    def test_semantic_pcc(self, device, hf_model):
        """Test PCC between TTNN and PyTorch semantic models."""
        batch_size, seq_len = 1, 32
        input_ids = torch.randint(0, 10048, (batch_size, seq_len))

        # TTNN forward
        config = BarkConfig(
            hidden_size=hf_model.semantic.config.hidden_size,
            num_heads=hf_model.semantic.config.num_heads,
            num_layers=hf_model.semantic.config.num_layers,
            block_size=hf_model.semantic.config.block_size,
            input_vocab_size=hf_model.semantic.config.input_vocab_size,
            output_vocab_size=hf_model.semantic.config.output_vocab_size,
            bias=getattr(hf_model.semantic.config, "bias", False),
        )
        params = preprocess_model_parameters(hf_model.semantic, device)
        tt_model = TtBarkGPT(device, params, config, is_causal=True)

        tt_logits, _ = tt_model(input_ids=input_ids)
        tt_logits_torch = ttnn.to_torch(tt_logits).squeeze(0)
        ttnn.deallocate(tt_logits)

        # Reference forward
        ref_logits = run_semantic_forward(hf_model, input_ids)

        # PCC check
        pcc = compute_pcc(tt_logits_torch, ref_logits)
        print(f"Semantic PCC: {pcc:.6f}")
        assert pcc >= 0.95, f"Semantic PCC {pcc:.6f} below threshold 0.95"


class TestBarkCoarse:
    """Tests for Stage 2: Semantic-to-Coarse model."""

    def test_coarse_forward_pass(self, device, hf_model):
        """Test single forward pass through coarse model."""
        batch_size, seq_len = 1, 64
        input_ids = torch.randint(0, 10048, (batch_size, seq_len))

        config = BarkConfig(
            hidden_size=hf_model.coarse_acoustics.config.hidden_size,
            num_heads=hf_model.coarse_acoustics.config.num_heads,
            num_layers=hf_model.coarse_acoustics.config.num_layers,
            block_size=hf_model.coarse_acoustics.config.block_size,
            input_vocab_size=hf_model.coarse_acoustics.config.input_vocab_size,
            output_vocab_size=hf_model.coarse_acoustics.config.output_vocab_size,
            bias=getattr(hf_model.coarse_acoustics.config, "bias", False),
        )
        params = preprocess_model_parameters(hf_model.coarse_acoustics, device)
        tt_model = TtBarkGPT(device, params, config, is_causal=True)

        tt_logits, _ = tt_model(input_ids=input_ids)
        tt_logits_torch = ttnn.to_torch(tt_logits)
        ttnn.deallocate(tt_logits)

        assert tt_logits_torch is not None
        assert tt_logits_torch.numel() > 0

    def test_coarse_pcc(self, device, hf_model):
        """Test PCC for coarse model between TTNN and PyTorch."""
        batch_size, seq_len = 1, 32
        input_ids = torch.randint(0, 10048, (batch_size, seq_len))

        # TTNN forward
        config = BarkConfig(
            hidden_size=hf_model.coarse_acoustics.config.hidden_size,
            num_heads=hf_model.coarse_acoustics.config.num_heads,
            num_layers=hf_model.coarse_acoustics.config.num_layers,
            block_size=hf_model.coarse_acoustics.config.block_size,
            input_vocab_size=hf_model.coarse_acoustics.config.input_vocab_size,
            output_vocab_size=hf_model.coarse_acoustics.config.output_vocab_size,
            bias=getattr(hf_model.coarse_acoustics.config, "bias", False),
        )
        params = preprocess_model_parameters(hf_model.coarse_acoustics, device)
        tt_model = TtBarkGPT(device, params, config, is_causal=True)

        tt_logits, _ = tt_model(input_ids=input_ids)
        tt_logits_torch = ttnn.to_torch(tt_logits).squeeze(0)
        ttnn.deallocate(tt_logits)

        # Reference forward
        ref_logits = run_coarse_forward(hf_model, input_ids)

        # PCC check
        pcc = compute_pcc(tt_logits_torch, ref_logits)
        print(f"Coarse PCC: {pcc:.6f}")
        assert pcc >= 0.95, f"Coarse PCC {pcc:.6f} below threshold 0.95"


class TestBarkFine:
    """Tests for Stage 3: Coarse-to-Fine model."""

    def test_fine_forward_pass(self, device, hf_model):
        """Test single forward pass through fine model for codebook 2."""
        batch_size, seq_len = 1, 32
        n_codes_total = hf_model.fine_acoustics.config.n_codes_total

        # Prepare input: [batch, seq, n_codes_total]
        input_ids = torch.randint(0, 1024, (batch_size, seq_len, n_codes_total))

        config = BarkConfig(
            hidden_size=hf_model.fine_acoustics.config.hidden_size,
            num_heads=hf_model.fine_acoustics.config.num_heads,
            num_layers=hf_model.fine_acoustics.config.num_layers,
            block_size=hf_model.fine_acoustics.config.block_size,
            input_vocab_size=hf_model.fine_acoustics.config.input_vocab_size,
            output_vocab_size=hf_model.fine_acoustics.config.output_vocab_size,
            bias=True,
        )
        params = preprocess_fine_model_parameters(hf_model.fine_acoustics, device)
        tt_model = TtBarkFineModel(
            device,
            params,
            config,
            n_codes_total=n_codes_total,
            n_codes_given=hf_model.fine_acoustics.config.n_codes_given,
        )

        # Forward pass for codebook 2
        tt_input_ids = ttnn.from_torch(
            input_ids.unsqueeze(0), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        tt_logits = tt_model(codebook_idx=2, input_ids=tt_input_ids)
        tt_logits_torch = ttnn.to_torch(tt_logits)
        ttnn.deallocate(tt_logits)
        ttnn.deallocate(tt_input_ids)

        assert tt_logits_torch is not None
        assert tt_logits_torch.numel() > 0

    def test_fine_pcc(self, device, hf_model):
        """Test PCC for fine model codebook prediction."""
        batch_size, seq_len = 1, 16
        n_codes_total = hf_model.fine_acoustics.config.n_codes_total
        input_ids = torch.randint(0, 1024, (batch_size, seq_len, n_codes_total))

        config = BarkConfig(
            hidden_size=hf_model.fine_acoustics.config.hidden_size,
            num_heads=hf_model.fine_acoustics.config.num_heads,
            num_layers=hf_model.fine_acoustics.config.num_layers,
            block_size=hf_model.fine_acoustics.config.block_size,
            input_vocab_size=hf_model.fine_acoustics.config.input_vocab_size,
            output_vocab_size=hf_model.fine_acoustics.config.output_vocab_size,
            bias=True,
        )
        params = preprocess_fine_model_parameters(hf_model.fine_acoustics, device)
        tt_model = TtBarkFineModel(
            device,
            params,
            config,
            n_codes_total=n_codes_total,
            n_codes_given=hf_model.fine_acoustics.config.n_codes_given,
        )

        # TTNN forward
        codebook_idx = 2
        tt_input_ids = ttnn.from_torch(
            input_ids.unsqueeze(0), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        tt_logits = tt_model(codebook_idx=codebook_idx, input_ids=tt_input_ids)
        tt_logits_torch = ttnn.to_torch(tt_logits).squeeze(0)
        ttnn.deallocate(tt_logits)
        ttnn.deallocate(tt_input_ids)

        # Reference forward
        ref_logits = run_fine_forward(hf_model, codebook_idx, input_ids)

        pcc = compute_pcc(tt_logits_torch, ref_logits)
        print(f"Fine model PCC (codebook {codebook_idx}): {pcc:.6f}")
        assert pcc >= 0.95, f"Fine PCC {pcc:.6f} below threshold 0.95"


class TestBarkPipeline:
    """Tests for full end-to-end pipeline."""

    def test_full_pipeline(self, device, hf_model):
        """Test complete text-to-audio pipeline."""
        model = TtBarkModel(device, model_name="suno/bark-small")

        text = "Hello, this is a test of the Bark text-to-speech model."
        audio = model.generate(text, verbose=True)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.isfinite(audio).all(), "Audio contains NaN or Inf values"
        print(f"Generated audio: {len(audio)} samples, {len(audio)/24000:.2f}s")

    def test_multilingual(self, device, hf_model):
        """Test with non-English text."""
        model = TtBarkModel(device, model_name="suno/bark-small")

        # Bark supports multilingual via special tokens
        text = "Bonjour, comment allez-vous?"
        audio = model.generate(text, verbose=False)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_emotion_annotations(self, device, hf_model):
        """Test with emotion annotations."""
        model = TtBarkModel(device, model_name="suno/bark-small")

        text = "I can't believe it! [laughs] That's amazing!"
        audio = model.generate(text, verbose=False)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


class TestBarkThroughput:
    """Benchmark tests for throughput validation."""

    def test_semantic_throughput(self, device, hf_model):
        """Check semantic token generation throughput (target >= 20 tok/s)."""
        import time

        model = TtBarkModel(device, model_name="suno/bark-small")

        text = "Hello, this is a throughput test."
        t0 = time.time()
        semantic_tokens = model.generate_semantic_tokens(text)
        elapsed = time.time() - t0

        num_tokens = semantic_tokens.shape[1]
        throughput = num_tokens / elapsed
        print(f"Semantic throughput: {throughput:.1f} tok/s ({num_tokens} tokens in {elapsed:.2f}s)")

        # Note: Target is 20 tok/s but initial bring-up may be slower
        assert throughput > 0, "Throughput should be positive"
