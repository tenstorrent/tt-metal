#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SpeechT5 Performance Test

This test measures the performance of SpeechT5 TTNN components with L1 memory optimization.
It captures detailed performance metrics for encoder, decoder, and postnet components.

⚠️  IMPORTANT: These are SYNTHETIC benchmark results using random data under ideal conditions.
   Real-world demo performance is significantly slower due to:
   - Full autoregressive generation with sequence building
   - Memory management overhead (ensure_l1_memory calls)
   - Dynamic tensor operations and allocations
   - Causal mask computation for growing sequences

Synthetic Benchmark Results (L1 Optimized):
- Encoder: ~50 inf/sec (0.020s per inference)
- Decoder Step: ~60 inf/sec (0.017s per inference) [SYNTHETIC - not real demo performance]
- Postnet: ~346 inf/sec (0.003s per inference)
- Full Pipeline: ~59 tokens/sec (0.43x real-time factor) [SYNTHETIC]

Real Demo Performance (from actual measurements):
- Decoder Step: ~5.7s per step (much slower due to full pipeline overhead)
- Complete generation includes all TTNN operations + Python overhead
"""

import sys
import pytest
import torch
import time
import ttnn
from loguru import logger

from models.common.utility_functions import profiler, run_for_wormhole_b0

# Import SpeechT5 components
sys.path.append("/home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal")
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNSpeechT5Encoder,
    TTNNEncoderConfig,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNSpeechT5Decoder,
    TTNNDecoderConfig,
    preprocess_decoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNSpeechT5SpeechDecoderPostnet,
    TTNNPostNetConfig,
    preprocess_postnet_parameters,
)


class SpeechT5PerformanceTest:
    """
    Performance testing infrastructure for SpeechT5 TTNN components.
    Measures inference throughput with L1 memory optimization.
    """

    def __init__(self, device):
        self.device = device
        self._setup_models()

    def _setup_models(self):
        """Setup TTNN SpeechT5 components."""
        logger.info("Setting up SpeechT5 TTNN components for performance testing...")

        # Load HF model for config
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

        # Setup Encoder
        encoder_config = TTNNEncoderConfig(
            vocab_size=self.hf_model.config.vocab_size,
            hidden_size=self.hf_model.config.hidden_size,
            num_layers=self.hf_model.config.encoder_layers,
            num_heads=self.hf_model.config.encoder_attention_heads,
            ffn_dim=self.hf_model.config.encoder_ffn_dim,
            max_position_embeddings=self.hf_model.config.max_length,
            layer_norm_eps=self.hf_model.config.layer_norm_eps,
        )
        self.encoder = TTNNSpeechT5Encoder(
            self.device,
            preprocess_encoder_parameters(self.hf_model.speecht5.encoder, encoder_config, self.device),
            encoder_config,
        )

        # Setup Decoder
        decoder_config = TTNNDecoderConfig(
            hidden_size=self.hf_model.config.hidden_size,
            num_layers=self.hf_model.config.decoder_layers,
            num_heads=self.hf_model.config.decoder_attention_heads,
            ffn_dim=self.hf_model.config.decoder_ffn_dim,
            max_position_embeddings=self.hf_model.config.max_length,
            layer_norm_eps=self.hf_model.config.layer_norm_eps,
            num_mel_bins=self.hf_model.config.num_mel_bins,
            reduction_factor=self.hf_model.config.reduction_factor,
            speech_decoder_prenet_units=self.hf_model.config.speech_decoder_prenet_units,
            speech_decoder_prenet_layers=self.hf_model.config.speech_decoder_prenet_layers,
            speech_decoder_prenet_dropout=self.hf_model.config.speech_decoder_prenet_dropout,
            speaker_embedding_dim=self.hf_model.config.speaker_embedding_dim,
        )
        self.decoder = TTNNSpeechT5Decoder(
            self.device,
            preprocess_decoder_parameters(self.hf_model.speecht5.decoder, decoder_config, self.device),
            decoder_config,
        )

        # Setup Postnet
        postnet_config = TTNNPostNetConfig(
            hidden_size=self.hf_model.config.hidden_size,
            num_mel_bins=self.hf_model.config.num_mel_bins,
            reduction_factor=self.hf_model.config.reduction_factor,
            postnet_layers=self.hf_model.config.speech_decoder_postnet_layers,
            postnet_units=self.hf_model.config.speech_decoder_postnet_units,
            postnet_kernel=self.hf_model.config.speech_decoder_postnet_kernel,
        )
        self.postnet = TTNNSpeechT5SpeechDecoderPostnet(
            self.device,
            preprocess_postnet_parameters(self.hf_model.speech_decoder_postnet, postnet_config, self.device),
            postnet_config,
        )

        logger.info("SpeechT5 TTNN components setup complete")

    def warmup(self, num_warmup=5):
        """Warmup runs to stabilize performance measurements."""
        logger.info(f"Running {num_warmup} warmup iterations...")

        for i in range(num_warmup):
            # Encoder warmup
            input_ids = torch.randint(0, 100, (1, 50), dtype=torch.int32)
            ttnn_input = ttnn.from_torch(
                input_ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            encoder_out = self.encoder(ttnn_input)
            ttnn.deallocate(ttnn_input)

            # Decoder warmup
            decoder_input = torch.randn(1, 1, self.hf_model.config.num_mel_bins)
            encoder_hidden = torch.randn(1, 50, self.hf_model.config.hidden_size)
            speaker_emb = torch.randn(self.hf_model.config.speaker_embedding_dim)

            ttnn_decoder_input = ttnn.from_torch(
                decoder_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn_encoder_hidden = ttnn.from_torch(
                encoder_hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn_speaker_emb = ttnn.from_torch(
                speaker_emb,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            decoder_out = self.decoder(ttnn_decoder_input, ttnn_encoder_hidden, ttnn_speaker_emb)

            # Postnet warmup
            postnet_out = self.postnet(decoder_out)

            # Cleanup
            ttnn.deallocate(ttnn_decoder_input)
            ttnn.deallocate(ttnn_encoder_hidden)
            ttnn.deallocate(ttnn_speaker_emb)

        logger.info("Warmup complete")

    def benchmark_encoder(self, num_iterations=100):
        """Benchmark encoder performance."""
        logger.info(f"Benchmarking encoder with {num_iterations} iterations...")

        times = []
        for i in range(num_iterations):
            input_ids = torch.randint(0, 100, (1, 50), dtype=torch.int32)
            ttnn_input = ttnn.from_torch(
                input_ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            start_time = time.time()
            encoder_out = self.encoder(ttnn_input)
            ttnn.synchronize_device(self.device)
            end_time = time.time()

            times.append(end_time - start_time)
            ttnn.deallocate(ttnn_input)

        avg_time = sum(times) / len(times)
        throughput = 1.0 / avg_time

        logger.info(".6f")
        logger.info(".2f")

        return {"component": "encoder", "avg_time": avg_time, "throughput": throughput, "iterations": num_iterations}

    def benchmark_decoder(self, num_iterations=100):
        """Benchmark decoder step performance (with prenet + full decoder processing)."""
        logger.info(f"Benchmarking decoder step with {num_iterations} iterations...")
        logger.info("NOTE: This measures isolated decoder performance with synthetic data.")
        logger.info("Real demo performance includes additional overhead (memory mgmt, sequence building, etc.)")

        times = []
        for i in range(num_iterations):
            decoder_input = torch.randn(1, 1, self.hf_model.config.num_mel_bins)
            encoder_hidden = torch.randn(1, 50, self.hf_model.config.hidden_size)
            speaker_emb = torch.randn(self.hf_model.config.speaker_embedding_dim)

            ttnn_decoder_input = ttnn.from_torch(
                decoder_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn_encoder_hidden = ttnn.from_torch(
                encoder_hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn_speaker_emb = ttnn.from_torch(
                speaker_emb,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            start_time = time.time()
            decoder_out = self.decoder(ttnn_decoder_input, ttnn_encoder_hidden, ttnn_speaker_emb)
            ttnn.synchronize_device(self.device)
            end_time = time.time()

            times.append(end_time - start_time)

            # Cleanup
            ttnn.deallocate(ttnn_decoder_input)
            ttnn.deallocate(ttnn_encoder_hidden)
            ttnn.deallocate(ttnn_speaker_emb)

        avg_time = sum(times) / len(times)
        throughput = 1.0 / avg_time

        logger.info(".6f")
        logger.info(".2f")
        logger.info("⚠️  WARNING: This is synthetic benchmark performance.")
        logger.info("   Real demo performance is much slower due to:")
        logger.info("   - Full autoregressive loop with sequence building")
        logger.info("   - Memory management overhead (ensure_l1_memory calls)")
        logger.info("   - Dynamic tensor operations and allocations")
        logger.info("   - Causal mask computation for growing sequences")

        return {"component": "decoder", "avg_time": avg_time, "throughput": throughput, "iterations": num_iterations}

    def benchmark_postnet(self, num_iterations=100):
        """Benchmark postnet performance."""
        logger.info(f"Benchmarking postnet with {num_iterations} iterations...")

        times = []
        for i in range(num_iterations):
            decoder_input = torch.randn(1, 1, self.hf_model.config.num_mel_bins)
            encoder_hidden = torch.randn(1, 50, self.hf_model.config.hidden_size)
            speaker_emb = torch.randn(self.hf_model.config.speaker_embedding_dim)

            ttnn_decoder_input = ttnn.from_torch(
                decoder_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn_encoder_hidden = ttnn.from_torch(
                encoder_hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn_speaker_emb = ttnn.from_torch(
                speaker_emb,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            decoder_out = self.decoder(ttnn_decoder_input, ttnn_encoder_hidden, ttnn_speaker_emb)

            start_time = time.time()
            postnet_out = self.postnet(decoder_out)
            ttnn.synchronize_device(self.device)
            end_time = time.time()

            times.append(end_time - start_time)

            # Cleanup
            ttnn.deallocate(ttnn_decoder_input)
            ttnn.deallocate(ttnn_encoder_hidden)
            ttnn.deallocate(ttnn_speaker_emb)

        avg_time = sum(times) / len(times)
        throughput = 1.0 / avg_time

        logger.info(".6f")
        logger.info(".2f")

        return {"component": "postnet", "avg_time": avg_time, "throughput": throughput, "iterations": num_iterations}

    def benchmark_full_pipeline(self, num_tokens=50, num_iterations=10):
        """Benchmark full pipeline performance for autoregressive generation."""
        logger.info(f"Benchmarking full pipeline ({num_tokens} tokens, {num_iterations} iterations)...")

        times = []
        for iteration in range(num_iterations):
            # Encoder run
            input_ids = torch.randint(0, 100, (1, num_tokens), dtype=torch.int32)
            ttnn_input = ttnn.from_torch(
                input_ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            encoder_out = self.encoder(ttnn_input)[0]  # Encoder returns tuple, get first element
            ttnn.deallocate(ttnn_input)

            # Decoder autoregressive generation
            start_time = time.time()

            for token_idx in range(num_tokens):
                decoder_input = torch.randn(1, 1, self.hf_model.config.num_mel_bins)
                speaker_emb = torch.randn(self.hf_model.config.speaker_embedding_dim)

                ttnn_decoder_input = ttnn.from_torch(
                    decoder_input,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ttnn_speaker_emb = ttnn.from_torch(
                    speaker_emb,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

                decoder_out = self.decoder(ttnn_decoder_input, encoder_out, ttnn_speaker_emb)
                postnet_out = self.postnet(decoder_out)

                # Cleanup intermediate tensors
                ttnn.deallocate(ttnn_decoder_input)
                ttnn.deallocate(ttnn_speaker_emb)

            ttnn.synchronize_device(self.device)
            end_time = time.time()

            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        tokens_per_sec = num_tokens / avg_time
        real_time_factor = avg_time / 2.0  # Assuming 2s target audio

        logger.info(".4f")
        logger.info(".2f")
        logger.info(".2f")

        return {
            "component": "full_pipeline",
            "avg_time": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "real_time_factor": real_time_factor,
            "num_tokens": num_tokens,
            "iterations": num_iterations,
        }


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_speecht5_full_pipeline_performance_2_tokens(device):
    """Test SpeechT5 TTNN full pipeline performance with 2-token generation.

    NOTE: This measures synthetic performance with random data.
    Real demo performance (demo_ttnn.py) is much slower due to full pipeline overhead.
    """

    profiler.clear()

    # Setup performance test
    perf_test = SpeechT5PerformanceTest(device)

    # Warmup
    perf_test.warmup(num_warmup=2)

    # Benchmark components
    results = {}

    # Test only full pipeline with 2 tokens (synthetic performance)
    results["full_pipeline"] = perf_test.benchmark_full_pipeline(num_tokens=2, num_iterations=5)

    # Log full pipeline results
    logger.info("\n" + "=" * 60)
    logger.info("SPEECHT5 SYNTHETIC FULL PIPELINE PERFORMANCE RESULTS (2 tokens)")
    logger.info("=" * 60)
    logger.info("⚠️  These are SYNTHETIC benchmark results with random data.")
    logger.info("   Real demo performance is 100x+ slower due to pipeline overhead.")

    full_pipeline_data = results["full_pipeline"]
    logger.info(
        f"SYNTHETIC_FULL_PIPELINE: {full_pipeline_data['tokens_per_sec']:.2f} tokens/sec, "
        f"{full_pipeline_data['real_time_factor']:.2f}x RTF, {full_pipeline_data['avg_time']:.4f}s"
    )

    logger.info("=" * 60)

    # Performance assertions for 2-token generation
    assert (
        results["full_pipeline"]["tokens_per_sec"] > 0.1
    ), f"Full pipeline tokens/sec too low: {results['full_pipeline']['tokens_per_sec']}"

    # Store results for external analysis
    import json

    results_file = "/tmp/speecht5_performance_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Performance results saved to {results_file}")

    # Return results for potential use by other tests
    return results


if __name__ == "__main__":
    # Allow running standalone for quick testing
    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        results = test_speecht5_full_pipeline_performance_2_tokens(device)
        print("\n2-token full pipeline performance test completed successfully!")
        print("Results:", results)
    finally:
        ttnn.close_device(device)
