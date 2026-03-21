import os
import tempfile
import pytest
import torch
import numpy as np
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.t3000.speecht5.tt.speecht5_voice_conversion import TtSpeechT5VoiceConversion
from models.demos.t3000.speecht5.reference.speecht5_voice_conversion import SpeechT5VoiceConversionReference


def load_test_audio_data():
    """Load sample audio data for testing"""
    # Generate synthetic audio data for testing (16kHz, 3 seconds)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple sine wave with some harmonics
    audio_data = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 220 * t)    # A3 note
    )
    # Add some noise for realism
    audio_data += 0.1 * np.random.normal(0, 1, len(audio_data))
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    return torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)


def load_speaker_embeddings():
    """Load test speaker embeddings (x-vectors)"""
    # Standard x-vector dimension is 512
    embedding_dim = 512
    
    # Create two different speaker embeddings
    source_embedding = torch.randn(1, embedding_dim, dtype=torch.float32)
    target_embedding = torch.randn(1, embedding_dim, dtype=torch.float32)
    
    # Normalize embeddings (typical for x-vectors)
    source_embedding = source_embedding / torch.norm(source_embedding, dim=1, keepdim=True)
    target_embedding = target_embedding / torch.norm(target_embedding, dim=1, keepdim=True)
    
    return source_embedding, target_embedding


def compute_audio_quality_metrics(pred_audio, ref_audio):
    """Compute basic audio quality metrics"""
    if pred_audio.shape != ref_audio.shape:
        min_len = min(pred_audio.shape[-1], ref_audio.shape[-1])
        pred_audio = pred_audio[..., :min_len]
        ref_audio = ref_audio[..., :min_len]
    
    # Mean Squared Error
    mse = torch.mean((pred_audio - ref_audio) ** 2)
    
    # Signal-to-Noise Ratio (SNR)
    signal_power = torch.mean(ref_audio ** 2)
    noise_power = torch.mean((pred_audio - ref_audio) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    # Pearson Correlation Coefficient
    pred_flat = pred_audio.flatten()
    ref_flat = ref_audio.flatten()
    correlation = torch.corrcoef(torch.stack([pred_flat, ref_flat]))[0, 1]
    
    return {
        'mse': mse.item(),
        'snr': snr.item(), 
        'correlation': correlation.item()
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestSpeechT5VoiceConversion:
    
    def test_encoder_output_comparison(self, device):
        """Test SpeechT5 encoder output matches reference implementation"""
        logger.info("Testing SpeechT5 encoder output comparison")
        
        # Load test data
        input_audio = load_test_audio_data()
        source_embedding, _ = load_speaker_embeddings()
        
        # Initialize models
        reference_model = SpeechT5VoiceConversionReference()
        tt_model = TtSpeechT5VoiceConversion(device=device)
        
        # Run reference model encoder
        with torch.no_grad():
            ref_encoder_out = reference_model.encode_speech(input_audio, source_embedding)
        
        # Convert inputs to TTNN tensors
        tt_audio = ttnn.from_torch(input_audio, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_embedding = ttnn.from_torch(source_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
        # Run TT model encoder
        tt_encoder_out = tt_model.encode_speech(tt_audio, tt_embedding)
        tt_encoder_torch = ttnn.to_torch(tt_encoder_out)
        
        # Compare outputs
        assert_with_pcc(ref_encoder_out, tt_encoder_torch, pcc=0.98)
        logger.info(f"Encoder output shape: {ref_encoder_out.shape}")
        
    def test_decoder_output_comparison(self, device):
        """Test SpeechT5 decoder output matches reference implementation"""
        logger.info("Testing SpeechT5 decoder output comparison")
        
        # Load test data
        input_audio = load_test_audio_data()
        source_embedding, target_embedding = load_speaker_embeddings()
        
        # Initialize models
        reference_model = SpeechT5VoiceConversionReference()
        tt_model = TtSpeechT5VoiceConversion(device=device)
        
        # Get encoder outputs first
        with torch.no_grad():
            ref_encoder_out = reference_model.encode_speech(input_audio, source_embedding)
            ref_decoder_out = reference_model.decode_speech(ref_encoder_out, target_embedding)
        
        # Convert to TTNN tensors
        tt_encoder_out = ttnn.from_torch(ref_encoder_out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_target_embedding = ttnn.from_torch(target_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
        # Run TT model decoder
        tt_decoder_out = tt_model.decode_speech(tt_encoder_out, tt_target_embedding)
        tt_decoder_torch = ttnn.to_torch(tt_decoder_out)
        
        # Compare outputs
        assert_with_pcc(ref_decoder_out, tt_decoder_torch, pcc=0.96)
        logger.info(f"Decoder output shape: {ref_decoder_out.shape}")

    def test_end_to_end_voice_conversion(self, device):
        """Test complete voice conversion pipeline"""
        logger.info("Testing end-to-end voice conversion pipeline")
        
        # Load test data
        input_audio = load_test_audio_data()
        source_embedding, target_embedding = load_speaker_embeddings()
        
        # Initialize models
        reference_model = SpeechT5VoiceConversionReference()
        tt_model = TtSpeechT5VoiceConversion(device=device)
        
        # Run reference model
        with torch.no_grad():
            ref_converted_audio = reference_model.convert_voice(
                input_audio, source_embedding, target_embedding
            )
        
        # Convert inputs to TTNN tensors
        tt_audio = ttnn.from_torch(input_audio, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_source = ttnn.from_torch(source_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_target = ttnn.from_torch(target_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
        # Run TT model
        tt_converted_audio = tt_model.convert_voice(tt_audio, tt_source, tt_target)
        tt_audio_torch = ttnn.to_torch(tt_converted_audio)
        
        # Compare outputs
        assert_with_pcc(ref_converted_audio, tt_audio_torch, pcc=0.95)
        
        # Compute audio quality metrics
        metrics = compute_audio_quality_metrics(tt_audio_torch, ref_converted_audio)
        logger.info(f"Audio quality metrics: {metrics}")
        
        # Assert quality thresholds
        assert metrics['snr'] > 15.0, f"SNR too low: {metrics['snr']}"
        assert metrics['correlation'] > 0.8, f"Correlation too low: {metrics['correlation']}"
        
    def test_different_sequence_lengths(self, device):
        """Test model with different audio sequence lengths"""
        logger.info("Testing different sequence lengths")
        
        sequence_lengths = [1.0, 2.5, 4.0]  # seconds
        source_embedding, target_embedding = load_speaker_embeddings()
        
        for duration in sequence_lengths:
            logger.info(f"Testing duration: {duration}s")
            
            # Generate audio of specific duration
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
            input_audio = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            
            # Initialize models
            reference_model = SpeechT5VoiceConversionReference()
            tt_model = TtSpeechT5VoiceConversion(device=device)
            
            # Test inference
            with torch.no_grad():
                ref_output = reference_model.convert_voice(
                    input_audio, source_embedding, target_embedding
                )
            
            tt_audio = ttnn.from_torch(input_audio, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tt_source = ttnn.from_torch(source_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tt_target = ttnn.from_torch(target_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            
            tt_output = tt_model.convert_voice(tt_audio, tt_source, tt_target)
            tt_output_torch = ttnn.to_torch(tt_output)
            
            assert_with_pcc(ref_output, tt_output_torch, pcc=0.93)
            
    def test_batch_processing(self, device):
        """Test model with batch processing"""
        logger.info("Testing batch processing")
        
        batch_size = 2
        
        # Generate batch of audio data
        input_audios = []
        for i in range(batch_size):
            sample_rate = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Different frequency for each sample
            freq = 440 * (i + 1)
            audio_data = 0.5 * np.sin(2 * np.pi * freq * t)
            input_audios.append(audio_data)
        
        input_batch = torch.tensor(input_audios, dtype=torch.float32)
        
        # Generate batch of embeddings
        source_embeddings = torch.randn(batch_size, 512, dtype=torch.float32)
        target_embeddings = torch.randn(batch_size, 512, dtype=torch.float32)
        
        # Normalize embeddings
        source_embeddings = source_embeddings / torch.norm(source_embeddings, dim=1, keepdim=True)
        target_embeddings = target_embeddings / torch.norm(target_embeddings, dim=1, keepdim=True)
        
        # Initialize models
        reference_model = SpeechT5VoiceConversionReference()
        tt_model = TtSpeechT5VoiceConversion(device=device)
        
        # Test batch inference
        with torch.no_grad():
            ref_outputs = reference_model.convert_voice_batch(
                input_batch, source_embeddings, target_embeddings
            )
        
        tt_batch = ttnn.from_torch(input_batch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_source_batch = ttnn.from_torch(source_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_target_batch = ttnn.from_torch(target_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
        tt_outputs = tt_model.convert_voice_batch(tt_batch, tt_source_batch, tt_target_batch)
        tt_outputs_torch = ttnn.to_torch(tt_outputs)
        
        assert_with_pcc(ref_outputs, tt_outputs_torch, pcc=0.94)
        logger.info(f"Batch output shape: {ref_outputs.shape}")

    def test_performance_benchmarking(self, device):
        """Test model performance and throughput"""
        logger.info("Testing performance benchmarking")
        
        import time
        
        # Test data
        input_audio = load_test_audio_data()
        source_embedding, target_embedding = load_speaker_embeddings()
        
        # Initialize TT model
        tt_model = TtSpeechT5VoiceConversion(device=device)
        
        # Convert to TTNN tensors
        tt_audio = ttnn.from_torch(input_audio, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_source = ttnn.from_torch(source_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_target = ttnn.from_torch(target_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
        # Warmup runs
        for _ in range(3):
            _ = tt_model.convert_voice(tt_audio, tt_source, tt_target)
        
        # Benchmark runs
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            output = tt_model.convert_voice(tt_audio, tt_source, tt_target)
            # Ensure completion
            _ = ttnn.to_torch(output)
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        audio_duration = input_audio.shape[-1] / 16000  # assuming 16kHz
        real_time_factor = audio_duration / avg_time
        
        logger.info(f"Average inference time: {avg_time:.3f}s")
        logger.info(f"Audio duration: {audio_duration:.3f}s") 
        logger.info(f"Real-time factor: {real_time_factor:.2f}x")
        
        # Assert reasonable performance
        assert avg_time < 5.0, f"Inference too slow: {avg_time}s"
        
    def test_memory_usage(self, device):
        """Test memory usage and cleanup"""
        logger.info("Testing memory usage")
        
        # Test data
        input_audio = load_test_audio_data()
        source_embedding, target_embedding = load_speaker_embeddings()
        
        # Initialize model
        tt_model = TtSpeechT5VoiceConversion(device=device)
        
        # Multiple inference runs to check for memory leaks
        for i in range(5):
            tt_audio = ttnn.from_torch(input_audio, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tt_source = ttnn.from_torch(source_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tt_target = ttnn.from_torch(target_embedding, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            
            output = tt_model.convert_voice(tt_audio, tt_source, tt_target)
            result = ttnn.to_torch(output)
            
            # Clean up tensors
            ttnn.deallocate(tt_audio)
            ttnn.deallocate(tt_source)
            ttnn.deallocate(tt_target)
            ttnn.deallocate(output)
            
            logger.info(f"Memory test iteration {i+1}/5 completed")
        
        logger.info("Memory usage test completed successfully")