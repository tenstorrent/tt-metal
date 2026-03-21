# SPDX-License-Identifier: MIT

import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import ttnn
from pathlib import Path
import argparse
import time


def load_speaker_embeddings():
    """Load CMU Arctic speaker embeddings dataset"""
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    return embeddings_dataset


def preprocess_audio(audio_path, target_sr=16000):
    """Load and preprocess audio to 16kHz mono"""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio


def setup_ttnn_device():
    """Initialize TTNN device"""
    device = ttnn.open_device(device_id=0)
    return device


def load_pytorch_model():
    """Load PyTorch SpeechT5 voice conversion model and vocoder"""
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    model.eval()
    vocoder.eval()
    
    return processor, model, vocoder


def pytorch_inference(processor, model, vocoder, audio, speaker_embedding):
    """Run PyTorch reference inference"""
    inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_values"], 
            speaker_embedding, 
            vocoder=vocoder
        )
    
    return speech.cpu().numpy()


def ttnn_inference(device, audio, speaker_embedding):
    """Run TTNN inference (placeholder for actual implementation)"""
    # This will be implemented with actual TTNN model conversion
    # For now, return dummy output matching expected shape
    dummy_output = np.random.randn(len(audio) * 2)  # Placeholder
    return dummy_output


def compute_metrics(pytorch_output, ttnn_output):
    """Compute similarity metrics between outputs"""
    # Ensure same length for comparison
    min_len = min(len(pytorch_output), len(ttnn_output))
    pytorch_output = pytorch_output[:min_len]
    ttnn_output = ttnn_output[:min_len]
    
    # Mean squared error
    mse = np.mean((pytorch_output - ttnn_output) ** 2)
    
    # Cosine similarity
    cos_sim = np.dot(pytorch_output, ttnn_output) / (
        np.linalg.norm(pytorch_output) * np.linalg.norm(ttnn_output)
    )
    
    # Signal-to-noise ratio
    signal_power = np.mean(pytorch_output ** 2)
    noise_power = np.mean((pytorch_output - ttnn_output) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "snr_db": snr
    }


def main():
    parser = argparse.ArgumentParser(description="SpeechT5 Voice Conversion Demo")
    parser.add_argument("--input_audio", type=str, required=True, 
                       help="Path to input audio file")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for generated audio")
    parser.add_argument("--speaker_idx", type=int, default=0,
                       help="Speaker index from embeddings dataset")
    parser.add_argument("--run_pytorch", action="store_true",
                       help="Run PyTorch reference")
    parser.add_argument("--run_ttnn", action="store_true", 
                       help="Run TTNN implementation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading audio and speaker embeddings...")
    audio = preprocess_audio(args.input_audio)
    embeddings_dataset = load_speaker_embeddings()
    speaker_embedding = torch.tensor(embeddings_dataset[args.speaker_idx]["xvector"]).unsqueeze(0)
    
    print(f"Audio shape: {audio.shape}, Speaker embedding shape: {speaker_embedding.shape}")
    
    pytorch_output = None
    ttnn_output = None
    
    if args.run_pytorch:
        print("Loading PyTorch model...")
        processor, model, vocoder = load_pytorch_model()
        
        print("Running PyTorch inference...")
        start_time = time.time()
        pytorch_output = pytorch_inference(processor, model, vocoder, audio, speaker_embedding)
        pytorch_time = time.time() - start_time
        
        print(f"PyTorch inference time: {pytorch_time:.3f}s")
        print(f"PyTorch output shape: {pytorch_output.shape}")
        
        # Save PyTorch output
        sf.write(output_dir / "pytorch_output.wav", pytorch_output, 16000)
        print(f"Saved PyTorch output to {output_dir}/pytorch_output.wav")
    
    if args.run_ttnn:
        print("Setting up TTNN device...")
        device = setup_ttnn_device()
        
        print("Running TTNN inference...")
        start_time = time.time()
        ttnn_output = ttnn_inference(device, audio, speaker_embedding)
        ttnn_time = time.time() - start_time
        
        print(f"TTNN inference time: {ttnn_time:.3f}s")
        print(f"TTNN output shape: {ttnn_output.shape}")
        
        # Save TTNN output
        sf.write(output_dir / "ttnn_output.wav", ttnn_output, 16000)
        print(f"Saved TTNN output to {output_dir}/ttnn_output.wav")
        
        ttnn.close_device(device)
    
    # Compare outputs if both were generated
    if pytorch_output is not None and ttnn_output is not None:
        print("\nComputing quality metrics...")
        metrics = compute_metrics(pytorch_output, ttnn_output)
        
        print(f"Mean Squared Error: {metrics['mse']:.6f}")
        print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        print(f"Signal-to-Noise Ratio: {metrics['snr_db']:.2f} dB")
        
        if args.run_pytorch:
            speedup = pytorch_time / ttnn_time
            print(f"TTNN speedup: {speedup:.2f}x")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()