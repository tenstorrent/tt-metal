# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS Demo Script

This script demonstrates how to:
1. Load Qwen3-TTS weights from HuggingFace
2. Initialize the TTNN model
3. Run inference with performance measurements

Usage:
    python models/demos/qwen3_tts/demo/demo.py --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base

Requirements:
    pip install transformers safetensors
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.demos.qwen3_tts.tt.generator import Qwen3TTSGenerator, create_generator
from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat
from models.demos.qwen3_tts.tt.speech_tokenizer import (
    SpeechTokenizerConfig,
    TtSpeechTokenizerDecoder,
    extract_speech_tokenizer_weights,
)


def load_hf_weights(model_id: str, cache_dir: Optional[str] = None) -> dict:
    """
    Load weights from HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        cache_dir: Optional cache directory

    Returns:
        State dict with model weights
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("Please install transformers and safetensors: pip install transformers safetensors")

    print(f"Loading model from HuggingFace: {model_id}")
    start_time = time.time()

    # Download model files
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )
    model_path = Path(model_path)

    # Load safetensors
    state_dict = {}
    for safetensor_file in model_path.glob("*.safetensors"):
        print(f"  Loading {safetensor_file.name}...")
        state_dict.update(load_file(safetensor_file))

    load_time = time.time() - start_time
    print(f"Loaded {len(state_dict)} weight tensors in {load_time:.2f}s")

    return state_dict


def load_speech_tokenizer_weights(model_id: str, cache_dir: Optional[str] = None) -> dict:
    """
    Load Speech Tokenizer Decoder weights from HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        cache_dir: Optional cache directory

    Returns:
        State dict with speech tokenizer decoder weights
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")

    from huggingface_hub import snapshot_download

    print(f"Loading speech tokenizer weights from: {model_id}")

    model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["speech_tokenizer/*.safetensors"],
    )
    model_path = Path(model_path)

    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    if not speech_tokenizer_path.exists():
        print(f"  Warning: Speech tokenizer weights not found at {speech_tokenizer_path}")
        return {}

    print(f"  Loading {speech_tokenizer_path.name}...")
    state_dict = load_file(speech_tokenizer_path)

    # Extract decoder weights (remove "decoder." prefix)
    decoder_weights = extract_speech_tokenizer_weights(state_dict)
    print(f"Loaded {len(decoder_weights)} speech tokenizer decoder weight tensors")

    return decoder_weights


def save_audio(audio: torch.Tensor, output_path: str, sample_rate: int = 24000):
    """
    Save audio tensor to WAV file.

    Args:
        audio: Audio tensor of shape [batch, 1, num_samples] or [batch, num_samples]
        output_path: Path to save the WAV file
        sample_rate: Audio sample rate (default 24000 Hz)
    """
    try:
        import scipy.io.wavfile as wavfile
    except ImportError:
        raise ImportError("Please install scipy: pip install scipy")

    # Ensure audio is 2D: [num_samples] or 1D
    audio = audio.squeeze()
    if audio.dim() > 1:
        audio = audio[0]  # Take first sample from batch

    # Normalize to [-1, 1] and convert to int16
    # Convert to float32 first since bfloat16 isn't supported by numpy
    audio = audio.detach().cpu().float().numpy()
    audio = (audio * 32767).astype("int16")

    wavfile.write(output_path, sample_rate, audio)
    print(f"Audio saved to: {output_path}")


def run_prefill(
    model: Qwen3TTS,
    input_ids: torch.Tensor,
    device,
    talker_config: Qwen3TTSTalkerConfig,
    code_predictor_config: Qwen3TTSCodePredictorConfig,
    talker_trans_mat,
    cp_trans_mat,
    use_text_embedding: bool = False,
) -> tuple:
    """Run prefill forward pass and return logits with timing."""
    seq_len = input_ids.shape[1]

    input_ids_ttnn = ttnn.from_torch(
        input_ids,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    position_ids = torch.arange(seq_len)
    talker_cos, talker_sin = get_rope_tensors(
        device, talker_config.head_dim, seq_len, position_ids, talker_config.rope_theta
    )
    cp_cos, cp_sin = get_rope_tensors(
        device, code_predictor_config.head_dim, seq_len, position_ids, code_predictor_config.rope_theta
    )

    # Time only the forward pass
    start_time = time.time()
    codec_logits, cp_logits_list, _, _ = model.forward(
        input_ids_ttnn,
        talker_cos,
        talker_sin,
        talker_trans_mat,
        cp_cos,
        cp_sin,
        cp_trans_mat,
        use_text_embedding=use_text_embedding,
    )
    ttnn.synchronize_device(device)
    inference_time = time.time() - start_time

    # Combine logits for compatibility
    logits_list = [codec_logits] + cp_logits_list
    return logits_list, inference_time


def run_prefill_traced(
    generator: Qwen3TTSGenerator,
    input_ids: torch.Tensor,
    device,
) -> tuple:
    """Run traced prefill forward pass and return logits with timing."""
    # Time only the trace execution (not warmup/capture)
    start_time = time.time()
    logits_list = generator.prefill(input_ids, use_trace=True)
    ttnn.synchronize_device(device)
    inference_time = time.time() - start_time

    return logits_list, inference_time


def run_decode_step(
    model: Qwen3TTS,
    input_ids: torch.Tensor,
    position: int,
    device,
    talker_config: Qwen3TTSTalkerConfig,
    code_predictor_config: Qwen3TTSCodePredictorConfig,
    talker_trans_mat,
    cp_trans_mat,
) -> tuple:
    """Run single decode step and return logits with timing."""
    input_ids_ttnn = ttnn.from_torch(
        input_ids,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    position_ids = torch.tensor([position])
    talker_cos, talker_sin = get_rope_tensors(device, talker_config.head_dim, 1, position_ids, talker_config.rope_theta)
    cp_cos, cp_sin = get_rope_tensors(
        device, code_predictor_config.head_dim, 1, position_ids, code_predictor_config.rope_theta
    )

    start_time = time.time()
    logits_list, _, _ = model.forward(
        input_ids_ttnn,
        talker_cos,
        talker_sin,
        talker_trans_mat,
        cp_cos,
        cp_sin,
        cp_trans_mat,
    )
    ttnn.synchronize_device(device)
    decode_time = time.time() - start_time

    return logits_list, decode_time


def run_demo(
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_id: int = 0,
    seq_len: int = 128,
    batch_size: int = 1,
    cache_dir: Optional[str] = None,
    weight_cache_path: Optional[str] = None,
    use_trace: bool = False,
    use_decode_trace: bool = False,
    num_decode_steps: int = 10,
    num_inference_runs: int = 3,
    generate_audio: bool = False,
    audio_output: str = "output.wav",
    text: Optional[str] = None,
):
    """
    Run Qwen3-TTS demo with performance measurements.

    Args:
        model_id: HuggingFace model ID
        device_id: TT device ID
        seq_len: Sequence length for test input
        batch_size: Batch size
        cache_dir: HuggingFace cache directory
        weight_cache_path: TTNN weight cache path
        use_trace: Enable prefill tracing for faster execution
        use_decode_trace: Enable decode tracing for faster token generation
        num_decode_steps: Number of decode steps to run for performance measurement
        num_inference_runs: Number of inference runs for averaging (excluding compile/warmup)
        generate_audio: Generate audio output using speech tokenizer decoder
        audio_output: Output path for generated audio WAV file
        text: Text to synthesize (required for TTS mode).
    """
    if text is None:
        raise ValueError("--text is required. Please provide text to synthesize.")

    print("=" * 80)
    print("Qwen3-TTS TTNN Demo")
    print("=" * 80)

    # Load HuggingFace weights
    state_dict = load_hf_weights(model_id, cache_dir)

    # Open TTNN device
    print(f"\nOpening TT device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        # Initialize configs
        talker_config = Qwen3TTSTalkerConfig()
        code_predictor_config = Qwen3TTSCodePredictorConfig()

        # Weight cache path
        if weight_cache_path:
            weight_cache_path = Path(weight_cache_path)
            weight_cache_path.mkdir(parents=True, exist_ok=True)

        # Initialize model
        print("\nInitializing Qwen3-TTS model...")
        model_init_start = time.time()

        model = Qwen3TTS(
            device=device,
            state_dict=state_dict,
            talker_config=talker_config,
            code_predictor_config=code_predictor_config,
            weight_cache_path=weight_cache_path,
        )

        model_init_time = time.time() - model_init_start
        print(f"Model initialized in {model_init_time:.2f}s")
        print(f"  Talker layers: {len(model.talker.layers)}")
        print(f"  CodePredictor layers: {len(model.code_predictor.layers)}")
        print(f"  CodePredictor LM heads: {len(model.code_predictor.lm_heads)}")

        # Pre-compute transformation matrices
        print("\nPre-computing transformation matrices...")
        talker_trans_mat = get_transformation_mat(talker_config.head_dim, device)
        cp_trans_mat = get_transformation_mat(code_predictor_config.head_dim, device)

        # Tokenize text input
        print(f"\nText: {text}")
        from transformers import AutoProcessor

        print("Loading tokenizer...")
        processor = AutoProcessor.from_pretrained(model_id)

        # Format text with special tokens (Qwen3-TTS format)
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=formatted_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        use_text_embedding = True
        print(f"Tokenized to {seq_len} tokens")

        # ============================================================
        # COMPILE / WARMUP PHASE (not included in perf numbers)
        # ============================================================
        print("\n" + "=" * 80)
        print("COMPILE / WARMUP PHASE")
        print("=" * 80)

        if use_trace:
            # Setup generator with tracing
            print("\nSetting up generator with tracing...")
            generator = create_generator(
                model=model,
                device=device,
                max_batch_size=batch_size,
                max_seq_len=seq_len * 2,
            )
            generator.setup()

            # Warmup compile all ops
            print("Warming up prefill (compiling ops)...")
            warmup_start = time.time()
            generator.warmup_prefill(seq_len=seq_len)
            warmup_time = time.time() - warmup_start
            print(f"  Warmup time: {warmup_time:.2f}s")

            # Capture trace
            print("Capturing prefill trace...")
            trace_start = time.time()
            generator.capture_prefill_trace(seq_len=seq_len)
            trace_time = time.time() - trace_start
            print(f"  Trace capture time: {trace_time:.2f}s")

            if use_decode_trace:
                print("Warming up decode...")
                generator.warmup_decode()
                print("Capturing decode trace...")
                generator.capture_decode_trace(start_pos=seq_len)

        else:
            # Non-traced warmup - run one forward pass to compile
            print("\nWarming up (compiling ops)...")
            warmup_start = time.time()
            _, _ = run_prefill(
                model,
                input_ids,
                device,
                talker_config,
                code_predictor_config,
                talker_trans_mat,
                cp_trans_mat,
                use_text_embedding=use_text_embedding,
            )
            warmup_time = time.time() - warmup_start
            print(f"  Warmup time: {warmup_time:.2f}s")

        print("\nCompile/warmup complete!")

        # ============================================================
        # INFERENCE PHASE (performance measurements)
        # ============================================================
        print("\n" + "=" * 80)
        print("INFERENCE PHASE")
        print("=" * 80)

        prefill_times = []
        decode_times = []

        for run in range(num_inference_runs):
            print(f"\n--- Run {run + 1}/{num_inference_runs} ---")

            # PREFILL (TTFT measurement)
            if use_trace:
                # Note: traced mode currently only supports codec embedding (benchmark mode)
                if use_text_embedding:
                    print("  Warning: Tracing not yet supported with text embedding, using non-traced mode")
                    logits_list, prefill_time = run_prefill(
                        model,
                        input_ids,
                        device,
                        talker_config,
                        code_predictor_config,
                        talker_trans_mat,
                        cp_trans_mat,
                        use_text_embedding=use_text_embedding,
                    )
                else:
                    logits_list, prefill_time = run_prefill_traced(generator, input_ids, device)
            else:
                logits_list, prefill_time = run_prefill(
                    model,
                    input_ids,
                    device,
                    talker_config,
                    code_predictor_config,
                    talker_trans_mat,
                    cp_trans_mat,
                    use_text_embedding=use_text_embedding,
                )
            prefill_times.append(prefill_time)
            print(f"  Prefill (TTFT): {prefill_time*1000:.2f} ms")

            # Note: Decode step measurement removed - requires proper autoregressive generation
            # with voice clone preprocessing. See demo_voice_clone_ttnn.py for full pipeline.

        # Cleanup traces
        if use_trace:
            generator.release_traces()

        # ============================================================
        # PERFORMANCE SUMMARY
        # ============================================================
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)

        avg_prefill = sum(prefill_times) / len(prefill_times)
        print(f"\n{'Metric':<25} {'Value':<20} {'Unit':<15}")
        print("-" * 60)
        print(f"{'Batch Size':<25} {batch_size:<20}")
        print(f"{'Sequence Length':<25} {seq_len:<20}")
        print(f"{'Tracing':<25} {'Enabled' if use_trace else 'Disabled':<20}")
        print(f"{'Decode Tracing':<25} {'Enabled' if use_decode_trace else 'Disabled':<20}")
        print("-" * 60)
        print(f"{'TTFT (avg)':<25} {avg_prefill*1000:<20.2f} {'ms':<15}")

        if decode_times:
            avg_decode = sum(decode_times) / len(decode_times)
            tokens_per_sec = 1.0 / avg_decode if avg_decode > 0 else 0
            print(f"{'Decode Time (avg)':<25} {avg_decode*1000:<20.2f} {'ms/token':<15}")
            print(f"{'Decode Throughput':<25} {tokens_per_sec:<20.2f} {'tokens/sec':<15}")

        print("-" * 60)

        # Output shapes
        print(f"\nOutput code groups: {len(logits_list)}")
        for i, logits in enumerate(logits_list):
            logits_torch = ttnn.to_torch(logits)
            print(f"  Code group {i}: {logits_torch.shape}")

        # ============================================================
        # AUDIO GENERATION (optional)
        # ============================================================
        if generate_audio:
            print("\n" + "=" * 80)
            print("AUDIO GENERATION")
            print("=" * 80)

            # Load speech tokenizer weights
            speech_tokenizer_weights = load_speech_tokenizer_weights(model_id, cache_dir)

            if not speech_tokenizer_weights:
                print("Warning: Could not load speech tokenizer weights. Skipping audio generation.")
            else:
                # Initialize speech tokenizer decoder
                print("\nInitializing Speech Tokenizer Decoder...")
                speech_config = SpeechTokenizerConfig()
                speech_decoder = TtSpeechTokenizerDecoder(
                    device=device,
                    state_dict=speech_tokenizer_weights,
                    config=speech_config,
                )

                # Convert logits to token IDs (argmax)
                print("Converting logits to tokens...")
                token_ids_list = []
                for logits in logits_list:
                    logits_torch = ttnn.to_torch(logits)  # [batch, 1, seq_len, vocab_size]
                    # Squeeze extra dimensions and take argmax
                    logits_torch = logits_torch.squeeze()  # [seq_len, vocab_size] or [batch, seq_len, vocab_size]
                    if logits_torch.dim() == 2:
                        logits_torch = logits_torch.unsqueeze(0)  # [1, seq_len, vocab_size]
                    token_ids = torch.argmax(logits_torch, dim=-1)  # [batch, seq_len]
                    token_ids_list.append(token_ids)

                # Stack tokens: [batch, num_code_groups, seq_len]
                # Note: Only use first 16 code groups (num_quantizers)
                num_quantizers = speech_config.num_quantizers
                if len(token_ids_list) > num_quantizers:
                    token_ids_list = token_ids_list[:num_quantizers]
                elif len(token_ids_list) < num_quantizers:
                    # Pad with zeros if not enough code groups
                    while len(token_ids_list) < num_quantizers:
                        token_ids_list.append(torch.zeros_like(token_ids_list[0]))

                token_ids = torch.stack(token_ids_list, dim=1)  # [batch, num_quantizers, seq_len]
                print(f"Token IDs shape: {token_ids.shape}")

                # Generate audio
                print("Generating audio waveform...")
                audio_start = time.time()
                audio = speech_decoder.forward(token_ids)
                audio_time = time.time() - audio_start
                print(f"  Audio generation time: {audio_time*1000:.2f} ms")
                print(f"  Audio shape: {audio.shape}")

                # Save audio
                sample_rate = speech_config.output_sample_rate
                duration_sec = audio.shape[-1] / sample_rate
                print(f"  Audio duration: {duration_sec:.2f} seconds ({sample_rate} Hz)")

                save_audio(audio, audio_output, sample_rate)

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)

    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS TTNN Demo")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="TT device ID",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--weight-cache-path",
        type=str,
        default=None,
        help="TTNN weight cache path",
    )
    parser.add_argument(
        "--use-trace",
        action="store_true",
        help="Enable prefill tracing for faster execution",
    )
    parser.add_argument(
        "--use-decode-trace",
        action="store_true",
        help="Enable decode tracing for faster token generation",
    )
    parser.add_argument(
        "--num-decode-steps",
        type=int,
        default=10,
        help="Number of decode steps for performance measurement",
    )
    parser.add_argument(
        "--num-inference-runs",
        type=int,
        default=3,
        help="Number of inference runs for averaging",
    )
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate audio output using speech tokenizer decoder",
    )
    parser.add_argument(
        "--audio-output",
        type=str,
        default="output.wav",
        help="Output path for generated audio (default: output.wav)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize (required)",
    )

    args = parser.parse_args()

    run_demo(
        model_id=args.model_id,
        device_id=args.device_id,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        weight_cache_path=args.weight_cache_path,
        use_trace=args.use_trace,
        use_decode_trace=args.use_decode_trace,
        num_decode_steps=args.num_decode_steps,
        num_inference_runs=args.num_inference_runs,
        generate_audio=args.generate_audio,
        audio_output=args.audio_output,
        text=args.text,
    )


if __name__ == "__main__":
    main()
