#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice Demo - Run text-to-speech inference on Tenstorrent hardware.

Usage:
    # SFT mode
    python models/demos/wormhole/cosyvoice/demo/demo.py \\
        --text "Hello, welcome to CosyVoice on Tenstorrent." \\
        --mode sft --output output.wav

    # Zero-shot voice cloning
    python models/demos/wormhole/cosyvoice/demo/demo.py \\
        --text "Hello world." \\
        --mode zero_shot --ref_audio reference.wav --output output.wav

    # Cross-lingual
    python models/demos/wormhole/cosyvoice/demo/demo.py \\
        --text "你好，欢迎使用CosyVoice。" \\
        --mode cross_lingual --ref_audio reference.wav --output output.wav

    # Instruct mode
    python models/demos/wormhole/cosyvoice/demo/demo.py \\
        --text "This is an expressive voice." \\
        --mode instruct --instruct "Speak excitedly" --output output.wav
"""

import argparse
import sys
import time

import torch
from loguru import logger

import ttnn


def parse_args():
    parser = argparse.ArgumentParser(description="CosyVoice TTS Demo on TT hardware")
    parser.add_argument("--text", type=str, required=True, help="Input text to synthesize")
    parser.add_argument("--mode", type=str, default="sft",
                        choices=["sft", "zero_shot", "cross_lingual", "instruct"],
                        help="Inference mode")
    parser.add_argument("--ref_audio", type=str, default=None,
                        help="Reference audio path for zero-shot/cross-lingual")
    parser.add_argument("--instruct", type=str, default=None,
                        help="Instruction for instruct mode")
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "zh", "ja", "yue", "ko"],
                        help="Input text language")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--device", type=str, default=None,
                        help="TT device ID (default: auto-detect)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize TT device
    logger.info("Initializing Tenstorrent device...")
    device = ttnn.open_device(device_id=args.device or 0)

    try:
        # Import pipeline (import here to allow device init first)
        from models.demos.wormhole.cosyvoice.tt.pipeline import TtCosyVoicePipeline
        from models.demos.wormhole.cosyvoice.tt.model_config import CosyVoiceModelConfig

        # Create model config
        config = CosyVoiceModelConfig()

        # Create pipeline with placeholder state dict
        # Real weights are loaded from pretrained checkpoints
        state_dict = _load_or_create_state_dict(config)
        pipeline = TtCosyVoicePipeline(device, config, state_dict)

        # Run inference
        logger.info(f"Running {args.mode} inference...")
        logger.info(f"Text: {args.text}")

        start_time = time.time()
        audio = pipeline.tts(
            text=args.text,
            mode=args.mode,
            ref_audio=args.ref_audio,
            instruct=args.instruct,
            language=args.language,
        )
        elapsed = time.time() - start_time

        # Save output
        _save_audio(audio, args.output)

        logger.info(f"Generated {audio.shape[-1]} audio samples in {elapsed:.2f}s")
        logger.info(f"Output saved to: {args.output}")
        logger.info(f"Real-time factor: {elapsed / (audio.shape[-1] / 24000):.2f}x")

    finally:
        ttnn.close_device(device)


def _load_or_create_state_dict(config):
    """Load pretrained weights or create placeholder state dict.

    In Stage 1, we create a placeholder state dict with correctly-shaped
    tensors. Real weights are loaded from HuggingFace checkpoints using
    the download_weights.py script.
    """
    state_dict = {}
    hidden_size = config.llm_hidden_size

    # Attempt to load real weights
    weight_path = "/tmp/tt-metal-weights/cosyvoice/cosyvoice.pt"
    try:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        logger.info(f"Loaded pretrained weights from {weight_path}")
        return checkpoint
    except (FileNotFoundError, RuntimeError) as e:
        logger.warning(f"Could not load pretrained weights ({e}), using placeholders")

    # LLM embeddings
    state_dict["llm.model.model.embed_tokens.weight"] = torch.randn(config.text_token_size, hidden_size)
    state_dict["llm.speech_embedding.weight"] = torch.randn(config.speech_token_size + 3, hidden_size)
    state_dict["llm.llm_embedding.weight"] = torch.randn(2, hidden_size)

    # LLM decoder
    state_dict["llm.llm_decoder.weight"] = torch.randn(hidden_size, config.speech_token_size + 3)
    state_dict["llm.llm_decoder.bias"] = torch.zeros(config.speech_token_size + 3)

    # Text encoder
    state_dict["llm.text_encoder_affine_layer.weight"] = torch.randn(hidden_size, hidden_size)

    # Speaker embedding
    state_dict["llm.spk_embed_affine_layer.weight"] = torch.randn(192, hidden_size)

    # Transformer layers
    for i in range(config.llm_num_layers):
        prefix = f"llm.llm.model.model.layers.{i}"
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(config.llm_intermediate_size, hidden_size)
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(config.llm_intermediate_size, hidden_size)
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, config.llm_intermediate_size)
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size)

    # Final norm
    state_dict["llm.llm.model.model.norm.weight"] = torch.ones(hidden_size)

    # Flow decoder
    state_dict["flow.input_embedding.weight"] = torch.randn(config.flow_vocab_size, config.flow_input_size)
    state_dict["flow.spk_embed_affine_layer.weight"] = torch.randn(192, config.flow_output_size)
    state_dict["flow.encoder_proj.weight"] = torch.randn(config.flow_input_size, config.flow_output_size)

    return state_dict


def _save_audio(audio: torch.Tensor, path: str):
    """Save audio tensor to WAV file."""
    import scipy.io.wavfile as wav

    # Ensure audio is in correct range
    audio = audio.squeeze().clamp(-1, 1)

    # Save as 16-bit WAV
    audio_int16 = (audio * 32767).short()
    wav.write(path, 24000, audio_int16.numpy())
    logger.info(f"Audio saved to {path}")


if __name__ == "__main__":
    main()
