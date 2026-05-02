#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Voxtral-4B-TTS-2603 demo on Tenstorrent N150.

Synthesizes speech from text using a preset voice.

Usage:
  python3 demo.py --text "Hello, world." --voice casual_male --output hello.wav
  python3 demo.py --text "Bonjour le monde." --voice fr_female --output bonjour.wav
"""

import argparse
import os
from pathlib import Path

import torch

DEFAULT_MODEL_DIR = os.environ.get(
    "VOXTRAL_MODEL_DIR",
    "/home/ttuser/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/b81be46c3777f88621676791b512bb01dc1cb970",
)

VOICES = [
    "casual_male",
    "casual_female",
    "neutral_male",
    "neutral_female",
    "cheerful_female",
    "fr_male",
    "fr_female",
    "de_male",
    "de_female",
    "es_male",
    "es_female",
    "it_male",
    "it_female",
    "nl_male",
    "nl_female",
    "pt_male",
    "pt_female",
    "hi_male",
    "hi_female",
    "ar_male",
]


def main():
    parser = argparse.ArgumentParser(description="Voxtral TTS demo")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default="casual_male", choices=VOICES)
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--n-ode-steps", type=int, default=8, help="Number of ODE solver steps (8=default, 4=faster)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "consolidated.safetensors").exists():
        print(f"Model not found at {model_dir}")
        print("Download with: huggingface-cli download mistralai/Voxtral-4B-TTS-2603")
        return

    print(f"Loading Voxtral TTS from {model_dir}...")
    import ttnn

    device = ttnn.open_device(device_id=0)

    try:
        from models.demos.voxtral_tts.tt.load_checkpoint import load_voice_embeddings
        from models.demos.voxtral_tts.tt.model import VoxtralTTSModel

        model = VoxtralTTSModel.from_pretrained(model_dir, device)
        voices = load_voice_embeddings(model_dir)

        if args.voice not in voices:
            print(f"Voice '{args.voice}' not found. Available: {list(voices.keys())}")
            return

        voice_emb = voices[args.voice].unsqueeze(0)
        print(f"Voice: {args.voice}, shape: {voice_emb.shape}")

        # Tokenize input text
        from mistral_common.protocol.instruct.chunk import TextChunk
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tok = MistralTokenizer.from_file(str(model_dir / "tekken.json"))
        req = ChatCompletionRequest(messages=[UserMessage(content=[TextChunk(text=args.text)])])
        token_ids = torch.tensor([tok.encode_chat_completion(req).tokens])
        print(f"Text: '{args.text}' → {token_ids.shape[1]} tokens")

        # Generate speech
        print(f"Generating with {args.n_ode_steps} ODE steps...")
        waveform = model.generate_tts(token_ids, voice_emb, n_ode_steps=args.n_ode_steps)

        # Save to WAV
        n_samples = waveform.shape[1]
        duration = n_samples / 24000
        print(f"Generated: {n_samples} samples = {duration:.2f}s at 24kHz")

        try:
            import soundfile as sf

            sf.write(args.output, waveform[0].numpy(), 24000)
            print(f"Saved to: {args.output}")
        except ImportError:
            # Fallback: save raw float32 data
            raw_path = args.output.replace(".wav", ".raw.float32")
            waveform[0].numpy().tofile(raw_path)
            print(f"soundfile not available. Saved raw float32 to: {raw_path}")
            print("Convert with: sox -r 24000 -e floating-point -b 32 raw.float32 output.wav")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
