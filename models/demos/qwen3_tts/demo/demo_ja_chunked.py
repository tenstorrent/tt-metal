# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Chunked TTS: split long text into segments, generate each with same speaker, concatenate audio.
"""

import argparse
import os
import re
import time

import numpy as np
import soundfile as sf


def split_text(text, max_chars=80):
    """Split text at sentence boundaries (。!！?？) keeping chunks under max_chars."""
    sentences = re.split(r'(?<=[。！！？？\n])', text)
    sentences = [s for s in sentences if s.strip()]

    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current += s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunked TTS for long text")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--load_speaker", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--max_chars", type=int, default=80)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    args = parser.parse_args()

    chunks = split_text(args.text, args.max_chars)
    print(f"Split into {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"  [{i}] ({len(c)} chars) {c[:60]}...")

    import ttnn
    from models.demos.qwen3_tts.demo.demo_ja import build_tt_generator
    from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

    generator, mesh_device = build_tt_generator(args.model_path, args.max_new_tokens)

    speaker_emb_tt = SpeakerEncoder.load_embedding(args.load_speaker, mesh_device)
    print(f"Loaded speaker embedding from {args.load_speaker}")

    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    all_wavs = []
    total_elapsed = 0.0
    sr_out = 24000

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
        print(f"Text: {chunk}")
        t0 = time.time()
        waveform, sr = generator.generate(
            text=chunk,
            language="japanese",
            speaker_emb_tt=speaker_emb_tt,
            **gen_config,
        )
        elapsed = time.time() - t0
        total_elapsed += elapsed
        duration = len(waveform) / sr
        sr_out = sr
        print(f"  -> {duration:.2f}s audio in {elapsed:.2f}s (RTF={elapsed/duration:.3f})")
        all_wavs.append(waveform)

        # Add 300ms silence between chunks
        all_wavs.append(np.zeros(int(sr * 0.3), dtype=waveform.dtype))

    # Remove trailing silence
    if all_wavs:
        all_wavs.pop()

    combined = np.concatenate(all_wavs)
    sf.write(args.output, combined, sr_out)
    total_duration = len(combined) / sr_out
    print(f"\nDone: {args.output} ({total_duration:.2f}s audio, {total_elapsed:.2f}s total, RTF={total_elapsed/total_duration:.3f})")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
