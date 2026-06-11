# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""Higgs Audio v2 TTNN demo — three generation modes producing real 24kHz audio.

  python models/demos/audio/higgs_audio_v2/demo/demo.py --mode tts
  python models/demos/audio/higgs_audio_v2/demo/demo.py --mode multispeaker
  python models/demos/audio/higgs_audio_v2/demo/demo.py --mode voiceclone --ref-audio ref.wav

The LLM backbone runs on TTNN; the HF processor/codec handle text/audio I/O.
Outputs land in --out-dir (default: ./higgs_demo_out).
"""
import argparse
import os
import pathlib

import ttnn
from loguru import logger

from models.demos.audio.higgs_audio_v2.demo.generator import HiggsAudioTTSGenerator


SYSTEM = "Generate speech in the style of a calm neutral male voice."

TTS_TEXT = "The quick brown fox jumps over the lazy dog. Tenstorrent hardware now speaks."

MS_SYSTEM = "Generate a short two-speaker dialog with distinct voices."
MULTI_SPEAKER_TEXT = "Create a brief dialog where speaker A says hello and speaker B responds warmly."

VOICE_CLONE_TEXT = "Hello, this sentence is spoken in the cloned reference voice."


def _tts_conversation(text):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def _multispeaker_conversation(ref_a, ref_b, text):
    # Higgs multi-speaker = voice-conditioned per speaker (matches PR #40907): each
    # speaker is primed by a reference clip (assistant audio turn), then the final
    # user turn requests the dialog. This routes the reference frames through the
    # audio DualFFN branch (the conditioning the prefill mask blend enables).
    return [
        {"role": "system", "content": [{"type": "text", "text": MS_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": "Speaker A reference."}]},
        {"role": "assistant", "content": [{"type": "audio", "url": str(ref_a)}]},
        {"role": "user", "content": [{"type": "text", "text": "Speaker B reference."}]},
        {"role": "assistant", "content": [{"type": "audio", "url": str(ref_b)}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def _voiceclone_conversation(ref_audio, text):
    # A reference utterance (assistant audio turn) primes the voice; the final
    # user turn is what gets spoken in that voice.
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": "Please speak in this voice."}]},
        {"role": "assistant", "content": [{"type": "audio", "url": str(ref_audio)}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def run(mode, out_dir, ref_audio=None, ref_audio_b=None, text=None, precision="performance", max_new_tokens=750,
        temperature=1.0, top_k=50, top_p=0.95, seed=1234, silence_patience=32):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # l1_small_size pool is needed by the TTNN codec's conv1d (halo/sliding-window);
    # sized to hold both the LLM's small buffers and the codec's (~43KB) when the
    # on-device codec (HIGGS_TTNN_CODEC=1) runs co-resident with the LLM.
    l1_small = 98304 if os.environ.get("HIGGS_TTNN_CODEC") == "1" else 32768
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=l1_small)
    try:
        gen = HiggsAudioTTSGenerator(mesh_device, precision=precision)

        if mode == "tts":
            conv = _tts_conversation(text or TTS_TEXT)
        elif mode == "multispeaker":
            assert ref_audio and ref_audio_b, "multispeaker needs --ref-audio (A) and --ref-audio-b (B)"
            conv = _multispeaker_conversation(ref_audio, ref_audio_b, text or MULTI_SPEAKER_TEXT)
        elif mode == "voiceclone":
            assert ref_audio, "voiceclone mode needs --ref-audio"
            conv = _voiceclone_conversation(ref_audio, text or VOICE_CLONE_TEXT)
        else:
            raise ValueError(mode)

        logger.info(f"=== MODE: {mode} ===")
        audio_seq = gen.generate(conv, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
                                 silence_patience=silence_patience)
        out_path = out_dir / f"higgs_{mode}.wav"
        out_path, dur = gen.save(audio_seq, out_path)
        print(f"DEMO_OK mode={mode} out={out_path} duration_s={dur:.2f} rows={audio_seq.shape[1]}")
    finally:
        ttnn.close_mesh_device(mesh_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tts", "multispeaker", "voiceclone"], default="tts")
    ap.add_argument("--out-dir", default="./higgs_demo_out")
    ap.add_argument("--ref-audio", default=None, help="reference clip (voiceclone, or speaker A for multispeaker)")
    ap.add_argument("--ref-audio-b", default=None, help="speaker B reference clip (multispeaker)")
    ap.add_argument("--text", default=None)
    ap.add_argument("--precision", default="performance")
    ap.add_argument("--max-new-tokens", type=int, default=750)
    ap.add_argument("--temperature", type=float, default=1.0, help="0 = greedy (Higgs default 1.0)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--silence-patience", type=int, default=32,
                    help="stop after this many identical repeated rows (silent-tail guard)")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run(args.mode, args.out_dir, ref_audio=args.ref_audio, ref_audio_b=args.ref_audio_b, text=args.text,
        precision=args.precision, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed,
        silence_patience=args.silence_patience)


if __name__ == "__main__":
    main()
