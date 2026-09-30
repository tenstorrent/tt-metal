# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 demo: text -> speech on Tenstorrent hardware.

    python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_to_speech \
        --text "Paris is a beautiful city!" --out-dir /tmp/voxtral_wav

Runs the SAME `pipeline.run_text_to_speech` the e2e test asserts on -- there is one copy of the
wiring, so a green test guarantees this demo works.

The prompt is the model's own speech-request layout with a preset voice (Source A ships 20 as
`voice_embedding/<id>.pt`) substituted into its `[AUDIO]` placeholders. Decoding stops on the
model's own `end_audio` token, per row; each WAV is cut at its own row's end.
"""
from __future__ import annotations

import argparse
import os
import sys

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline


def write_wav(path, samples, sampling_rate):
    """A 16-bit PCM WAV, written with the stdlib so the demo needs no audio dependency."""
    import wave

    import torch

    clipped = torch.clamp(samples.reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).to(torch.int16).numpy().tobytes()
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sampling_rate))
        handle.writeframes(pcm)
    return len(pcm) // 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voxtral-4B-TTS-2603 text-to-speech on TTNN")
    parser.add_argument("--text", action="append", default=None, help="a prompt to speak; repeat for more")
    parser.add_argument("--texts-file", default=None, help="one prompt per line")
    parser.add_argument("--voice", default=common.DEFAULT_VOICE, help="a preset from tekken.json's voice list")
    parser.add_argument(
        "--batch", type=int, default=common.DEFAULT_BATCH, help="independent samples per call (default 32)"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="safety cap in 12.5 Hz frames")
    parser.add_argument("--layers", type=int, default=None, help="cap EVERY repeated stack (None = all)")
    parser.add_argument("--out-dir", default="/tmp/voxtral_4b_tts_2603_wav")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--score", action="store_true", help="also transcribe (Whisper WER) and MOS-score the output")
    args = parser.parse_args(argv)

    texts = list(args.text or [])
    if args.texts_file:
        with open(args.texts_file) as handle:
            texts += [line.strip() for line in handle if line.strip()]
    if not texts:
        texts = list(common.SPEECH_TEXTS[: args.batch])
    if len(texts) < common.DEFAULT_BATCH:
        # The decode step runs at the model's batch tile of 32; a shorter request is filled by
        # repeating its own prompts, and only the requested rows are written out.
        requested = len(texts)
        texts = (texts * common.DEFAULT_BATCH)[: common.DEFAULT_BATCH]
    else:
        requested = len(texts)

    common.use_all_cpu_threads()
    hf_model = common.load_reference_model()
    max_frames, provenance = common.resolve_max_frames(hf_model)
    if args.max_frames:
        max_frames, provenance = int(args.max_frames), "--max-frames"
    print(f"max_frames {max_frames}  <- {provenance}")

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=24576,
        trace_region_size=200 * 1024 * 1024,
        num_command_queues=1,
    )
    try:
        input_ids, audio_mask, voice_embedding = common.build_voice_prompt(texts, args.voice)
        pipe = pipeline.build_pipeline(
            device,
            model=hf_model,
            heads=("text_to_speech",),
            layers=args.layers,
            batch=len(texts),
            kv_capacity=pipeline.tts_kv_capacity(input_ids.shape[-1], max_frames),
        )
        voice = pipe.stage_voice(audio_mask, voice_embedding, input_ids=input_ids)
        print(
            f"voice={args.voice!r} ({voice_embedding.shape[0]} audio tokens)  batch={input_ids.shape[0]}  "
            f"prompt_tokens={input_ids.shape[1]}"
        )

        result = pipe.run_text_to_speech(input_ids=input_ids, max_frames=max_frames, voice=voice)
        print(f"frames decoded: {result['frames_decoded']}  ({result['stop_reason']})")

        os.makedirs(args.out_dir, exist_ok=True)
        waves = pipeline.trim_to_end(result)
        for i in range(requested):
            path = os.path.join(args.out_dir, f"sample_{i:02d}.wav")
            n = write_wav(path, waves[i], result["sampling_rate"])
            print(f"  {path}  {n / result['sampling_rate']:.2f} s  <- {texts[i][:60]!r}")

        if args.score:
            from models.demos.voxtral_4b_tts_2603.reference import quality

            scores = quality.score(waves[:requested], result["sampling_rate"], texts[:requested])
            for i in range(requested):
                print(f"  [{i:02d}] WER={scores['wer'][i]:.3f} MOS={scores['mos'][i]:.2f}  {scores['transcripts'][i]!r}")
            print(f"corpus WER={scores['corpus_wer']:.4f}  mean MOS={sum(scores['mos']) / requested:.3f}")
        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
