#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS interactive demo: type text, hear it in a cloned voice.

Loads the weights once and keeps them, so every line after the first costs only its own
generation. Each line writes a numbered 24 kHz wav.

Clone a voice (needs the **Base** checkpoint):

    python -m models.demos.audio.qwen3_tts.demo.demo_server \\
        --ref my_voice.wav --ref-text "exactly what my_voice.wav says"

Or start from one of the nine built-in speakers (needs **CustomVoice**):

    python -m models.demos.audio.qwen3_tts.demo.demo_server --speaker ryan

Commands, besides plain text to speak:

    \\ref PATH | TRANSCRIPT   clone a new clip; the transcript is required and goes after |
    \\speaker NAME            switch to a built-in CustomVoice speaker (drops any clone)
    \\language NAME           set the language tag, or Auto to let the model infer
    \\seed N                  set the base seed; utterance i uses N + i, so repeats vary
    \\similarity              toggle measuring each clone against the reference clip
    \\quit                    exit (Ctrl-D and Ctrl-C also work)

The first utterance compiles its kernels and is slow. Every later one runs from the captured
traces, at roughly 44 ms a frame.
"""

import argparse
import os
import time

OUTPUT_DIR = "outputs"
DEFAULT_SEED = 42


def _read_reference(device, pipeline, path, transcript):
    """Build a `CloneReference`, releasing the traces first so eager work is safe."""
    from models.demos.audio.qwen3_tts import audio as host_audio
    from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import build_clone_reference

    transcript = str(transcript or "").strip()
    if not transcript:
        raise ValueError("a clone needs the clip's transcript: \\ref PATH | what the clip says")

    clip = host_audio.read_clip(path)
    if pipeline is not None:
        pipeline.release()
    started = time.time()
    reference = build_clone_reference(device, clip, transcript)
    print(f"  VOICE: {reference.frames} frames, {reference.frames / 12.5:.1f} s in {time.time() - started:.1f} s")
    return clip, reference


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS interactive voice cloning on Tenstorrent hardware")
    parser.add_argument("--ref", help="Reference voice clip to clone (.wav/.flac/.ogg, any rate)")
    parser.add_argument("--ref-text", help="What the reference clip says, word for word. Required with --ref")
    parser.add_argument("--speaker", help="A built-in CustomVoice speaker instead of a clone (e.g. ryan)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Base sampling seed (default: {DEFAULT_SEED})")
    parser.add_argument("--language", default=None, help="Language name (default: Auto to clone, English otherwise)")
    parser.add_argument("--max-frames", type=int, default=400, help="Frame budget, 12.5 a second (default: 400)")
    parser.add_argument("--ckpt", default=None, help="Checkpoint directory (default: $QWEN3_TTS_CKPT, else the hub)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"Where the wavs go (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    if args.ckpt:
        os.environ["QWEN3_TTS_CKPT"] = os.path.abspath(args.ckpt)
    if bool(args.ref) == bool(args.speaker):
        parser.error("pass exactly one of --ref (clone a clip) or --speaker (a built-in voice)")
    if args.ref and not str(args.ref_text or "").strip():
        parser.error("--ref-text is required with --ref: this model clones in context")

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        import readline  # noqa: F401  -- gives input() history and line editing
    except ImportError:
        pass

    import soundfile

    import ttnn
    from models.demos.audio.qwen3_tts.demo.demo import open_device, speaker_similarity
    from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline

    bar = "=" * 78
    device = open_device()
    try:
        clip = reference = None
        speaker = args.speaker
        language = args.language
        seed = args.seed
        measure = False

        print(bar)
        print("Loading. The first utterance also compiles its kernels; later ones do not.")
        print(bar)
        if args.ref:
            clip, reference = _read_reference(device, None, args.ref, args.ref_text)
        started = time.time()
        pipeline = Qwen3TTSPipeline(device, max_frames=args.max_frames, seed=seed)
        print(f"  WEIGHTS: {time.time() - started:.1f} s")

        print(bar)
        print("Ready. Type text and press ENTER to speak it.")
        print("Commands: \\ref PATH | TRANSCRIPT   \\speaker NAME   \\language NAME")
        print("          \\seed N   \\similarity   \\quit")
        print(bar)

        index = 1
        while True:
            try:
                line = input(f"\ntext [{index}]> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break
            if not line:
                continue

            if line.startswith("\\"):
                command, _, argument = line.partition(" ")
                argument = argument.strip()
                if command == "\\quit":
                    print("bye")
                    break
                elif command == "\\seed":
                    try:
                        seed = int(argument)
                        pipeline.reseed(seed)
                        print(f"  base seed = {seed}")
                    except ValueError:
                        print(f"  ERROR: \\seed needs an integer, got {argument!r}")
                elif command == "\\language":
                    language = argument or None
                    print(f"  language = {language or 'unset'}")
                elif command == "\\similarity":
                    measure = not measure
                    print(f"  similarity measurement {'on' if measure else 'off'}")
                elif command == "\\speaker":
                    speaker, clip, reference = argument, None, None
                    print(f"  speaker = {speaker}  (needs the CustomVoice checkpoint)")
                elif command == "\\ref":
                    path, _, transcript = argument.partition("|")
                    try:
                        clip, reference = _read_reference(device, pipeline, path.strip(), transcript.strip())
                        speaker = None
                    except Exception as error:
                        print(f"  ERROR: {error}")
                else:
                    print(f"  unknown command {command!r}")
                continue

            path = os.path.join(args.output_dir, f"out_{index}.wav")
            try:
                pipeline.reseed(seed + index)
                started = time.time()
                if reference is not None:
                    waveform, codes = pipeline.generate_clone(line, reference, language=language or "Auto")
                else:
                    waveform, codes = pipeline.generate(line, speaker=speaker, language=language or "English")
                elapsed = time.time() - started

                spoken = waveform.reshape(-1)
                duration = spoken.shape[0] / 24000
                soundfile.write(path, spoken.numpy(), 24000)
                timings = pipeline.last_timings
                print(
                    f"  END-TO-END: {elapsed:.2f} s  |  {duration:.2f} s audio ({elapsed / duration:.2f}x RT)  |  "
                    f"{os.path.abspath(path)}"
                )
                print(
                    f"    prefill {timings['prefill_s']:.1f} s, capture {timings['capture_s']:.1f} s, "
                    f"decode {timings['decode_s']:.1f} s ({timings['frames']} frames at "
                    f"{timings['ms_per_frame']:.0f} ms), codec {timings['codec_s']:.1f} s"
                )
                if measure and clip is not None:
                    print(
                        f"  speaker similarity to the reference: {speaker_similarity(device, pipeline, clip, spoken):.4f}"
                    )
                index += 1
            except KeyboardInterrupt:
                print("\n  interrupted during generation; shutting down.")
                break
            except RuntimeError as error:
                if "end-of-speech before any frame" not in str(error):
                    raise
                print("  no audio for this text and seed; try again, the seed advances each time")
                index += 1
            except Exception as error:
                print(f"\n  ERROR: {error}\n")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
