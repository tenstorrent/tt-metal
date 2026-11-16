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

Describe a voice instead of recording one (needs **VoiceDesign**):

    python -m models.demos.audio.qwen3_tts.demo.demo_server \\
        --instruct "A calm older man speaking slowly, with a slight rasp."

Or start from one of the nine built-in speakers (needs **CustomVoice**):

    python -m models.demos.audio.qwen3_tts.demo.demo_server --speaker ryan

Commands, besides plain text to speak:

    \\ref PATH | TRANSCRIPT   clone a new clip; the transcript is required and goes after |
    \\instruct DESCRIPTION    redesign the voice from words (VoiceDesign checkpoint)
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


def _read_reference(device, pipeline, path, transcript, x_vector=False):
    """Build a `CloneReference`, releasing the traces first so eager work is safe."""
    from models.demos.audio.qwen3_tts import audio as host_audio
    from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import build_clone_reference

    transcript = str(transcript or "").strip()
    if not transcript and not x_vector:
        raise ValueError(
            "in-context cloning needs the clip's transcript: \\ref PATH | what the clip says. "
            "Start the server with --x-vector to clone from the voice alone instead"
        )

    clip = host_audio.read_clip(path)
    if pipeline is not None:
        pipeline.release()
    started = time.time()
    reference = build_clone_reference(device, clip, transcript, x_vector_only=x_vector)
    took = time.time() - started
    if x_vector:
        print(f"  VOICE: the clip's speaker vector alone (x-vector) in {took:.1f} s")
    else:
        print(f"  VOICE: {reference.frames} frames, {reference.frames / 12.5:.1f} s in {took:.1f} s")
    return clip, reference


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS interactive voice cloning on Tenstorrent hardware")
    parser.add_argument("--ref", help="Reference voice clip to clone (.wav/.flac/.ogg, any rate)")
    parser.add_argument("--ref-text", help="What the reference clip says, word for word. Required with --ref")
    parser.add_argument("--speaker", help="A built-in CustomVoice speaker instead of a clone (e.g. ryan)")
    parser.add_argument("--instruct", help="Describe the voice in words instead (VoiceDesign checkpoint)")
    parser.add_argument(
        "--x-vector",
        action="store_true",
        help="Clone from the voice alone, no transcript needed (upstream's x_vector_only_mode)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the text in a token per frame instead of putting it all in the prompt",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Base sampling seed (default: {DEFAULT_SEED})")
    parser.add_argument("--language", default=None, help="Language name (default: Auto to clone, English otherwise)")
    parser.add_argument("--max-frames", type=int, default=400, help="Frame budget, 12.5 a second (default: 400)")
    parser.add_argument("--ckpt", default=None, help="Checkpoint directory (default: $QWEN3_TTS_CKPT, else the hub)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"Where the wavs go (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    if args.ckpt:
        os.environ["QWEN3_TTS_CKPT"] = os.path.abspath(args.ckpt)
    voices = [name for name, value in (("--ref", args.ref), ("--speaker", args.speaker)) if value]
    if len(voices) > 1:
        parser.error(f"pass one of --ref or --speaker, not both: got {', '.join(voices)}")
    if not voices and not args.instruct:
        parser.error("pass --ref (clone a clip), --speaker (a built-in voice) or --instruct (describe a voice)")
    if args.ref and args.instruct and not args.x_vector:
        parser.error("--instruct with --ref needs --x-vector: in-context cloning leaves no room for one")
    if args.x_vector and not args.ref:
        parser.error("--x-vector describes how to use --ref, so it needs one")
    if args.ref and not args.x_vector and not str(args.ref_text or "").strip():
        parser.error("--ref-text is required with --ref, or pass --x-vector to clone from the voice alone")

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
        instruct = args.instruct
        language = args.language
        seed = args.seed
        measure = False

        print(bar)
        print("Loading. The first utterance also compiles its kernels; later ones do not.")
        print(bar)
        if args.ref:
            clip, reference = _read_reference(device, None, args.ref, args.ref_text, args.x_vector)
        elif instruct:
            print(f"  VOICE:  {instruct!r}")
        started = time.time()
        pipeline = Qwen3TTSPipeline(device, max_frames=args.max_frames, seed=seed)
        print(f"  WEIGHTS: {time.time() - started:.1f} s")

        print(bar)
        print("Ready. Type text and press ENTER to speak it.")
        print("Commands: \\ref PATH | TRANSCRIPT   \\instruct DESCRIPTION   \\speaker NAME")
        print("          \\language NAME")
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
                    speaker, instruct, clip, reference = argument, None, None, None
                    print(f"  speaker = {speaker}  (needs the CustomVoice checkpoint)")
                elif command == "\\instruct":
                    if not argument:
                        print("  ERROR: \\instruct needs a description of the voice")
                    else:
                        instruct, speaker, clip, reference = argument, None, None, None
                        print(f"  voice = {instruct!r}  (needs the VoiceDesign checkpoint)")
                elif command == "\\ref":
                    path, _, transcript = argument.partition("|")
                    try:
                        clip, reference = _read_reference(
                            device, pipeline, path.strip(), transcript.strip(), args.x_vector
                        )
                        speaker = None
                        if not args.x_vector:
                            instruct = None  # in-context cloning has nowhere to put one
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
                    waveform, codes = pipeline.generate_clone(
                        line,
                        reference,
                        language=language or "Auto",
                        streaming=args.streaming,
                        x_vector_only=args.x_vector,
                        instruct=instruct if args.x_vector else None,
                    )
                elif speaker:
                    waveform, codes = pipeline.generate(
                        line,
                        speaker=speaker,
                        language=language or "English",
                        streaming=args.streaming,
                        instruct=instruct,
                    )
                else:
                    # Unreachable otherwise: a line needs a reference, a speaker or an instruction.
                    waveform, codes = pipeline.generate_design(
                        line, instruct, language=language or "Auto", streaming=args.streaming
                    )
                elapsed = time.time() - started

                spoken = waveform.reshape(-1)
                duration = spoken.shape[0] / 24000
                soundfile.write(path, spoken.numpy(), 24000)
                timings = pipeline.last_timings
                print(
                    f"  END-TO-END: {elapsed:.2f} s  |  {duration:.2f} s audio "
                    f"({duration / elapsed:.2f}x faster than real time)  |  "
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
