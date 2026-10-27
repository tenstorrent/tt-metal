#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS demo: speak text on Tenstorrent hardware, in a cloned voice or a named one.

Hardware: one Wormhole N150 or Blackhole P150/P300 chip.
Models:   the 28-layer talker and the 5-layer code predictor (both KV-cached and
          Metal-traced), the codec decoder, and for a clone the codec encoder and the
          speaker encoder, all on TTNN; tokenizer, mel front-end and sampling on host.

Clone a voice from a clip (needs the **Base** checkpoint):

    python -m models.demos.audio.qwen3_tts.demo.demo "Text to speak." \\
        --ref my_voice.wav --ref-text "exactly what my_voice.wav says"

Describe a voice instead of recording one (needs the **VoiceDesign** checkpoint):

    python -m models.demos.audio.qwen3_tts.demo.demo "Text to speak." \\
        --instruct "A calm older man speaking slowly, with a slight rasp."

Or use one of the nine built-in speakers (needs the **CustomVoice** checkpoint):

    python -m models.demos.audio.qwen3_tts.demo.demo "Text to speak." --speaker ryan

`--ref` is the voice to clone: any .wav/.flac/.ogg at any sample rate, downmixed to mono
and resampled to 24 kHz. Output is a 24 kHz wav.

**`--ref-text` is not optional.** This model clones in context: the prompt carries the
clip's transcript beside its codes, so the model is told what the clip said as well as how
it sounded. A wrong transcript degrades the clone. Three to ten seconds of clean speech is
the useful range; the whole clip is used, so the transcript must cover all of it.

Checkpoint resolution: --ckpt > $QWEN3_TTS_CKPT > $HF_MODEL > the Base repo from the HF hub.
The three ways of choosing a voice live in three releases, and each refuses the others'
input: Base carries the speaker encoder and no speakers, CustomVoice the nine speakers and
no encoder, VoiceDesign neither.
"""

import argparse
import os
import time

# The speaker encoder's weights, kept per device so a REPL measuring every utterance does
# not reload 12M parameters each time.
_SPEAKER_PARAMETERS = {}


def open_device():
    import ttnn

    # The two decode traces need a trace region, and the convolutions need L1 scratch.
    # 65536 rather than 32768: the smaller region starves the codec's parallelisation,
    # measured 56.9 s against 10.4 s for the same clip. Larger is worse, not better, since
    # the region is carved out of the L1 the convolutions sharded activations into: 131072
    # left every utterance three times slower.
    return ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=90_000_000)


def speaker_similarity(device, pipeline, reference_clip, spoken):
    """Cosine between the reference clip's speaker vector and the generated speech's.

    The speaker encoder is what the prompt uses to carry a voice, so it is also the fairest
    judge of whether the voice came out the other end. Around 0.99 means the same speaker;
    an unrelated voice measures about 0.82.
    """
    import torch

    from models.demos.audio.qwen3_tts import audio as host_audio
    from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_speaker import preprocess_speaker_parameters, speaker_embedding

    pipeline.release()  # eager work cannot run beside a live trace
    parameters = _SPEAKER_PARAMETERS.get(id(device))
    if parameters is None:
        parameters = _SPEAKER_PARAMETERS[id(device)] = preprocess_speaker_parameters(device)
    vectors = [speaker_embedding(device, host_audio.speaker_mel(clip), parameters) for clip in (reference_clip, spoken)]
    return float(torch.nn.functional.cosine_similarity(*vectors).item())


def report_timings(timings, elapsed, duration, reference=None):
    """Where the time went, stage by stage.

    Worth splitting out because the total is misleading on a first run: a new prompt length
    compiles the prefill and a new frame count compiles the codec, and those land inside
    whichever stage triggered them. The decode line is the one that reflects steady state,
    since both its traces are captured before it starts.
    """
    print("\nTimings:")
    print(f"  prefill          {timings['prefill_s']:8.2f} s   (prompt = {timings['prompt']} positions)")
    print(f"  trace capture    {timings['capture_s']:8.2f} s   (talker step and predictor step, once per utterance)")
    print(
        f"  decode           {timings['decode_s']:8.2f} s   "
        f"({timings['frames']} frames at {timings['ms_per_frame']:.0f} ms each)"
    )
    codec_frames = timings.get("codec_frames", timings["frames"])
    # Only in-context cloning sends the clip's frames through the decoder.
    trailer = ", the reference rides along and is cut" if codec_frames > timings["frames"] else ""
    print(f"  codec decoder    {timings['codec_s']:8.2f} s   ({codec_frames} frames{trailer})")
    # Audio over wall clock, so above 1 is faster than real time.
    print(f"  total            {elapsed:8.2f} s   ({duration / elapsed:.2f}x faster than real time)")

    # Warm, a frame is about 32 ms and a codec frame about 2. Well above either was a compile.
    slow = []
    if timings["ms_per_frame"] > 100:
        slow.append("the decode loop")
    if 1000 * timings["codec_s"] / max(codec_frames, 1) > 45:
        slow.append("the codec decoder")
    if slow:
        print(f"\n  {' and '.join(slow)} compiled kernels on this run, which happens once per")
        print("  prompt length and once per frame count. A repeat of the same shapes is faster.")


# ── public API ──────────────────────────────────────────────────────────────


def run(
    text,
    ref=None,
    ref_text=None,
    speaker=None,
    instruct=None,
    out="out.wav",
    seed=None,
    language=None,
    max_frames=400,
    ckpt=None,
    similarity=True,
    streaming=False,
    x_vector=False,
):
    """Speak `text` in the `ref` clip's voice, as a named `speaker`, or as `instruct` describes.

    Writes a 24 kHz wav to `out` and returns its path, or None if the model stopped before
    producing a frame.
    """
    import soundfile

    import ttnn

    if ckpt:
        os.environ["QWEN3_TTS_CKPT"] = os.path.abspath(ckpt)
    # One source of voice, since they share a prompt position. `--instruct` designs one alone
    # and directs one otherwise; in-context cloning has no room for it.
    voices = [name for name, value in (("--ref", ref), ("--speaker", speaker)) if value]
    if len(voices) > 1:
        raise ValueError(f"pass one of --ref or --speaker, not both: got {', '.join(voices)}")
    if not voices and not instruct:
        raise ValueError("pass --ref (clone a clip), --speaker (a built-in voice) or --instruct (describe a voice)")
    if ref and instruct and not x_vector:
        raise ValueError(
            "--instruct with --ref needs --x-vector: in-context cloning fills the prompt with "
            "the clip's own codes and upstream leaves no room for an instruction there"
        )
    if x_vector and not ref:
        raise ValueError("--x-vector describes how to use --ref, so it needs one")
    if ref and not x_vector and not str(ref_text or "").strip():
        raise ValueError(
            "--ref-text is required with --ref: this model clones in context, see the module "
            "docstring. Or pass --x-vector to clone from the voice alone, which needs no transcript"
        )

    from models.demos.audio.qwen3_tts import audio as host_audio
    from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline, build_clone_reference

    bar = "=" * 78
    clip = None
    if ref:
        clip = host_audio.read_clip(ref)
        seconds = clip.shape[0] / 24000
        how = "voice only" if x_vector else f"transcript {ref_text!r}"
        print(f"Reference: {ref}  ({seconds:.1f} s, {how})")
        if seconds > 15:
            print(f"  note: {seconds:.0f} s is longer than this model needs. 3 to 10 s is the useful range,")
            print("        and every reference frame costs a prompt position and codec time.")

    device = open_device()
    try:
        reference = None
        started = time.time()
        if ref:
            print(bar)
            encoders = "speaker encoder" if x_vector else "codec encoder + speaker encoder"
            print(f"Reading the voice ({encoders}, once per clip) ...")
            print(bar)
            reference = build_clone_reference(device, clip, ref_text, x_vector_only=x_vector)
            read = f"{reference.frames} reference frames" if reference.frames else "the voice, no codes"
            print(f"  {read} in {time.time() - started:.1f} s")

        loaded = time.time()
        pipeline = Qwen3TTSPipeline(device, max_frames=max_frames, seed=seed)
        print(f"  weights in {time.time() - loaded:.1f} s")

        tag = language or ("English" if speaker else "Auto")
        print(f"\nText: {text!r}   (seed={seed}, language={tag})")
        if instruct:
            print(f"Voice: {instruct!r}")
        print("  the first utterance compiles its kernels; later ones in the same process do not")
        started = time.time()
        if ref:
            waveform, codes = pipeline.generate_clone(
                text, reference, language=tag, streaming=streaming, x_vector_only=x_vector, instruct=instruct
            )
        elif speaker:
            waveform, codes = pipeline.generate(
                text, speaker=speaker, language=tag, streaming=streaming, instruct=instruct
            )
        else:
            waveform, codes = pipeline.generate_design(text, instruct, language=tag, streaming=streaming)
        elapsed = time.time() - started

        spoken = waveform.reshape(-1)
        duration = spoken.shape[0] / 24000
        out = os.path.abspath(out)
        if os.path.dirname(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
        soundfile.write(out, spoken.numpy(), 24000)

        print(f"\n  -> {out}  ({duration:.2f} s of audio, {codes.shape[0]} frames)")
        report_timings(pipeline.last_timings, elapsed, duration, reference)

        if ref and similarity:
            measured = speaker_similarity(device, pipeline, clip, spoken)
            print(f"\n  speaker similarity to the reference clip: {measured:.4f}  (an unrelated voice measures ~0.82)")
        return out
    except RuntimeError as error:
        if "end-of-speech before any frame" not in str(error):
            raise
        print("\nThe model stopped before producing a frame. Try another --seed.")
        return None
    finally:
        ttnn.close_device(device)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: voice-cloning text to speech on Tenstorrent hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # clone a voice (Base checkpoint)
  python -m models.demos.audio.qwen3_tts.demo.demo "Hello from my own voice." \\
      --ref my_voice.wav --ref-text "what my voice clip says, word for word"

  # a voice described in words (VoiceDesign checkpoint)
  python -m models.demos.audio.qwen3_tts.demo.demo "Hello there." \\
      --instruct "A calm older man speaking slowly, with a slight rasp."

  # one of the nine built-in speakers (CustomVoice checkpoint)
  python -m models.demos.audio.qwen3_tts.demo.demo "Hello there." --speaker ryan

  # a longer utterance, reproducible
  python -m models.demos.audio.qwen3_tts.demo.demo "..." --ref ref.wav --ref-text "..." \\
      --seed 0 --out hello.wav
""",
    )
    parser.add_argument("text", help="Text to speak")
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
    parser.add_argument("--out", default="out.wav", help="Output wav path (default: out.wav)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (default: unseeded)")
    parser.add_argument("--language", default=None, help="Language name (default: Auto to clone, English otherwise)")
    parser.add_argument("--max-frames", type=int, default=400, help="Frame budget, 12.5 a second (default: 400)")
    parser.add_argument("--ckpt", default=None, help="Checkpoint directory (default: $QWEN3_TTS_CKPT, else the hub)")
    parser.add_argument(
        "--no-similarity", action="store_true", help="Skip measuring how close the clone is to the reference"
    )
    args = parser.parse_args()

    run(
        text=args.text,
        ref=args.ref,
        ref_text=args.ref_text,
        speaker=args.speaker,
        instruct=args.instruct,
        streaming=args.streaming,
        x_vector=args.x_vector,
        out=args.out,
        seed=args.seed,
        language=args.language,
        max_frames=args.max_frames,
        ckpt=args.ckpt,
        similarity=not args.no_similarity,
    )


if __name__ == "__main__":
    main()
