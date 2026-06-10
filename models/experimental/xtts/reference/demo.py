"""Reference demo for XTTS-v2 (Coqui TTS).

Mirrors the usage examples from the model card:
https://huggingface.co/coqui/XTTS-v2

Two paths are provided:

  * ``--mode api``    – high-level ``TTS.api`` wrapper (downloads the model
                        from the Coqui model hub on first run).
  * ``--mode direct`` – load ``XttsConfig`` / ``Xtts`` from a local checkpoint
                        directory and call ``model.synthesize`` directly. This
                        is the path the TTNN port is validated against.

Example
-------
    # Simple API path (auto-downloads the model)
    python demo.py --mode api \\
        --text "It took me quite a long time to develop a voice." \\
        --speaker_wav /path/to/speaker.wav \\
        --language en \\
        --output output.wav

    # Direct path (explicit local checkpoint)
    python demo.py --mode direct \\
        --checkpoint_dir /path/to/xtts/ \\
        --speaker_wav /path/to/speaker.wav \\
        --language en \\
        --output output.wav
"""

import argparse
import os

import torch

DEFAULT_TEXT = "It took me quite a long time to develop a voice, and now that I have it " "I'm not going to be silent."
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def run_api(args):
    """High-level path — model card example #1."""
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(MODEL_NAME).to(device)

    print("Model Architecture: ", tts.synthesizer.tts_model)
    tts.tts_to_file(
        text=args.text,
        file_path=args.output,
        speaker_wav=args.speaker_wav,
        language=args.language,
    )
    print(f"wrote {args.output}")


def run_direct(args):
    """Direct model-loading path — model card example #3."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(args.checkpoint_dir, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=args.checkpoint_dir, eval=True)
    if torch.cuda.is_available():
        model.cuda()

    outputs = model.synthesize(
        args.text,
        config,
        speaker_wav=args.speaker_wav,
        gpt_cond_len=args.gpt_cond_len,
        language=args.language,
    )

    import torchaudio

    wav = torch.as_tensor(outputs["wav"]).reshape(1, -1)
    torchaudio.save(args.output, wav, sample_rate=24000)
    print(f"wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description="XTTS-v2 reference demo")
    parser.add_argument("--mode", choices=["api", "direct"], default="api")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speaker_wav", required=True, help="reference speaker clip (wav)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", default="output.wav")
    # direct-mode only
    parser.add_argument("--checkpoint_dir", help="local XTTS-v2 checkpoint dir (direct mode)")
    parser.add_argument("--gpt_cond_len", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "direct" and not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required when --mode direct")

    if args.mode == "api":
        run_api(args)
    else:
        run_direct(args)


if __name__ == "__main__":
    main()
