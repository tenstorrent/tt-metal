# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T-v2 S2TT (speech-to-text translation) demo CLI.

Translates spoken audio from ``--src-lang`` to text in ``--tgt-lang``
using the TTNN model, and prints the HuggingFace reference output
alongside for comparison.

Usage::

    python models/demos/facebook_seamless_m4t_v2_large/demo/demo_s2tt.py \\
        --wav hello.wav --src-lang eng --tgt-lang fra

Args:
    --wav            Path to a 16-bit PCM WAV file. Mono / stereo, any
                     sample rate (will be downmixed to mono + resampled
                     to 16 kHz on the host).
    --src-lang       Source language (used for HF reference parity).
    --tgt-lang       Target language code (e.g. ``fra``).
    --max-new-tokens AR generation budget (incl. 2-token prefix).
    --max-seconds    Audio is truncated to this many seconds before
                     feature extraction (default 5.0).
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _run_hf_reference(
    wav_path: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int,
    max_seconds: float,
) -> str:
    """Run HF ``SeamlessM4Tv2ForSpeechToText.generate`` and return the decoded string."""
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import _load_wav_to_16k_mono

    audio = _load_wav_to_16k_mono(wav_path)
    if max_seconds is not None:
        max_samples = int(max_seconds * 16000)
        audio = audio[:max_samples]

    proc = AutoProcessor.from_pretrained(wl.HF_PATH)
    feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(wl.HF_PATH, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_features=feats["input_features"],
            attention_mask=feats["attention_mask"],
            tgt_lang=tgt_lang,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )
    if hasattr(out, "sequences"):
        out = out.sequences
    text = proc.decode(out[0].tolist(), skip_special_tokens=True)
    del model
    return text


@app.command()
def main(
    wav: str = typer.Option(..., "--wav", help="Path to a WAV file."),
    src_lang: str = typer.Option("eng", "--src-lang", help="Source language code (e.g. eng)."),
    tgt_lang: str = typer.Option(..., "--tgt-lang", help="Target language code (e.g. fra)."),
    device_arg: str = typer.Option(
        "p150", "--device", help="Informational; we always open device 0 on the local p150."
    ),
    max_new_tokens: int = typer.Option(
        128, "--max-new-tokens", help="AR generation budget (includes the 2-token prefix)."
    ),
    max_seconds: float = typer.Option(
        5.0, "--max-seconds", help="Truncate audio to this many seconds before feature extraction."
    ),
    skip_hf: bool = typer.Option(False, "--skip-hf", help="Skip the HF reference run (TTNN only)."),
):
    """Translate spoken audio with the TTNN SeamlessM4T-v2 model."""
    from transformers import AutoProcessor

    import ttnn
    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import SpeechToTextModel

    typer.echo(f"[demo] device-arg={device_arg} (always uses device 0 on local p150)")
    typer.echo("[demo] loading HF checkpoint into host memory ...")
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    typer.echo("[demo] opening TTNN device 0 ...")
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        typer.echo(
            "[demo] building SpeechToTextModel (24-layer speech encoder + adapter + 24-layer decoder + LM head) ..."
        )
        model = SpeechToTextModel(
            device=dev,
            hf_state_dict=hf_sd,
            processor=processor,
        )
        typer.echo("[demo] running TTNN translate() ...")
        ttnn_out = model.translate(
            audio_path=wav,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_new_tokens=max_new_tokens,
            max_audio_seconds=max_seconds,
        )
    finally:
        ttnn.close_device(dev)

    typer.echo("")
    typer.echo(f"Source audio: {wav}")
    typer.echo(f"Translation: {ttnn_out}")

    if not skip_hf:
        typer.echo("[demo] running HF reference for comparison ...")
        hf_out = _run_hf_reference(wav, src_lang, tgt_lang, max_new_tokens, max_seconds)
        typer.echo(f"HF: {hf_out}")


if __name__ == "__main__":
    app()
