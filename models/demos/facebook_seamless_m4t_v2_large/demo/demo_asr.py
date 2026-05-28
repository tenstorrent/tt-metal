# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T-v2 ASR (speech-to-text, same language) demo CLI.

ASR is a thin specialisation of S2TT where the source and target
languages are the same. We surface a single ``--lang`` flag instead of
the two-language interface to make the calling convention obvious.

Usage::

    python models/demos/facebook_seamless_m4t_v2_large/demo/demo_asr.py \\
        --wav hello.wav --lang eng

Args:
    --wav            Path to a WAV file.
    --lang           Language code of the spoken audio (e.g. ``eng``).
    --max-new-tokens AR generation budget (incl. 2-token prefix).
    --max-seconds    Audio truncation budget before feature extraction.
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _run_hf_reference(
    wav_path: str,
    lang: str,
    max_new_tokens: int,
    max_seconds: float,
) -> str:
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
            tgt_lang=lang,
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
    lang: str = typer.Option("eng", "--lang", help="Language code of the spoken audio (e.g. eng)."),
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
    """Transcribe spoken audio with the TTNN SeamlessM4T-v2 model (same source/target language)."""
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
        typer.echo("[demo] running TTNN transcribe() ...")
        ttnn_out = model.transcribe(
            audio_path=wav,
            lang=lang,
            max_new_tokens=max_new_tokens,
            max_audio_seconds=max_seconds,
        )
    finally:
        ttnn.close_device(dev)

    typer.echo("")
    typer.echo(f"Source audio: {wav}")
    typer.echo(f"Transcription: {ttnn_out}")

    if not skip_hf:
        typer.echo("[demo] running HF reference for comparison ...")
        hf_out = _run_hf_reference(wav, lang, max_new_tokens, max_seconds)
        typer.echo(f"HF: {hf_out}")


if __name__ == "__main__":
    app()
