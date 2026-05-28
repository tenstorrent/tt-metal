# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T-v2 T2ST (text-to-speech translation) demo CLI.

Synthesises a WAV file from a source text + target language using the TTNN
``TextToSpeechModel`` (hybrid TTNN + HF text-decoder rerun + HF char-prep
helpers; see ``tt/text_to_speech_model.py`` docstring for the exact boundary).

Usage::

    python models/demos/facebook_seamless_m4t_v2_large/demo/demo_t2st.py \\
        --src "Hello world." \\
        --src-lang eng \\
        --tgt-lang fra \\
        --out /tmp/t2st_test.wav

Output is a 16 kHz mono int16 WAV; the script also prints the predicted
duration in seconds.
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _save_wav(path: str, samples_float32, sampling_rate: int = 16000) -> None:
    """Write float32 mono samples to ``path`` as 16-bit PCM WAV."""
    import numpy as np
    import scipy.io.wavfile as wav

    arr = samples_float32.astype(np.float32, copy=False)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    # Clip + scale to int16. The vocoder output already lives roughly in
    # [-1, 1] but small overshoots are possible.
    arr = np.clip(arr, -1.0, 1.0)
    arr_i16 = (arr * 32767.0).astype(np.int16)
    wav.write(path, sampling_rate, arr_i16)


@app.command()
def main(
    src: str = typer.Option(..., "--src", help="Source sentence to synthesise."),
    src_lang: str = typer.Option(..., "--src-lang", help="Source language code (e.g. eng)."),
    tgt_lang: str = typer.Option(..., "--tgt-lang", help="Target language code (e.g. fra)."),
    out: str = typer.Option(..., "--out", help="Output WAV path (16 kHz mono)."),
    speaker_id: int = typer.Option(0, "--speaker-id", help="Vocoder speaker id (default 0)."),
    max_new_tokens: int = typer.Option(
        128, "--max-new-tokens", help="AR text-generation budget (includes 2-token prefix)."
    ),
):
    """Synthesise ``--src`` (in ``--src-lang``) into speech for ``--tgt-lang``."""
    import time

    from transformers import AutoProcessor

    import ttnn
    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_speech_model import TextToSpeechModel

    typer.echo("[demo] loading HF checkpoint into host memory ...")
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    typer.echo("[demo] opening TTNN device 0 ...")
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        typer.echo(
            "[demo] building TextToSpeechModel ("
            "TTNN text_encoder + text_generator + t2u_generator + code_hifigan_vocoder; "
            "HF host text_decoder rerun + char prep) ..."
        )
        model = TextToSpeechModel(
            device=dev,
            hf_state_dict=hf_sd,
            processor=processor,
        )

        typer.echo(f"[demo] synthesising '{src}'  ({src_lang} -> {tgt_lang}) ...")
        t0 = time.time()
        audio = model.synthesize(
            src_text=src,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            speaker_id=speaker_id,
            max_new_tokens=max_new_tokens,
        )
        dt = time.time() - t0
    finally:
        ttnn.close_device(dev)

    duration_sec = float(audio.shape[-1]) / 16000.0
    _save_wav(out, audio, sampling_rate=16000)
    typer.echo("")
    typer.echo(f"Generated: {out}, duration: {duration_sec:.3f} seconds (wall-clock {dt:.2f} s)")


if __name__ == "__main__":
    app()
