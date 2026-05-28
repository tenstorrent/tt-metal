# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T-v2 S2ST (speech-to-speech translation) demo CLI.

Translates spoken audio in one language to a synthesised WAV in another
language, using the TTNN ``SpeechToSpeechModel`` (hybrid TTNN + HF
text-decoder rerun + HF char-prep helpers; see
``tt/speech_to_speech_model.py`` docstring for the exact boundary).

Usage::

    python models/demos/facebook_seamless_m4t_v2_large/demo/demo_s2st.py \\
        --wav models/demos/facebook_seamless_m4t_v2_large/demo/inputs/sample_hello.wav \\
        --src-lang eng \\
        --tgt-lang fra \\
        --out /tmp/s2st_test.wav

Output is a 16 kHz mono int16 WAV. The script also prints the predicted
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
    arr = np.clip(arr, -1.0, 1.0)
    arr_i16 = (arr * 32767.0).astype(np.int16)
    wav.write(path, sampling_rate, arr_i16)


@app.command()
def main(
    wav: str = typer.Option(..., "--wav", help="Path to a WAV file (any sample rate)."),
    src_lang: str = typer.Option("eng", "--src-lang", help="Source language code (e.g. eng)."),
    tgt_lang: str = typer.Option(..., "--tgt-lang", help="Target language code (e.g. fra)."),
    out: str = typer.Option(..., "--out", help="Output WAV path (16 kHz mono)."),
    speaker_id: int = typer.Option(0, "--speaker-id", help="Vocoder speaker id (default 0)."),
    max_new_tokens: int = typer.Option(
        128, "--max-new-tokens", help="AR text-generation budget (includes 2-token prefix)."
    ),
    max_seconds: float = typer.Option(
        5.0, "--max-seconds", help="Truncate audio to this many seconds before feature extraction."
    ),
    skip_hf: bool = typer.Option(False, "--skip-hf", help="Skip the HF reference run (TTNN only)."),
):
    """Translate ``--wav`` (in ``--src-lang``) into speech for ``--tgt-lang``."""
    import time

    from transformers import AutoProcessor

    import ttnn
    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_speech_model import SpeechToSpeechModel

    typer.echo("[demo] loading HF checkpoint into host memory ...")
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    typer.echo("[demo] opening TTNN device 0 ...")
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        typer.echo(
            "[demo] building SpeechToSpeechModel ("
            "TTNN speech_encoder + text_generator + t2u_generator + code_hifigan_vocoder; "
            "HF host text_decoder rerun + char prep) ..."
        )
        model = SpeechToSpeechModel(
            device=dev,
            hf_state_dict=hf_sd,
            processor=processor,
        )

        typer.echo(f"[demo] synthesising {wav!s}  ({src_lang} -> {tgt_lang}) ...")
        t0 = time.time()
        audio = model.synthesize(
            audio_path=wav,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            speaker_id=speaker_id,
            max_new_tokens=max_new_tokens,
            max_audio_seconds=max_seconds,
        )
        dt = time.time() - t0
    finally:
        ttnn.close_device(dev)

    duration_sec = float(audio.shape[-1]) / 16000.0
    _save_wav(out, audio, sampling_rate=16000)
    typer.echo("")
    typer.echo(f"Generated: {out}, duration: {duration_sec:.3f} seconds (wall-clock {dt:.2f} s)")

    if not skip_hf:
        typer.echo("[demo] running HF reference for comparison ...")
        hf_audio, hf_seconds = _run_hf_reference(
            wav_path=wav,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            speaker_id=speaker_id,
            max_new_tokens=max_new_tokens,
            max_seconds=max_seconds,
        )
        hf_out = out.rsplit(".", 1)
        hf_out_path = (hf_out[0] + ".hf." + hf_out[1]) if len(hf_out) == 2 else (out + ".hf.wav")
        _save_wav(hf_out_path, hf_audio, sampling_rate=16000)
        typer.echo(f"HF: {hf_out_path}, duration: {hf_seconds:.3f} seconds")

        # Best-effort re-ASR.
        ttnn_reasr = _try_reasr(out, lang=tgt_lang)
        hf_reasr = _try_reasr(hf_out_path, lang=tgt_lang)
        typer.echo(f"TTNN re-ASR : {ttnn_reasr!r}")
        typer.echo(f"HF   re-ASR : {hf_reasr!r}")


def _run_hf_reference(
    wav_path: str,
    src_lang: str,
    tgt_lang: str,
    speaker_id: int,
    max_new_tokens: int,
    max_seconds: float,
):
    """Run HF ``SeamlessM4Tv2ForSpeechToSpeech.generate`` and return audio + duration_s."""
    import numpy as np
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import _load_wav_to_16k_mono

    audio = _load_wav_to_16k_mono(wav_path)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * 16000)]

    proc = AutoProcessor.from_pretrained(wl.HF_PATH)
    feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(wl.HF_PATH, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_features=feats["input_features"],
            attention_mask=feats["attention_mask"],
            tgt_lang=tgt_lang,
            speaker_id=speaker_id,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
    if isinstance(out, tuple):
        waveform, waveform_lengths = out[0], out[1]
    else:
        waveform, waveform_lengths = out.waveform, out.waveform_lengths
    arr = waveform[0].detach().cpu().numpy().astype(np.float32)
    length = int(waveform_lengths.view(-1)[0].item()) if waveform_lengths is not None else int(arr.shape[-1])
    length = max(0, min(length, int(arr.shape[-1])))
    del model
    return arr[:length], float(length) / 16000.0


def _try_reasr(wav_path: str, lang: str = "eng"):
    """Best-effort: transcribe ``wav_path`` via HF SeamlessM4Tv2ForSpeechToText."""
    try:
        import numpy as np
        import scipy.io.wavfile as wav
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl

        sr, data = wav.read(wav_path)
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.float32:
            audio = data
        else:
            audio = data.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import torchaudio

            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, 16000)
            audio = t.squeeze(0).numpy()

        proc = AutoProcessor.from_pretrained(wl.HF_PATH)
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(wl.HF_PATH, torch_dtype=torch.float32)
        model.eval()
        inputs = proc(audios=audio.astype(np.float32), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                input_features=inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                tgt_lang=lang,
                do_sample=False,
                num_beams=1,
                max_new_tokens=32,
            )
        if hasattr(out, "sequences"):
            seq = out.sequences
        else:
            seq = out
        text = proc.decode(seq[0].tolist(), skip_special_tokens=True)
        del model
        return text
    except Exception as e:
        print(f"[reasr-warning] {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    app()
