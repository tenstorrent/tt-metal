# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_ASR_BACKEND = "transformers_whisper"
DEFAULT_ASR_MODEL = "openai/whisper-small.en"


class ASRBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class WERResult:
    source_audio_path: str
    generated_audio_path: str
    backend: str
    model_id: str
    source_transcript: str
    generated_transcript: str
    wer: float


def _load_audio_16khz_mono(audio_path: str | Path) -> np.ndarray:
    if importlib.util.find_spec("librosa") is None or importlib.util.find_spec("soundfile") is None:
        raise ASRBackendError(
            "Audio loading for WER requires optional dependencies. " "Install them with: pip install librosa soundfile"
        )

    import librosa
    import soundfile as sf

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    audio, sample_rate = sf.read(path)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return audio


def normalize_transcript(text: str) -> str:
    if importlib.util.find_spec("jiwer") is None:
        raise ASRBackendError("Transcript normalization for WER requires jiwer. Install it with: pip install jiwer")

    import jiwer

    transforms = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    normalized = transforms(text)
    if not normalized:
        return ""
    return " ".join(normalized[0])


def _load_transformers_whisper(model_id: str, device: str):
    if importlib.util.find_spec("transformers") is None:
        raise ASRBackendError(
            "Transformers-based WER requires optional dependencies. " "Install them with: pip install transformers"
        )

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    model = model.eval().to(device)
    return processor, model


def transcribe_audio(
    audio_path: str | Path,
    *,
    backend: str = DEFAULT_ASR_BACKEND,
    model_id: str = DEFAULT_ASR_MODEL,
    device: str = "cpu",
) -> str:
    if backend != "transformers_whisper":
        raise ValueError(f"Unsupported ASR backend: {backend}")

    processor, model = _load_transformers_whisper(model_id=model_id, device=device)
    waveform = _load_audio_16khz_mono(audio_path)
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)
    attention_mask = None
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        attention_mask = inputs.attention_mask.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(input_features=input_features, attention_mask=attention_mask)
    transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcript.strip()


def compute_wer(
    source_audio_path: str | Path,
    generated_audio_path: str | Path,
    *,
    backend: str = DEFAULT_ASR_BACKEND,
    model_id: str = DEFAULT_ASR_MODEL,
    device: str = "cpu",
) -> WERResult:
    if importlib.util.find_spec("jiwer") is None:
        raise ASRBackendError("WER computation requires jiwer. Install it with: pip install jiwer")

    import jiwer

    source_transcript = transcribe_audio(
        source_audio_path,
        backend=backend,
        model_id=model_id,
        device=device,
    )
    generated_transcript = transcribe_audio(
        generated_audio_path,
        backend=backend,
        model_id=model_id,
        device=device,
    )
    normalized_source = normalize_transcript(source_transcript)
    normalized_generated = normalize_transcript(generated_transcript)
    wer = float(jiwer.wer(normalized_source, normalized_generated))

    return WERResult(
        source_audio_path=str(source_audio_path),
        generated_audio_path=str(generated_audio_path),
        backend=backend,
        model_id=model_id,
        source_transcript=source_transcript,
        generated_transcript=generated_transcript,
        wer=wer,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute content-preservation WER between the original source audio and generated RVC audio."
    )
    parser.add_argument("--source-audio", required=True, help="Original source audio file.")
    parser.add_argument("--generated-audio", required=True, help="Generated audio file from the RVC pipeline.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_ASR_BACKEND,
        choices=["transformers_whisper"],
        help="ASR backend.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_ASR_MODEL,
        help="ASR model identifier for the selected backend.",
    )
    parser.add_argument("--device", default="cpu", help="Execution device for ASR.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = compute_wer(
        source_audio_path=args.source_audio,
        generated_audio_path=args.generated_audio,
        backend=args.backend,
        model_id=args.model_id,
        device=args.device,
    )
    print(f"wer={result.wer:.6f}")
    print(f"source_transcript={result.source_transcript}")
    print(f"generated_transcript={result.generated_transcript}")


if __name__ == "__main__":
    main()
