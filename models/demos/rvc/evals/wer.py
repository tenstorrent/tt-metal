# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Content-preservation WER evaluation for RVC outputs."""

from __future__ import annotations

import argparse
import importlib.util
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import av
import librosa
import numpy as np
import torch

BASE_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
INPUT_AUDIO_FILE = "sample-speech.wav"
OUTPUT_AUDIO_FILE = "output/output_torch.wav"

DEFAULT_BACKEND = "transformers_whisper"
DEFAULT_ASR_MODEL = "openai/whisper-large-v3"
DEFAULT_MAX_WER = 2.5


class ASRBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class WERResult:
    backend: str
    model_id: str
    reference_transcript: str
    candidate_transcript: str
    wer: float


def _audio_to_float32_mono(input_file, output_file, output_format, sample_rate):
    inp = av.open(input_file, "r")
    out = av.open(output_file, "w", format=output_format)
    if output_format == "ogg":
        output_format = "libvorbis"
    if output_format == "f32le":
        output_format = "pcm_f32le"

    stream = out.add_stream(output_format, rate=sample_rate)
    try:
        stream.layout = "mono"
    except Exception:
        pass

    for frame in inp.decode(audio=0):
        for packet in stream.encode(frame):
            out.mux(packet)

    out.close()
    inp.close()


def load_audio_input(sample_rate: int) -> torch.Tensor:
    audio_path = (BASE_DIRECTORY / INPUT_AUDIO_FILE).resolve()
    if not str(audio_path).startswith(str(BASE_DIRECTORY.resolve())):
        raise RuntimeError(f"Audio file must be located under {BASE_DIRECTORY}")
    if not audio_path.exists():
        raise RuntimeError(f"Audio file does not exist: {audio_path}")

    try:
        with open(audio_path, "rb") as infile:
            with BytesIO() as outfile:
                _audio_to_float32_mono(infile, outfile, "f32le", sample_rate)
                waveform = np.frombuffer(outfile.getvalue(), np.float32).flatten()
                return torch.from_numpy(waveform)
    except AttributeError:
        waveform, original_sr = librosa.load(str(audio_path), sr=None, mono=True)
        waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=sample_rate)
        return torch.from_numpy(waveform.astype(np.float32))


def load_audio_output(sample_rate: int) -> torch.Tensor:
    audio_path = (BASE_DIRECTORY / OUTPUT_AUDIO_FILE).resolve()
    if not str(audio_path).startswith(str(BASE_DIRECTORY.resolve())):
        raise RuntimeError(f"Audio file must be located under {BASE_DIRECTORY}")
    if not audio_path.exists():
        raise RuntimeError(f"Audio file does not exist: {audio_path}")

    try:
        with open(audio_path, "rb") as infile:
            with BytesIO() as outfile:
                _audio_to_float32_mono(infile, outfile, "f32le", sample_rate)
                waveform = np.frombuffer(outfile.getvalue(), np.float32).flatten()
                return torch.from_numpy(waveform)
    except AttributeError:
        waveform, original_sr = librosa.load(str(audio_path), sr=None, mono=True)
        waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=sample_rate)
        return torch.from_numpy(waveform.astype(np.float32))


def normalize_transcript(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _word_edit_distance(reference_words: list[str], candidate_words: list[str]) -> int:
    rows = len(reference_words) + 1
    cols = len(candidate_words) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference_words[i - 1] == candidate_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _load_transformers_asr_pipeline(model_id: str, device: str):
    if importlib.util.find_spec("transformers") is None:
        raise ASRBackendError(
            "Transformers-based WER requires optional dependencies. " "Install them with: pip install transformers"
        )

    from transformers import pipeline

    if device == "cpu":
        pipeline_device = -1
        torch_dtype = torch.float32
    else:
        pipeline_device = device
        torch_dtype = torch.float16

    try:
        return pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=pipeline_device,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        raise ASRBackendError(
            "Failed to initialize the ASR backend. "
            "If the model is not cached locally, this step may require network access. "
            f"model_id={model_id!r}"
        ) from exc


def transcribe_audio(
    waveform: torch.Tensor,
    *,
    backend: str = DEFAULT_BACKEND,
    model_id: str = DEFAULT_ASR_MODEL,
    device: str = "cpu",
) -> str:
    if backend != "transformers_whisper":
        raise ValueError(f"Unsupported ASR backend: {backend}")

    asr = _load_transformers_asr_pipeline(model_id=model_id, device=device)
    asr_input = {"array": waveform.detach().cpu().numpy(), "sampling_rate": 16000}
    if model_id.endswith(".en"):
        result = asr(asr_input, return_timestamps=True)
    else:
        result = asr(
            asr_input,
            return_timestamps=True,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
    return normalize_transcript(result["text"])


def compute_wer(
    *,
    backend: str = DEFAULT_BACKEND,
    model_id: str = DEFAULT_ASR_MODEL,
    device: str = "cpu",
) -> WERResult:
    reference_transcript = transcribe_audio(
        load_audio_input(16000),
        backend=backend,
        model_id=model_id,
        device=device,
    )
    candidate_transcript = transcribe_audio(
        load_audio_output(16000),
        backend=backend,
        model_id=model_id,
        device=device,
    )

    reference_words = reference_transcript.split()
    candidate_words = candidate_transcript.split()
    if not reference_words:
        raise ValueError("Reference transcript is empty after normalization.")

    distance = _word_edit_distance(reference_words, candidate_words)
    wer = 100.0 * distance / len(reference_words)
    return WERResult(
        backend=backend,
        model_id=model_id,
        reference_transcript=reference_transcript,
        candidate_transcript=candidate_transcript,
        wer=wer,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute content-preservation WER using fixed RVC demo audio.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["transformers_whisper"],
        help="ASR backend.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_ASR_MODEL,
        help="ASR model identifier for the selected backend.",
    )
    parser.add_argument("--device", default="cpu", help="Execution device for the ASR backend.")
    parser.add_argument(
        "--max-wer",
        type=float,
        default=DEFAULT_MAX_WER,
        help="Maximum acceptable WER percentage.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = compute_wer(
        backend=args.backend,
        model_id=args.model_id,
        device=args.device,
    )
    passed = result.wer < args.max_wer
    print(f"reference_transcript={result.reference_transcript}")
    print(f"candidate_transcript={result.candidate_transcript}")
    print(f"wer={result.wer:.6f}")
    print(f"threshold={args.max_wer:.6f}")
    print(f"pass={str(passed).lower()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
