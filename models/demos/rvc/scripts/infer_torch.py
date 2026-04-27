# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

import soundfile as sf
import torch
from scipy import signal

from models.demos.rvc.evals import DEFAULT_WHISPER_MODEL, count_whisper_token
from models.demos.rvc.evals.speaker_similarity import compute_speaker_similarity
from models.demos.rvc.torch_impl.vc.pipeline import Pipeline, ah, bh
from models.demos.rvc.utils.audio import load_audio
from models.demos.rvc.utils.f0 import F0Method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RVC Nano inference from a wav file.")
    parser.add_argument("-o", "--output", required=True, help="Output audio path (wav).")
    parser.add_argument("--speaker-id", type=int, default=0, help="Speaker ID (default: 0).")
    parser.add_argument(
        "--f0-method", default="rapt", choices=["rapt", "dio", "harvest", "crepe", "rmvpe"], help="F0 method."
    )
    parser.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--index-rate", type=float, default=0.75, help="Index blending rate.")
    parser.add_argument("--file-index", default=None, help="Optional FAISS feature index path.")
    parser.add_argument("--rms-mix-rate", type=float, default=0.25, help="RMS mix rate.")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect rate.")
    parser.add_argument("--iters", type=int, default=1, help="Timed inference iterations.")
    parser.add_argument(
        "--count-token", action="store_true", help="Transcribe output audio with Whisper and count transcript tokens."
    )
    parser.add_argument(
        "--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Whisper model identifier for --count-token."
    )
    parser.add_argument("--whisper-device", default="cpu", help="Execution device for Whisper.")
    parser.add_argument(
        "--compute-embedding-similarity",
        action="store_true",
        help="Compute speaker embedding cosine similarity between input and output audio.",
    )
    parser.add_argument("--speaker-similarity-device", default="cpu", help="Execution device for speaker similarity.")
    return parser.parse_args()


def prepare_audio_input(sample_rate: int) -> torch.Tensor:
    audio = load_audio(sample_rate)
    audio_max = torch.abs(audio).max().item()
    if audio_max > 1:
        audio /= audio_max
    audio = signal.filtfilt(bh, ah, audio)
    return torch.from_numpy(audio.copy()).unsqueeze(0).to(torch.float32)


def main() -> None:
    args = parse_args()

    if args.iters < 1:
        raise ValueError("--iters must be at least 1.")

    if not os.getenv("RVC_CONFIGS_DIR"):
        raise RuntimeError("RVC_CONFIGS_DIR is not set.")
    if not os.getenv("RVC_ASSETS_DIR"):
        raise RuntimeError("RVC_ASSETS_DIR is not set.")

    pipe = Pipeline(
        if_f0=True,
        version="v1",
        num="48k",
        speaker_id=args.speaker_id,
        f0_up_key=args.f0_up_key,
        f0_method=F0Method.from_str(args.f0_method),
        index_rate=args.index_rate,
        file_index=args.file_index,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
    )

    audio = None
    start_time = time.time()
    for _ in range(args.iters):
        audio = pipe.infer()
    end_time = time.time()
    avg_sec = (end_time - start_time) / args.iters

    if audio is None:
        raise RuntimeError("Inference did not produce output audio.")

    audio_torch = audio.detach().cpu()
    audio_np = audio_torch.numpy()
    sf.write(args.output, audio_np, pipe.tgt_sr, subtype="PCM_16")

    if args.count_token:
        transcript, num_tokens = count_whisper_token(
            audio_torch,
            pipe.tgt_sr,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
        )
        print(f"whisper_transcript={transcript}")
        print(f"num_whisper_tokens={num_tokens}")
        print(f"tokens/second={num_tokens / avg_sec:.2f}")

    if args.compute_embedding_similarity:
        input_audio = prepare_audio_input(pipe.sr)[0]
        speaker_similarity = compute_speaker_similarity(
            input_audio,
            audio_torch,
            reference_sample_rate=pipe.sr,
            candidate_sample_rate=pipe.tgt_sr,
            device=args.speaker_similarity_device,
        )
        print(f"speaker_similarity={speaker_similarity:.6f}")
        print(f"speaker_similarity_percent={speaker_similarity * 100:.2f}")
        print(f"speaker_similarity_pass={str(speaker_similarity > 0.75).lower()}")

    output_duration_sec = audio_np.shape[0] / pipe.tgt_sr
    rtf = avg_sec / output_duration_sec if output_duration_sec > 0 else float("inf")
    print(f"avg_sec={avg_sec:.6f}")
    print(f"output_duration_sec={output_duration_sec:.6f}")
    print(f"rtf={rtf:.6f}")
    print(f"output_shape={audio_np.shape}")
    print(f"num_output_samples={audio_np.shape[0]}")


if __name__ == "__main__":
    main()
