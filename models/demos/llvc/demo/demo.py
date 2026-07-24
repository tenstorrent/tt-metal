# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LLVC TTNN demo: convert audio in streaming or non-streaming mode.

Examples
--------
# Random-weight smoke run on a synthetic tone (no checkpoint needed):
python models/demos/llvc/demo/demo.py --synthetic --stream

# Real checkpoint on a wav file / folder (KoeAI weights from `download_models.py`):
python models/demos/llvc/demo/demo.py \
    --config experiments/llvc/config.json \
    --checkpoint llvc_models/models/checkpoints/llvc/G_500000.pth \
    --input test_wavs --out-dir converted_out --stream --chunk-factor 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.llvc.tt.config import LLVCConfig
from models.demos.llvc.tt.model import create_llvc
from models.demos.llvc.tt.state_io import load_llvc_config_and_model


def _glob_audio(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files: list[str] = []
    for ext in ("wav", "mp3", "flac"):
        files.extend(str(p) for p in Path(path).rglob(f"*.{ext}"))
    return sorted(files)


def _resample(audio: torch.Tensor, sr: int, target: int) -> torch.Tensor:
    """Linear resample via torch interpolation (no torchaudio/librosa dependency)."""
    if sr == target:
        return audio
    n = int(round(audio.shape[-1] * target / sr))
    a = audio.reshape(1, 1, -1)
    a = torch.nn.functional.interpolate(a, size=n, mode="linear", align_corners=False)
    return a.reshape(-1)


def _load_audio(path: str, sample_rate: int) -> torch.Tensor:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    audio = torch.from_numpy(data).mean(dim=1)  # mono [T]
    return _resample(audio, sr, sample_rate)


def _safe_output_path(out_dir: str, name: str) -> str:
    """Resolve ``name`` inside ``out_dir``, rejecting path-traversal in ``name``.

    ``name`` is reduced to its basename and the resolved destination is required
    to stay within ``out_dir`` so a crafted input filename cannot escape it.
    """
    safe_name = os.path.basename(name)
    out_dir_abs = os.path.abspath(out_dir)
    dest = os.path.abspath(os.path.join(out_dir_abs, safe_name))
    if os.path.commonpath([out_dir_abs, dest]) != out_dir_abs:
        raise ValueError(f"Unsafe output path derived from {name!r}")
    return dest


def _save_audio(audio: torch.Tensor, path: str, sample_rate: int) -> None:
    import soundfile as sf

    audio = audio.detach().cpu().reshape(-1).float().numpy()
    sf.write(path, audio, sample_rate)


def run_demo(args: argparse.Namespace, *, device: ttnn.Device) -> None:
    if args.config and args.checkpoint:
        logger.info("Loading LLVC config+checkpoint: {} / {}", args.config, args.checkpoint)
        config, reference = load_llvc_config_and_model(args.config, args.checkpoint)
        model = create_llvc(config, device=device, reference=reference)
    else:
        logger.warning("No checkpoint provided: running with random weights (smoke test only).")
        config = LLVCConfig()
        model = create_llvc(config, device=device)

    sr = config.sample_rate

    if args.synthetic:
        t = torch.linspace(0, 1.0, sr, dtype=torch.float32)
        audio = 0.3 * torch.sin(2 * torch.pi * 220.0 * t)
        _run_one(model, audio, sr, args, out_name="synthetic.wav")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    files = _glob_audio(args.input)
    if not files:
        raise FileNotFoundError(f"No audio files found under {args.input}")
    rtfs, latencies = [], []
    for fname in files:
        audio = _load_audio(fname, sr)
        out, rtf, latency = _run_one(model, audio, sr, args, out_name=os.path.basename(fname))
        if rtf is not None:
            rtfs.append(rtf)
            latencies.append(latency)
    if rtfs:
        logger.info(
            "Mean RTF: {:.3f}  Mean chunk latency: {:.2f} ms", sum(rtfs) / len(rtfs), sum(latencies) / len(latencies)
        )


def _run_one(model, audio, sr, args, *, out_name):
    if args.stream:
        out, rtf, latency = model.stream(audio, chunk_factor=args.chunk_factor)
        logger.info("[{}] streaming RTF={:.3f} chunk_latency={:.2f}ms", out_name, rtf, latency)
    else:
        out = model(audio)
        rtf, latency = None, None
        logger.info("[{}] non-streaming conversion done, out shape {}", out_name, tuple(out.shape))
    if not args.synthetic:
        _save_audio(out.squeeze(0), _safe_output_path(args.out_dir, out_name), sr)
    return out, rtf, latency


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLVC TTNN demo")
    p.add_argument("--config", type=str, default=None, help="KoeAI config.json path")
    p.add_argument("--checkpoint", type=str, default=None, help="KoeAI generator checkpoint (.pth)")
    p.add_argument("--input", type=str, default="test_wavs", help="Audio file or directory")
    p.add_argument("--out-dir", type=str, default="converted_out", help="Output directory")
    p.add_argument("--stream", action="store_true", help="Use streaming (chunked) inference")
    p.add_argument("--chunk-factor", type=int, default=1, help="Chunk size multiplier (latency vs RTF)")
    p.add_argument("--synthetic", action="store_true", help="Run on a synthetic tone (no files/checkpoint)")
    p.add_argument("--device-id", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # conv1d's halo step allocates config tensors in the L1-small region, so it
    # must be non-zero; a trace region is reserved for Stage-2 trace capture.
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768, trace_region_size=23887872)
    device.enable_program_cache()
    try:
        run_demo(args, device=device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
