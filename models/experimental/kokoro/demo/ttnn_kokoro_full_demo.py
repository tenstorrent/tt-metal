# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro-82M full TTNN demo (latest ``TTKModel`` port).

Run from the ``tt-metal`` repo root with this tree's venv (matches the local ``ttnn`` / Metal build)::

    source python_env/bin/activate
    python models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py --text "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger

import ttnn


def _find_local_checkpoint() -> Path | None:
    candidates = (
        Path("/home/ubuntu/ign-tt/kokoro/examples/checkpoints/kokoro-v1_0.pth"),
        Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots",
    )
    for path in candidates:
        if path.is_file():
            return path
        if path.is_dir():
            for child in path.rglob("kokoro-v1_0.pth"):
                return child
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro-82M full TTNN demo (TTKModel)")
    parser.add_argument("--text", type=str, default="Hello from Tenstorrent Kokoro full TTNN.")
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--lang-code", type=str, default="a")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="kokoro_experimental_ttnn.wav")
    parser.add_argument(
        "--l1-small-size",
        type=int,
        default=98304,
        help="TT device small L1 allocator size in bytes.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local kokoro-v1_0.pth; if omitted, auto-detect local cache or download from HuggingFace.",
    )
    parser.add_argument(
        "--torch-stft-fallback",
        action="store_true",
        help="Use PyTorch float32 STFT fallback in TT decoder for higher numerical parity.",
    )
    parser.add_argument(
        "--torch-phase-fallback",
        action="store_true",
        help="Use PyTorch float32 SineGen phase fallback for higher numerical parity.",
    )
    parser.add_argument(
        "--torch-sinegen",
        action="store_true",
        help="Deprecated alias: enables both --torch-stft-fallback and --torch-phase-fallback.",
    )
    args = parser.parse_args()

    try:
        from kokoro import KPipeline
    except ImportError:
        logger.error('Install upstream kokoro: pip install "kokoro>=0.9.2"')
        return 2

    from models.experimental.kokoro.reference.model import KModel
    from models.experimental.kokoro.tt.tt_kmodel import KokoroConfig, TTKModel, preprocess_tt_kmodel

    pipe = KPipeline(lang_code=args.lang_code, model=False)
    results = list(pipe(args.text, voice=args.voice, speed=args.speed))
    if not results:
        logger.error("Pipeline produced no chunks.")
        return 3
    pack = pipe.load_voice(args.voice)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = ttnn.open_device(device_id=0, l1_small_size=int(args.l1_small_size))
    try:
        checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else _find_local_checkpoint()
        if args.checkpoint and (checkpoint is None or not checkpoint.is_file()):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            return 2
        if checkpoint is None:
            logger.info("No local checkpoint found; KModel will download from HuggingFace if needed.")
        else:
            logger.info(f"Using checkpoint: {checkpoint}")

        ref_model = KModel(
            repo_id=KokoroConfig.repo_id,
            model=str(checkpoint) if checkpoint is not None else None,
            disable_complex=True,
        ).eval()
        params = preprocess_tt_kmodel(ref_model, device)

        use_stft_fallback = bool(args.torch_stft_fallback or args.torch_sinegen)
        use_phase_fallback = bool(args.torch_phase_fallback or args.torch_sinegen)
        if args.torch_sinegen:
            logger.warning("--torch-sinegen is deprecated; use --torch-stft-fallback and/or --torch-phase-fallback.")

        model = TTKModel(
            device,
            ref_model,
            params,
            use_torch_stft_fallback=use_stft_fallback,
            use_torch_phase_fallback=use_phase_fallback,
        )
        logger.info(f"use_torch_stft_fallback={use_stft_fallback} use_torch_phase_fallback={use_phase_fallback}")
        wave_chunks: list[torch.Tensor] = []
        torch.manual_seed(0)
        for chunk_idx, result in enumerate(results):
            phonemes = result.phonemes
            if not phonemes:
                logger.warning(f"Skipping empty phonemes chunk index={chunk_idx}")
                continue
            ref_s = pack[len(phonemes) - 1].to("cpu")
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)
            ref_s = ref_s.float()
            out = model(phonemes=phonemes, ref_s=ref_s, speed=args.speed, deterministic=True)
            wave_chunks.append(out.audio.detach().float().flatten())
            logger.info(f"Chunk {chunk_idx}: phoneme_len={len(phonemes)} samples={wave_chunks[-1].numel()} source=tt")

        if not wave_chunks:
            logger.error("No audio produced from pipeline chunks.")
            return 3
        audio = torch.cat(wave_chunks, dim=0).numpy()
    finally:
        ttnn.close_device(device)

    sf.write(str(out_path), audio, KokoroConfig.sample_rate_hz)
    logger.info(f"Wrote {out_path.resolve()} samples={audio.shape[-1]} sr={KokoroConfig.sample_rate_hz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
