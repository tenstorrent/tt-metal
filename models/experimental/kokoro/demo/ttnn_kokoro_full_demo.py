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
import time
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
        "--l1-activations",
        action="store_true",
        help=(
            "Keep the generator upsample/resblock loop activations L1-resident (~4%% faster, "
            "PCC-neutral). Safe for short utterances; may OOM on very long inputs."
        ),
    )
    parser.add_argument(
        "--disable-complex",
        action="store_true",
        help=(
            "Use the istftnet disable_complex=True STFT formulation: reference KModel and the TT "
            "decoder both run the on-device CustomSTFT port (conv2d/conv_transpose2d, no fallback)."
        ),
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
            disable_complex=args.disable_complex,
        ).eval()
        params = preprocess_tt_kmodel(ref_model, device)

        use_stft_fallback = bool(args.torch_stft_fallback)
        use_phase_fallback = bool(args.torch_phase_fallback)
        activations_in_l1 = bool(args.l1_activations)

        model = TTKModel(
            device,
            ref_model,
            params,
            use_torch_stft_fallback=use_stft_fallback,
            use_torch_phase_fallback=use_phase_fallback,
            activations_in_l1=activations_in_l1,
            disable_complex=args.disable_complex,
        )
        logger.info(
            f"use_torch_stft_fallback={use_stft_fallback} use_torch_phase_fallback={use_phase_fallback} "
            f"activations_in_l1={activations_in_l1} disable_complex={args.disable_complex}"
        )
        wave_chunks: list[torch.Tensor] = []
        torch.manual_seed(0)

        # --- Performance metrics accumulators. ---
        # Wall-clock (incl. host-driven prosody/LSTM loops + device compute) is what an end
        # user perceives, so latency/RTF/throughput are measured against perf_counter walls.
        sample_rate_hz = KokoroConfig.sample_rate_hz
        total_inference_s = 0.0  # sum of per-chunk forward latencies
        time_to_first_audio_s: float | None = None  # wall from loop start to first chunk's audio
        total_chars = 0  # input text characters synthesized (for char/s throughput)
        per_chunk_metrics: list[dict] = []
        loop_t0 = time.perf_counter()

        for chunk_idx, result in enumerate(results):
            phonemes = result.phonemes
            if not phonemes:
                logger.warning(f"Skipping empty phonemes chunk index={chunk_idx}")
                continue
            ref_s = pack[len(phonemes) - 1].to("cpu")
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)
            ref_s = ref_s.float()

            # ``out.audio.detach().float()`` below forces a device readback, so the forward
            # is fully resolved by the time we stop the timer — no extra synchronize needed.
            chunk_t0 = time.perf_counter()
            out = model(phonemes=phonemes, ref_s=ref_s, speed=args.speed, deterministic=True)
            chunk_audio = out.audio.detach().float().flatten()
            chunk_t1 = time.perf_counter()

            if time_to_first_audio_s is None:
                time_to_first_audio_s = chunk_t1 - loop_t0

            chunk_infer_s = chunk_t1 - chunk_t0
            total_inference_s += chunk_infer_s
            # Input characters for this chunk (graphemes = original text; phonemes as fallback).
            chunk_chars = len(getattr(result, "graphemes", None) or phonemes)
            total_chars += chunk_chars

            chunk_samples = chunk_audio.numel()
            chunk_audio_s = chunk_samples / sample_rate_hz
            per_chunk_metrics.append(
                {
                    "phonemes": len(phonemes),
                    "chars": chunk_chars,
                    "samples": chunk_samples,
                    "audio_s": chunk_audio_s,
                    "infer_s": chunk_infer_s,
                    "rtf": (chunk_infer_s / chunk_audio_s) if chunk_audio_s > 0 else float("nan"),
                }
            )

            wave_chunks.append(chunk_audio)
            logger.info(
                f"Chunk {chunk_idx}: phoneme_len={len(phonemes)} chars={chunk_chars} "
                f"samples={chunk_samples} audio_s={chunk_audio_s:.2f} infer_s={chunk_infer_s:.3f} "
                f"rtf={per_chunk_metrics[-1]['rtf']:.3f} source=tt"
            )

        if not wave_chunks:
            logger.error("No audio produced from pipeline chunks.")
            return 3
        audio = torch.cat(wave_chunks, dim=0).numpy()
    finally:
        ttnn.close_device(device)

    sf.write(str(out_path), audio, KokoroConfig.sample_rate_hz)
    logger.info(f"Wrote {out_path.resolve()} samples={audio.shape[-1]} sr={KokoroConfig.sample_rate_hz}")

    # --- Performance summary. ---
    total_audio_s = audio.shape[-1] / KokoroConfig.sample_rate_hz
    overall_rtf = (total_inference_s / total_audio_s) if total_audio_s > 0 else float("nan")
    throughput_char_s = (total_chars / total_inference_s) if total_inference_s > 0 else float("nan")
    ttfa = time_to_first_audio_s if time_to_first_audio_s is not None else float("nan")

    logger.info("Kokoro-82M demo performance metrics:")
    logger.info(f"  {'chunks':<22}: {len(per_chunk_metrics)}")
    logger.info(f"  {'input characters':<22}: {total_chars}")
    logger.info(f"  {'generated audio (s)':<22}: {total_audio_s:.2f}")
    logger.info(f"  {'total latency (s)':<22}: {total_inference_s:.3f}")
    logger.info(f"  {'time to first audio (s)':<22}: {ttfa:.3f}")
    logger.info(f"  {'real-time factor (RTF)':<22}: {overall_rtf:.3f}  (infer_s / audio_s, <1 = faster than real time)")
    logger.info(f"  {'throughput (char/s)':<22}: {throughput_char_s:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
