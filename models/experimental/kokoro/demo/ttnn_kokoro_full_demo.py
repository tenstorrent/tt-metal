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


def _log_mel(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Log-mel spectrogram ``[n_mels, frames]`` for a 1-D waveform.

    Used for a phase-invariant audio-similarity metric: raw-waveform PCC collapses to ~0 when two
    otherwise-identical signals are time-shifted or the on-device source path decorrelates phase,
    whereas the log-mel envelope compares per-band energy over time and is robust to both.
    """
    import librosa
    import numpy as np

    x = audio.detach().float().flatten().cpu().numpy().astype(np.float32)
    mel = librosa.feature.melspectrogram(y=x, sr=sample_rate, n_fft=1024, hop_length=256, n_mels=80, power=2.0)
    return torch.from_numpy(librosa.power_to_db(mel, ref=np.max))


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
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Metal-trace the decoder (asr/F0/N/s -> audio) — captured once per aligned length in a "
            "warmup pass and replayed in the measured loop. On by default; pass --no-trace to run the "
            "eager decoder. Reserves a DRAM trace region; forces the deterministic RNG path."
        ),
    )
    parser.add_argument(
        "--trace-region-size",
        type=int,
        default=200_000_000,
        help="DRAM trace region bytes when --trace is set.",
    )
    parser.add_argument(
        "--pcc-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the reference HuggingFace KModel (CPU float32) on each chunk and report the "
            "PCC of the TT audio against it. On by default; pass --no-pcc-check to skip."
        ),
    )
    args = parser.parse_args()

    try:
        from kokoro import KPipeline
    except ImportError:
        logger.error('Install upstream kokoro: pip install "kokoro>=0.9.2"')
        return 2

    from models.common.utility_functions import comp_pcc
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

    device = ttnn.open_device(
        device_id=0,
        l1_small_size=int(args.l1_small_size),
        trace_region_size=int(args.trace_region_size) if args.trace else 0,
    )
    # Reuse compiled programs across identical-shape op calls (the prosody LSTM alone dispatches
    # the same gate matmuls ~T_tokens times per direction). Without this every call rebuilds the
    # program host-side — a large slice of cold latency. Safe/standard; identical numerics.
    device.enable_program_cache()
    model = None
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
            trace=bool(args.trace),
        )
        # --trace decodes via captured metal traces, which require the deterministic RNG path.
        if args.trace:
            logger.info("--trace: decoder runs via metal trace (deterministic RNG forced)")
        logger.info(
            f"use_torch_stft_fallback={use_stft_fallback} use_torch_phase_fallback={use_phase_fallback} "
            f"activations_in_l1={activations_in_l1} disable_complex={args.disable_complex}"
        )
        wave_chunks: list[torch.Tensor] = []
        torch.manual_seed(0)

        def _ref_s_for(phonemes: str) -> torch.Tensor:
            ref_s = pack[len(phonemes) - 1].to("cpu")
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)
            return ref_s.float()

        # --- Warmup pass to capture traces. ---
        # A metal trace is captured on the FIRST decoder forward at each aligned-length bucket and
        # only replayed on LATER forwards of that same bucket. Without a warmup, the measured loop's
        # first (and possibly only) forward at each length just captures — never replays — so the
        # trace win never shows. Run one warmup forward per chunk here to populate every trace the
        # measured loop will hit; the timed loop below then replays them. Output is discarded, but the
        # readback forces the capture to fully resolve on device.
        if args.trace:
            logger.info("--trace: warmup pass (capturing decoder trace per chunk length)...")
            warm_t0 = time.perf_counter()
            for chunk_idx, result in enumerate(results):
                phonemes = result.phonemes
                if not phonemes:
                    continue
                warm_out = model(phonemes=phonemes, ref_s=_ref_s_for(phonemes), speed=args.speed, deterministic=True)
                warm_out.audio.detach()  # force device readback so the capture resolves
            captures = model._trace_mgr.captures if model._trace_mgr is not None else 0
            captures_a = model._trace_mgr_a.captures if model._trace_mgr_a is not None else 0
            logger.info(
                f"warmup complete in {time.perf_counter() - warm_t0:.3f}s "
                f"(trace captures: decoder={captures} traceA={captures_a})"
            )

        # --- Performance metrics accumulators. ---
        # Wall-clock (incl. host-driven prosody/LSTM loops + device compute) is what an end
        # user perceives, so latency/RTF/throughput are measured against perf_counter walls.
        sample_rate_hz = KokoroConfig.sample_rate_hz
        total_inference_s = 0.0  # sum of per-chunk forward latencies
        time_to_first_audio_s: float | None = None  # wall from loop start to first chunk's audio
        total_chars = 0  # input text characters synthesized (for char/s throughput)
        per_chunk_metrics: list[dict] = []
        chunk_mel_pccs: list[float] = []  # per-chunk log-mel PCC (phase/shift-tolerant) vs reference KModel
        ref_wave_chunks: list[torch.Tensor] = []  # reference HF audio chunks (saved when --pcc-check)
        loop_t0 = time.perf_counter()

        for chunk_idx, result in enumerate(results):
            phonemes = result.phonemes
            if not phonemes:
                logger.warning(f"Skipping empty phonemes chunk index={chunk_idx}")
                continue
            ref_s = _ref_s_for(phonemes)

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

            pcc_str = ""
            if args.pcc_check:
                # Reference HF KModel on CPU (float32); same phonemes/ref_s/speed as the TT forward.
                ref_audio = ref_model(phonemes=phonemes, ref_s=ref_s, speed=args.speed).detach().float().flatten()
                ref_wave_chunks.append(ref_audio)
                # Lengths can differ by a few samples (pred_dur rounding + trace-bucket trimming);
                # compare on the common prefix, which carries essentially all the signal energy.
                # Log-mel PCC: phase/shift-tolerant, tracks perceptual similarity. Compare on the
                # common frame prefix (frame counts differ when TT/ref durations drift).
                ref_mel = _log_mel(ref_audio, sample_rate_hz)
                tt_mel = _log_mel(chunk_audio, sample_rate_hz)
                f = min(ref_mel.shape[1], tt_mel.shape[1])
                _, mel_pcc = comp_pcc(ref_mel[:, :f], tt_mel[:, :f], pcc=0.0)
                chunk_mel_pccs.append(float(mel_pcc))
                pcc_str = f" mel_pcc={mel_pcc:.4f} " f"(tt_len={chunk_audio.numel()} ref_len={ref_audio.numel()})"

            logger.info(
                f"Chunk {chunk_idx}: phoneme_len={len(phonemes)} chars={chunk_chars} "
                f"samples={chunk_samples} audio_s={chunk_audio_s:.2f} infer_s={chunk_infer_s:.3f} "
                f"rtf={per_chunk_metrics[-1]['rtf']:.3f} source=tt{pcc_str}"
            )

        if not wave_chunks:
            logger.error("No audio produced from pipeline chunks.")
            return 3
        logger.info(f"program cache entries: {device.num_program_cache_entries()}")
        audio = torch.cat(wave_chunks, dim=0).numpy()
        ref_audio_full = torch.cat(ref_wave_chunks, dim=0).numpy() if ref_wave_chunks else None
    finally:
        if model is not None:
            model.release_traces()  # free captured traces + persistent buffers before device close
        ttnn.close_device(device)

    sf.write(str(out_path), audio, KokoroConfig.sample_rate_hz)
    logger.info(f"Wrote {out_path.resolve()} samples={audio.shape[-1]} sr={KokoroConfig.sample_rate_hz}")

    if ref_audio_full is not None:
        ref_path = out_path.with_name(f"{out_path.stem}_ref{out_path.suffix}")
        sf.write(str(ref_path), ref_audio_full, KokoroConfig.sample_rate_hz)
        logger.info(
            f"Wrote reference HF audio {ref_path.resolve()} samples={ref_audio_full.shape[-1]} "
            f"sr={KokoroConfig.sample_rate_hz}"
        )

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
    if chunk_mel_pccs:
        mean_mel = sum(chunk_mel_pccs) / len(chunk_mel_pccs)
        logger.info(
            f"  {'mel PCC vs ref':<22}: mean={mean_mel:.4f} min={min(chunk_mel_pccs):.4f}  (log-mel, phase/shift-tolerant)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
