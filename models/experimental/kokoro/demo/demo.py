# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro-82M full TTNN demo (latest ``TTKModel`` port).

Text -> 24 kHz WAV through the full on-device pipeline (PL-BERT -> prosody predictor -> ASR text
encoder -> ISTFTNet decoder), with optional metal tracing of the decoder and a mel-PCC parity check
against the reference HuggingFace ``KModel`` running on CPU.

Two entry points share the same code path:

* **pytest** (CI + regression gates) — the parametrized cases in ``test_demo.py``::

      pytest "models/experimental/kokoro/demo/test_demo.py::test_demo[default]" -s --timeout=1200

* **CLI** (interactive), from the ``tt-metal`` repo root with this tree's venv (matches the local
  ``ttnn`` / Metal build)::

      source python_env/bin/activate
      python models/experimental/kokoro/demo/demo.py --text "..."

``run_demo`` is the programmatic entrypoint behind both. It owns the device lifecycle and returns a
results dict with ``generations`` (per-chunk records), ``statistics`` (perf + parity metrics) and
``model_params`` (the resolved run configuration).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tt.tt_kmodel import KokoroConfig, TTKModel, preprocess_tt_kmodel

DEFAULT_TEXT = "Hello from Tenstorrent Kokoro full TTNN."
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG_CODE = "a"
DEFAULT_SPEED = 1.0
DEFAULT_OUTPUT = "kokoro_experimental_ttnn.wav"
DEFAULT_L1_SMALL_SIZE = 98304
DEFAULT_TRACE_REGION_SIZE = 200_000_000
DEFAULT_SEED = 0


def find_local_checkpoint() -> Path | None:
    """Locate a local ``kokoro-v1_0.pth`` without hardcoding a machine-specific path.

    Search order: the ``KOKORO_CHECKPOINT`` env var (a file, or a directory to scan), then the
    HuggingFace hub cache. Returns ``None`` if nothing is found, in which case ``KModel`` downloads
    from HuggingFace. Use ``--checkpoint`` / ``checkpoint=`` to point at an explicit path.
    """
    candidates: list[Path] = []
    env_ckpt = os.environ.get("KOKORO_CHECKPOINT")
    if env_ckpt:
        candidates.append(Path(env_ckpt).expanduser())
    candidates.append(Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots")
    for path in candidates:
        if path.is_file():
            return path
        if path.is_dir():
            for child in path.rglob("kokoro-v1_0.pth"):
                return child
    return None


def _resolve_checkpoint(checkpoint: str | Path | None) -> Path | None:
    """Resolve an explicit checkpoint path, else auto-detect. ``None`` = let ``KModel`` download."""
    if checkpoint is not None:
        resolved = Path(checkpoint).expanduser()
        if not resolved.is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        return resolved
    return find_local_checkpoint()


def log_mel(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
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


def _resolve_tt_metal_commit() -> str:
    """Resolve the tt-metal commit used for this run."""
    repo_root = Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except Exception:
        commit = ""

    return commit or "unknown"


def _capture_traces(model: TTKModel, chunks: list, ref_s_for, speed: float) -> dict:
    """Warmup pass capturing one decoder metal trace per aligned chunk length.

    A metal trace is captured on the FIRST decoder forward at each aligned-length bucket and only
    replayed on LATER forwards of that same bucket. Without a warmup, the measured loop's first (and
    possibly only) forward at each length just captures — never replays — so the trace win never
    shows. Run one warmup forward per chunk here to populate every trace the measured loop will hit;
    the timed loop then replays them. Output is discarded, but the readback forces the capture to
    fully resolve on device.
    """
    logger.info("trace: warmup pass (capturing decoder trace per chunk length)...")
    warm_t0 = time.perf_counter()
    for result in chunks:
        phonemes = result.phonemes
        if not phonemes:
            continue
        warm_out = model(phonemes=phonemes, ref_s=ref_s_for(phonemes), speed=speed, deterministic=True)
        warm_out.audio.detach()  # force device readback so the capture resolves
    warm_s = time.perf_counter() - warm_t0

    trace_mgr = getattr(model, "_trace_mgr", None)
    trace_mgr_a = getattr(model, "_trace_mgr_a", None)
    stats = {
        "warmup_s": warm_s,
        "trace_captures_decoder": trace_mgr.captures if trace_mgr is not None else 0,
        "trace_captures_prosody": trace_mgr_a.captures if trace_mgr_a is not None else 0,
    }
    logger.info(
        f"warmup complete in {warm_s:.3f}s (trace captures: decoder={stats['trace_captures_decoder']} "
        f"traceA={stats['trace_captures_prosody']})"
    )
    return stats


def run_demo(
    text: str = DEFAULT_TEXT,
    *,
    voice: str = DEFAULT_VOICE,
    lang_code: str = DEFAULT_LANG_CODE,
    speed: float = DEFAULT_SPEED,
    output_path: str | Path | None = DEFAULT_OUTPUT,
    checkpoint: str | Path | None = None,
    l1_small_size: int = DEFAULT_L1_SMALL_SIZE,
    trace: bool = True,
    trace_region_size: int = DEFAULT_TRACE_REGION_SIZE,
    torch_stft_fallback: bool = False,
    torch_phase_fallback: bool = False,
    l1_activations: bool = False,
    disable_complex: bool = False,
    pcc_check: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Programmatic entrypoint for the Kokoro-82M TTNN demo.

    Synthesizes ``text`` on device (one forward per upstream ``KPipeline`` chunk), writes the
    concatenated 24 kHz waveform to ``output_path``, and — when ``pcc_check`` is set — scores each
    chunk's mel PCC against the reference CPU ``KModel`` and writes that reference audio alongside it.

    Returns a dict with keys:
        - generations: list[dict] with per-chunk text/audio/latency/mel-PCC records
        - statistics: perf + parity counters (latency, RTF, throughput, mel PCC, trace captures)
        - model_params: the resolved run configuration
        - audio_path / reference_audio_path: written WAV paths (``None`` when not written)
    """
    demo_t0 = time.perf_counter()

    try:
        from kokoro import KPipeline
    except ImportError:
        raise SystemExit('Install upstream kokoro: pip install "kokoro>=0.9.2"')

    # --- Flag-combination validation (fail fast, before the device is opened). ---
    # A metal trace captures a pure device graph, so ANY host round-trip inside the captured decoder
    # aborts the capture with an opaque TT_FATAL ("Reads/Writes are not supported during trace
    # capture") deep in the vocoder. Three configs do host work inside the traced region and so
    # require trace=False:
    #   * torch_phase_fallback — ``ttnn.to_torch`` in the SineGen phase chain (tt_sinegen.py).
    #   * torch_stft_fallback  — the float32 ``torch.stft`` / dense iSTFT runs on the host.
    #   * disable_complex      — the chunked CustomSTFT iSTFT branch round-trips every
    #     conv_transpose2d chunk through torch (tt_custom_stft.py ``_conv_transpose_branch_chunked``),
    #     which any audio long enough to take the chunked path will hit.
    if trace:
        host_in_traced_graph = [
            name
            for name, enabled in (
                ("torch_stft_fallback", torch_stft_fallback),
                ("torch_phase_fallback", torch_phase_fallback),
                ("disable_complex", disable_complex),
            )
            if enabled
        ]
        if host_in_traced_graph:
            raise SystemExit(
                f"{', '.join(host_in_traced_graph)} cannot be combined with trace=True (--trace, the "
                "default): these paths do host work inside the traced decoder, which aborts the metal "
                "trace capture. Re-run with --no-trace / trace=False."
            )

    resolved_checkpoint = _resolve_checkpoint(checkpoint)
    if resolved_checkpoint is None:
        logger.info("No local checkpoint found; KModel will download from HuggingFace if needed.")
    else:
        logger.info(f"Using checkpoint: {resolved_checkpoint}")

    # G2P + sentence chunking on the host (upstream kokoro); ``model=False`` keeps the torch vocoder out.
    pipe = KPipeline(lang_code=lang_code, model=False)
    chunks = list(pipe(text, voice=voice, speed=speed))
    if not chunks:
        raise SystemExit("Pipeline produced no chunks.")
    pack = pipe.load_voice(voice)

    out_path = Path(output_path) if output_path is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate_hz = KokoroConfig.sample_rate_hz
    statistics: dict = {}
    generations: list[dict] = []

    device = ttnn.open_device(
        device_id=0,
        l1_small_size=int(l1_small_size),
        trace_region_size=int(trace_region_size) if trace else 0,
    )
    model = None
    try:
        # Reuse compiled programs across identical-shape op calls (the prosody LSTM alone dispatches
        # the same gate matmuls ~T_tokens times per direction). Without this every call rebuilds the
        # program host-side — a large slice of cold latency. Safe/standard; identical numerics.
        device.enable_program_cache()

        ref_model = KModel(
            repo_id=KokoroConfig.repo_id,
            model=str(resolved_checkpoint) if resolved_checkpoint is not None else None,
            disable_complex=disable_complex,
        ).eval()
        params = preprocess_tt_kmodel(ref_model, device)

        model = TTKModel(
            device,
            ref_model,
            params,
            use_torch_stft_fallback=torch_stft_fallback,
            use_torch_phase_fallback=torch_phase_fallback,
            activations_in_l1=l1_activations,
            disable_complex=disable_complex,
            trace=trace,
        )
        # Tracing decodes via captured metal traces, which require the deterministic RNG path.
        if trace:
            logger.info("trace: decoder runs via metal trace (deterministic RNG forced)")
        logger.info(
            f"use_torch_stft_fallback={torch_stft_fallback} use_torch_phase_fallback={torch_phase_fallback} "
            f"activations_in_l1={l1_activations} disable_complex={disable_complex}"
        )
        torch.manual_seed(seed)

        def _ref_s_for(phonemes: str) -> torch.Tensor:
            ref_s = pack[len(phonemes) - 1].to("cpu")
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)
            return ref_s.float()

        if trace:
            statistics.update(_capture_traces(model, chunks, _ref_s_for, speed))
        else:
            statistics.update({"warmup_s": None, "trace_captures_decoder": 0, "trace_captures_prosody": 0})

        # --- Performance metrics accumulators. ---
        # Wall-clock (incl. host-driven prosody/LSTM loops + device compute) is what an end
        # user perceives, so latency/RTF/throughput are measured against perf_counter walls.
        wave_chunks: list[torch.Tensor] = []
        ref_wave_chunks: list[torch.Tensor] = []  # reference HF audio chunks (saved when pcc_check)
        chunk_mel_pccs: list[float] = []  # per-chunk log-mel PCC (phase/shift-tolerant) vs reference
        total_inference_s = 0.0  # sum of per-chunk forward latencies
        time_to_first_audio_s: float | None = None  # wall from loop start to first chunk's audio
        total_chars = 0  # input text characters synthesized (for char/s throughput)
        loop_t0 = time.perf_counter()

        for chunk_idx, result in enumerate(chunks):
            phonemes = result.phonemes
            if not phonemes:
                logger.warning(f"Skipping empty phonemes chunk index={chunk_idx}")
                continue
            ref_s = _ref_s_for(phonemes)

            # ``out.audio.detach().float()`` below forces a device readback, so the forward
            # is fully resolved by the time we stop the timer — no extra synchronize needed.
            chunk_t0 = time.perf_counter()
            out = model(phonemes=phonemes, ref_s=ref_s, speed=speed, deterministic=True)
            chunk_audio = out.audio.detach().float().flatten()
            chunk_t1 = time.perf_counter()

            if time_to_first_audio_s is None:
                time_to_first_audio_s = chunk_t1 - loop_t0

            chunk_infer_s = chunk_t1 - chunk_t0
            total_inference_s += chunk_infer_s
            # Input characters for this chunk (graphemes = original text; phonemes as fallback).
            graphemes = getattr(result, "graphemes", None) or phonemes
            chunk_chars = len(graphemes)
            total_chars += chunk_chars

            chunk_samples = chunk_audio.numel()
            chunk_audio_s = chunk_samples / sample_rate_hz
            wave_chunks.append(chunk_audio)

            record = {
                "index": chunk_idx + 1,
                "graphemes": str(graphemes),
                "phonemes": len(phonemes),
                "chars": chunk_chars,
                "samples": chunk_samples,
                "audio_s": chunk_audio_s,
                "infer_s": chunk_infer_s,
                "rtf": (chunk_infer_s / chunk_audio_s) if chunk_audio_s > 0 else float("nan"),
                "mel_pcc": None,
            }

            pcc_str = ""
            if pcc_check:
                # Reference HF KModel on CPU (float32); same phonemes/ref_s/speed as the TT forward.
                ref_audio = ref_model(phonemes=phonemes, ref_s=ref_s, speed=speed).detach().float().flatten()
                ref_wave_chunks.append(ref_audio)
                # Log-mel PCC: phase/shift-tolerant, tracks perceptual similarity. Lengths can differ
                # by a few samples (pred_dur rounding + trace-bucket trimming), so compare on the
                # common frame prefix, which carries essentially all the signal energy.
                ref_mel = log_mel(ref_audio, sample_rate_hz)
                tt_mel = log_mel(chunk_audio, sample_rate_hz)
                f = min(ref_mel.shape[1], tt_mel.shape[1])
                _, mel_pcc = comp_pcc(ref_mel[:, :f], tt_mel[:, :f], pcc=0.0)
                chunk_mel_pccs.append(float(mel_pcc))
                record["mel_pcc"] = float(mel_pcc)
                record["ref_samples"] = int(ref_audio.numel())
                pcc_str = f" mel_pcc={mel_pcc:.4f} (tt_len={chunk_audio.numel()} ref_len={ref_audio.numel()})"

            generations.append(record)
            logger.info(
                f"Chunk {chunk_idx}: phoneme_len={len(phonemes)} chars={chunk_chars} "
                f"samples={chunk_samples} audio_s={chunk_audio_s:.2f} infer_s={chunk_infer_s:.3f} "
                f"rtf={record['rtf']:.3f} source=tt{pcc_str}"
            )

        if not wave_chunks:
            raise SystemExit("No audio produced from pipeline chunks.")
        statistics["program_cache_entries"] = device.num_program_cache_entries()
        audio = torch.cat(wave_chunks, dim=0).numpy()
        ref_audio_full = torch.cat(ref_wave_chunks, dim=0).numpy() if ref_wave_chunks else None
    finally:
        if model is not None:
            model.release_traces()  # free captured traces + persistent buffers before device close
        ttnn.close_device(device)

    audio_path = None
    reference_audio_path = None
    if out_path is not None:
        sf.write(str(out_path), audio, sample_rate_hz)
        audio_path = out_path.resolve()
        logger.info(f"Wrote {audio_path} samples={audio.shape[-1]} sr={sample_rate_hz}")

        if ref_audio_full is not None:
            ref_path = out_path.with_name(f"{out_path.stem}_ref{out_path.suffix}")
            sf.write(str(ref_path), ref_audio_full, sample_rate_hz)
            reference_audio_path = ref_path.resolve()
            logger.info(
                f"Wrote reference HF audio {reference_audio_path} samples={ref_audio_full.shape[-1]} "
                f"sr={sample_rate_hz}"
            )

    # --- Performance summary. ---
    total_audio_s = audio.shape[-1] / sample_rate_hz
    statistics.update(
        {
            "chunks": len(generations),
            "input_characters": total_chars,
            "audio_samples": int(audio.shape[-1]),
            "generated_audio_s": total_audio_s,
            "total_latency_s": total_inference_s,
            "time_to_first_audio_s": time_to_first_audio_s,
            "real_time_factor": (total_inference_s / total_audio_s) if total_audio_s > 0 else float("nan"),
            "throughput_char_s": (total_chars / total_inference_s) if total_inference_s > 0 else float("nan"),
            "mel_pcc_mean": (sum(chunk_mel_pccs) / len(chunk_mel_pccs)) if chunk_mel_pccs else None,
            "mel_pcc_min": min(chunk_mel_pccs) if chunk_mel_pccs else None,
            "Full demo runtime": time.perf_counter() - demo_t0,
            "tt-metal_commit": _resolve_tt_metal_commit(),
        }
    )

    model_params = {
        "text": text,
        "voice": voice,
        "lang_code": lang_code,
        "speed": speed,
        "seed": seed,
        "sample_rate_hz": sample_rate_hz,
        "checkpoint": str(resolved_checkpoint) if resolved_checkpoint is not None else "hf-download",
        "l1_small_size": int(l1_small_size),
        "trace": bool(trace),
        "trace_region_size": int(trace_region_size) if trace else 0,
        "two_trace": bool(trace and os.environ.get("KOKORO_TRACE_A") == "1"),
        "torch_stft_fallback": bool(torch_stft_fallback),
        "torch_phase_fallback": bool(torch_phase_fallback),
        "l1_activations": bool(l1_activations),
        "disable_complex": bool(disable_complex),
        "pcc_check": bool(pcc_check),
    }

    return {
        "generations": generations,
        "statistics": statistics,
        "model_params": model_params,
        "audio_path": audio_path,
        "reference_audio_path": reference_audio_path,
    }


def _print_performance_metrics(results: dict) -> None:
    """Print performance metrics from results if available."""
    statistics = results.get("statistics") or {}
    if not statistics:
        return

    ttfa = statistics["time_to_first_audio_s"]
    logger.info("Kokoro-82M demo performance metrics:")
    logger.info(f"  {'chunks':<24}: {statistics['chunks']}")
    logger.info(f"  {'input characters':<24}: {statistics['input_characters']}")
    logger.info(f"  {'generated audio (s)':<24}: {statistics['generated_audio_s']:.2f}")
    logger.info(f"  {'total latency (s)':<24}: {statistics['total_latency_s']:.3f}")
    logger.info(f"  {'time to first audio (s)':<24}: {'n/a' if ttfa is None else f'{ttfa:.3f}'}")
    logger.info(
        f"  {'real-time factor (RTF)':<24}: {statistics['real_time_factor']:.3f}"
        "  (infer_s / audio_s, <1 = faster than real time)"
    )
    logger.info(f"  {'throughput (char/s)':<24}: {statistics['throughput_char_s']:.2f}")
    if statistics.get("warmup_s") is not None:
        logger.info(
            f"  {'trace warmup (s)':<24}: {statistics['warmup_s']:.3f}  "
            f"(captures: decoder={statistics['trace_captures_decoder']} "
            f"traceA={statistics['trace_captures_prosody']})"
        )
    logger.info(f"  {'program cache entries':<24}: {statistics['program_cache_entries']}")
    if statistics.get("mel_pcc_mean") is not None:
        logger.info(
            f"  {'mel PCC vs ref':<24}: mean={statistics['mel_pcc_mean']:.4f} "
            f"min={statistics['mel_pcc_min']:.4f}  (log-mel, phase/shift-tolerant)"
        )
    logger.info(f"  {'full demo runtime (s)':<24}: {statistics['Full demo runtime']:.2f}")
    tt_metal_commit = statistics.get("tt-metal_commit")
    if tt_metal_commit:
        logger.info(f"  {'tt-metal commit':<24}: {tt_metal_commit}")


def _print_model_params(results: dict) -> None:
    """Print the resolved run configuration."""
    model_params = results.get("model_params") or {}
    if not model_params:
        return
    logger.info("=== Model Parameters ===")
    for key in sorted(model_params):
        logger.info(f"{key}: {model_params[key]}")
    logger.info("=====================")


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kokoro-82M full TTNN demo (TTKModel)")
    p.add_argument("--text", type=str, default=DEFAULT_TEXT)
    p.add_argument("--voice", type=str, default=DEFAULT_VOICE)
    p.add_argument("--lang-code", type=str, default=DEFAULT_LANG_CODE)
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--l1-small-size",
        type=int,
        default=DEFAULT_L1_SMALL_SIZE,
        help="TT device small L1 allocator size in bytes.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local kokoro-v1_0.pth; if omitted, auto-detect local cache or download from HuggingFace.",
    )
    p.add_argument(
        "--torch-stft-fallback",
        action="store_true",
        help="Use PyTorch float32 STFT fallback in TT decoder for higher numerical parity.",
    )
    p.add_argument(
        "--torch-phase-fallback",
        action="store_true",
        help="Use PyTorch float32 SineGen phase fallback for higher numerical parity.",
    )
    p.add_argument(
        "--l1-activations",
        action="store_true",
        help=(
            "Keep the generator upsample/resblock loop activations L1-resident (~4%% faster, "
            "PCC-neutral). Safe for short utterances; may OOM on very long inputs."
        ),
    )
    p.add_argument(
        "--disable-complex",
        action="store_true",
        help=(
            "Use the istftnet disable_complex=True STFT formulation: reference KModel and the TT "
            "decoder both run the on-device CustomSTFT port (conv2d/conv_transpose2d, no fallback)."
        ),
    )
    p.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Metal-trace the decoder (asr/F0/N/s -> audio) — captured once per aligned length in a "
            "warmup pass and replayed in the measured loop. On by default; pass --no-trace to run the "
            "eager decoder. Reserves a DRAM trace region; forces the deterministic RNG path."
        ),
    )
    p.add_argument(
        "--trace-region-size",
        type=int,
        default=DEFAULT_TRACE_REGION_SIZE,
        help="DRAM trace region bytes when --trace is set.",
    )
    p.add_argument(
        "--pcc-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the reference HuggingFace KModel (CPU float32) on each chunk and report the "
            "PCC of the TT audio against it. On by default; pass --no-pcc-check to skip."
        ),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="torch RNG seed for the source-noise path.")
    return p


def main() -> None:
    args = create_parser().parse_args()

    results = run_demo(
        args.text,
        voice=args.voice,
        lang_code=args.lang_code,
        speed=args.speed,
        output_path=args.output,
        checkpoint=args.checkpoint,
        l1_small_size=args.l1_small_size,
        trace=args.trace,
        trace_region_size=args.trace_region_size,
        torch_stft_fallback=bool(args.torch_stft_fallback),
        torch_phase_fallback=bool(args.torch_phase_fallback),
        l1_activations=bool(args.l1_activations),
        disable_complex=bool(args.disable_complex),
        pcc_check=bool(args.pcc_check),
        seed=args.seed,
    )

    _print_performance_metrics(results)
    _print_model_params(results)


if __name__ == "__main__":
    main()
