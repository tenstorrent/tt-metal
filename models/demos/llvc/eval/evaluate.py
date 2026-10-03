# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LLVC quality / performance evaluation harness.

Produces the evidence a voice-conversion bring-up is expected to report:

1. Decoder throughput (latent frames per second) vs the real-time frame rate.
2. Token-level accuracy vs the PyTorch reference (per-frame correlation + SI-SDR,
   not just a single global PCC).
3. WER / content preservation (source vs converted, via Whisper). Bounty target
   is WER < 3.0% on the agreed eval set.
4. Speaker similarity vs the target speaker (requires ``--target-ref``).
5. Objective audio quality of the converted speech (DNSMOS) -- optional.

Two stages
----------
The device work and the file-based quality metrics are decoupled so the heavy
eval models (Whisper / resemblyzer / DNSMOS, which pull their own torch) never
share a process with ``ttnn``:

* ``--stage convert`` (needs ttnn + the checkpoint): converts the wavs, times the
  stream, computes throughput + token-level accuracy, and writes the converted
  wavs plus ``eval_report.json`` (with a manifest) into ``--out-dir``.
* ``--stage metrics`` (offline, no ttnn): reads that manifest and adds WER,
  speaker similarity, and DNSMOS on the saved wavs.
* ``--stage all`` (default): both in one process.

Metrics 3-5 are imported lazily; if a dependency is missing the harness logs a
skip line and still reports the rest.

Example
-------
# on the device box (clean tt-metal env):
python models/demos/llvc/eval/evaluate.py --stage convert \
    --config .../config.json --checkpoint .../G_500000.pth \
    --input .../test_wavs --out-dir llvc_eval_out --chunk-factor 2

# anywhere (after `pip install openai-whisper jiwer resemblyzer librosa onnxruntime requests`):
python models/demos/llvc/eval/evaluate.py --stage metrics \
    --out-dir llvc_eval_out --target-ref .../speaker_8312 --whisper-model small.en
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("llvc-eval")


# --------------------------------------------------------------------- audio io
# Inlined (rather than imported from demo.py) so this module stays ttnn-free:
# the `metrics` stage must import cleanly in a lightweight venv that only has the
# quality-eval deps (whisper / resemblyzer / torchmetrics), not the tt-metal stack.
def _glob_audio(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files: list[str] = []
    for ext in ("wav", "mp3", "flac"):
        files.extend(str(p) for p in Path(path).rglob(f"*.{ext}"))
    return sorted(files)


def _resample(audio: torch.Tensor, sr: int, target: int) -> torch.Tensor:
    if sr == target:
        return audio
    n = int(round(audio.shape[-1] * target / sr))
    a = audio.reshape(1, 1, -1)
    a = torch.nn.functional.interpolate(a, size=n, mode="linear", align_corners=False)
    return a.reshape(-1)


def _load_audio(path: str, sample_rate: int) -> torch.Tensor:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(data).mean(dim=1)
    return _resample(audio, sr, sample_rate)


def _save_audio(audio: torch.Tensor, path: str, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(path, audio.detach().cpu().reshape(-1).float().numpy(), sample_rate)


def _safe_output_path(out_dir: str, name: str) -> str:
    """Resolve ``name`` inside ``out_dir``, rejecting path-traversal in ``name``."""
    out_dir_abs = os.path.abspath(out_dir)
    dest = os.path.abspath(os.path.join(out_dir_abs, os.path.basename(name)))
    if os.path.commonpath([out_dir_abs, dest]) != out_dir_abs:
        raise ValueError(f"Unsafe output path derived from {name!r}")
    return dest


# --------------------------------------------------------------------- metrics
def per_frame_pcc(a: torch.Tensor, b: torch.Tensor, frame_len: int, *, energy_gate_db: float = -40.0):
    """Per-frame Pearson correlation over *voiced* frames (token-level accuracy).

    A single global PCC can hide localised divergence, so each decoder output hop
    (``frame_len = L`` samples) is correlated separately. Near-silent frames are
    excluded: their values are dominated by bf16 rounding noise, which decorrelates
    a 16-sample window even when the audible signal matches. A frame counts as
    voiced when its reference RMS is within ``energy_gate_db`` of the file's
    loudest frame. Returns ``(corr_over_voiced, voiced_count, total_count)``.
    """
    x = a.flatten().float()
    y = b.flatten().float()
    n = min(x.numel(), y.numel())
    n -= n % frame_len
    if n == 0:
        return torch.ones(1), 1, 1
    x = x[:n].reshape(-1, frame_len)
    y = y[:n].reshape(-1, frame_len)
    xv = x - x.mean(dim=1, keepdim=True)
    yv = y - y.mean(dim=1, keepdim=True)
    denom = xv.norm(dim=1) * yv.norm(dim=1)
    corr = ((xv * yv).sum(dim=1) / (denom + 1e-8)).clamp(-1.0, 1.0)

    ref_rms = y.pow(2).mean(dim=1).sqrt()
    thresh = ref_rms.max().clamp_min(1e-8) * (10.0 ** (energy_gate_db / 20.0))
    voiced = ref_rms >= thresh
    if int(voiced.sum()) == 0:
        voiced = torch.ones_like(voiced)
    return corr[voiced], int(voiced.sum()), int(voiced.numel())


def si_sdr_db(est: torch.Tensor, ref: torch.Tensor) -> float:
    """Scale-invariant signal-to-distortion ratio (dB) of TTNN output vs reference.

    A robust, scale-independent waveform-fidelity number that (unlike a 16-sample
    PCC) is not thrown off by silence or a constant gain difference.
    """
    e = est.flatten().float()
    r = ref.flatten().float()
    n = min(e.numel(), r.numel())
    e, r = e[:n] - e[:n].mean(), r[:n] - r[:n].mean()
    alpha = (e * r).sum() / r.pow(2).sum().clamp_min(1e-8)
    target = alpha * r
    noise = e - target
    ratio = target.pow(2).sum() / noise.pow(2).sum().clamp_min(1e-8)
    return float(10.0 * torch.log10(ratio.clamp_min(1e-8)))


def global_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().float()
    y = b.flatten().float()
    n = min(x.numel(), y.numel())
    x, y = x[:n], y[:n]
    xv = x - x.mean()
    yv = y - y.mean()
    denom = (xv.norm() * yv.norm()).item()
    if denom == 0.0:
        return 1.0
    return max(-1.0, min(1.0, (xv * yv).sum().item() / (denom + 1e-8)))


def best_lag(est: torch.Tensor, ref: torch.Tensor, max_lag: int = 4096) -> int:
    """Integer sample lag (via FFT cross-correlation) that best aligns ``est`` to ``ref``.

    Positive lag means ``est`` is delayed w.r.t. ``ref`` (align by dropping ``lag``
    leading samples of ``est``). Used to tell a genuine numerical divergence apart
    from a benign fixed latency offset: a time-shifted-but-correct waveform scores
    terribly on per-sample PCC / SI-SDR yet is perceptually fine (good WER).
    """
    e = est.flatten().float()
    r = ref.flatten().float()
    n = min(e.numel(), r.numel())
    if n == 0:
        return 0
    e = e[:n] - e[:n].mean()
    r = r[:n] - r[:n].mean()
    size = 1
    while size < 2 * n:
        size <<= 1
    cc = torch.fft.irfft(torch.fft.rfft(e, size) * torch.conj(torch.fft.rfft(r, size)), size)
    lags = torch.arange(size)
    true_lag = torch.where(lags <= size // 2, lags, lags - size)
    cc = cc.masked_fill(true_lag.abs() > min(max_lag, n - 1), float("-inf"))
    return int(true_lag[int(torch.argmax(cc))])


def aligned_scores(est: torch.Tensor, ref: torch.Tensor) -> tuple[int, float, float]:
    """``(lag, global_pcc, si_sdr_db)`` after removing the best integer sample lag."""
    e = est.flatten().float()
    r = ref.flatten().float()
    n = min(e.numel(), r.numel())
    e, r = e[:n], r[:n]
    lag = best_lag(e, r)
    if lag > 0:
        e, r = e[lag:], r[: n - lag]
    elif lag < 0:
        e, r = e[: n + lag], r[-lag:]
    return lag, global_pcc(e, r), si_sdr_db(e, r)


def decoder_throughput(rtf: float, sample_rate: int, hop: int) -> dict:
    """Latent-frame throughput derived from RTF.

    The encoder/decoder run on latent frames produced at one frame per ``hop=L``
    input samples, so real-time demands ``sample_rate / hop`` frames per second.
    Processing time per second of audio is ``RTF`` seconds, so the achieved
    throughput is ``(sample_rate / hop) / RTF``.
    """
    realtime_fps = sample_rate / hop
    achieved_fps = realtime_fps / max(rtf, 1e-9)
    return {
        "realtime_frames_per_s": realtime_fps,
        "achieved_frames_per_s": achieved_fps,
        "speed_vs_realtime": achieved_fps / realtime_fps,
    }


WER_TARGET = 0.03  # bounty Stage-1 content-preservation gate


def _normalize_transcript(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — ASR-noise, not content."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def try_wer(source_paths: list[str], converted_paths: list[str], *, whisper_model: str = "small.en") -> dict | None:
    """Word error rate between source and converted transcripts (content preservation)."""
    try:
        import jiwer
        import whisper
    except (ImportError, ModuleNotFoundError):
        logger.warning("WER skipped: install `openai-whisper` and `jiwer` to enable content-preservation eval.")
        return None

    asr = whisper.load_model(whisper_model)

    def transcribe(path: str) -> str:
        # Decode with soundfile and hand Whisper the raw 16 kHz mono waveform;
        # passing a path would make Whisper shell out to ffmpeg (not installed here).
        wav = _load_audio(path, 16000).numpy().astype("float32")
        return _normalize_transcript(asr.transcribe(wav, fp16=False, language="en")["text"])

    refs, hyps, pairs = [], [], []
    for src, conv in zip(source_paths, converted_paths):
        ref, hyp = transcribe(src), transcribe(conv)
        refs.append(ref)
        hyps.append(hyp)
        pairs.append({"source": os.path.basename(src), "ref": ref, "hyp": hyp})
    score = float(jiwer.wer(refs, hyps))
    return {
        "wer": score,
        "num_files": len(refs),
        "whisper_model": whisper_model,
        "target": WER_TARGET,
        "meets_target": score < WER_TARGET,
        "transcripts": pairs,
    }


def try_speaker_similarity(
    converted_paths: list[str],
    target_ref_paths: list[str],
    source_paths: list[str],
) -> dict | None:
    """Speaker-identity evidence vs the target speaker (``--target-ref`` required)."""
    try:
        import numpy as np
        from resemblyzer import VoiceEncoder, preprocess_wav
    except (ImportError, ModuleNotFoundError):
        logger.warning("Speaker similarity skipped: install `resemblyzer` to enable.")
        return None

    encoder = VoiceEncoder()

    def embed(path: str):
        e = encoder.embed_utterance(preprocess_wav(Path(path)))
        return e / (np.linalg.norm(e) + 1e-8)

    conv = [embed(p) for p in converted_paths]
    result: dict = {}

    if not target_ref_paths:
        raise ValueError(
            "--target-ref is required: score converted clips against the target speaker "
            "(LibriSpeech speaker 8312 / KoeAI training target), not converted↔converted consistency."
        )
    target = np.mean([embed(p) for p in target_ref_paths], axis=0)
    target /= np.linalg.norm(target) + 1e-8
    sims = [float(np.dot(e, target)) for e in conv]
    result["to_target_mean"] = float(np.mean(sims))
    result["to_target_min"] = float(np.min(sims))
    result["num_target_ref_files"] = len(target_ref_paths)

    src = [embed(p) for p in source_paths]
    cross = [float(np.dot(c, s)) for c, s in zip(conv, src)]
    result["converted_vs_source_mean"] = float(np.mean(cross))
    return result


def try_dnsmos(converted_paths: list[str], sample_rate: int) -> dict | None:
    """Non-intrusive DNSMOS audio-quality score of the converted speech."""
    try:
        # torchmetrics may be installed while its DNSMOS sub-deps (librosa /
        # onnxruntime / requests) are not; that surfaces at *construction*, not
        # import, so both must be inside the guard.
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

        metric = DeepNoiseSuppressionMeanOpinionScore(fs=sample_rate, personalized=False)
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning("DNSMOS skipped (%s): install torchmetrics[audio] + onnxruntime + librosa + requests.", exc)
        return None

    import soundfile as sf

    ovrl = []
    for conv in converted_paths:
        data, _ = sf.read(conv, dtype="float32")
        score = metric(torch.from_numpy(data).reshape(-1))
        ovrl.append(float(score[-1]))  # [p808, sig, bak, ovrl]
    return {"dnsmos_ovrl_mean": sum(ovrl) / len(ovrl), "num_files": len(ovrl)}


# ------------------------------------------------------------------- stages
def reference_stream(reference, audio: torch.Tensor, *, L: int, dec_chunk_size: int, chunk_factor: int) -> torch.Tensor:
    """Run the PyTorch reference with the same chunking as ``LLVCModel.stream``.

    Offline ``reference(wav)`` and streaming inference use different padding /
    lookahead alignment; comparing TTNN stream against offline reference can
    look like a correctness bug when it is only an alignment mismatch. Scoring
    both sides with identical chunking isolates real numerical divergence.
    """
    chunk_len = dec_chunk_size * L * chunk_factor
    waveform = audio.reshape(-1)
    original_len = int(waveform.shape[0])
    if original_len % chunk_len != 0:
        waveform = torch.nn.functional.pad(waveform, (0, chunk_len - (original_len % chunk_len)))
    waveform = torch.cat((waveform[L:], torch.zeros(L)))
    chunks = list(torch.split(waveform, chunk_len))

    enc_buf, dec_buf, out_buf = reference.init_buffers(1, waveform.device)
    convnet_ctx = None
    if hasattr(reference, "convnet_pre"):
        convnet_ctx = reference.convnet_pre.init_ctx_buf(1, waveform.device)

    outputs = []
    with torch.no_grad():
        for i, c in enumerate(chunks):
            front = torch.zeros(L * 2) if i == 0 else chunks[i - 1][-L * 2 :]
            chunk = torch.cat([front, c]).reshape(1, 1, -1)
            out, enc_buf, dec_buf, out_buf, convnet_ctx = reference(
                chunk, enc_buf, dec_buf, out_buf, convnet_ctx, pad=False
            )
            outputs.append(out)
    return torch.cat(outputs, dim=-1)[:, :, :original_len]


def convert_stage(args: argparse.Namespace) -> dict:
    """Device stage: convert wavs, time streaming, score token-level accuracy."""
    import ttnn
    from models.demos.llvc.tt.config import LLVC_L1_SMALL_SIZE, LLVC_TRACE_REGION_SIZE
    from models.demos.llvc.tt.model import create_llvc
    from models.demos.llvc.tt.state_io import load_llvc_config_and_model

    if not (args.config and args.checkpoint):
        raise ValueError("--stage convert needs --config and --checkpoint")

    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=LLVC_L1_SMALL_SIZE, trace_region_size=LLVC_TRACE_REGION_SIZE
    )
    device.enable_program_cache()
    try:
        config, reference = load_llvc_config_and_model(args.config, args.checkpoint)
        model = create_llvc(config, device=device, reference=reference)
        sr = config.sample_rate
        hop = config.L

        files = _glob_audio(args.input)
        if not files:
            raise FileNotFoundError(f"No audio files found under {args.input}")
        if args.limit:
            files = files[: args.limit]

        os.makedirs(args.out_dir, exist_ok=True)
        source_paths, converted_paths = [], []
        rtfs, latencies, si_sdrs, global_pccs, frame_corrs = [], [], [], [], []
        aligned_pccs, aligned_si_sdrs, lags = [], [], []
        voiced_total, frame_total = 0, 0

        for fname in files:
            audio = _load_audio(fname, sr)

            # Score the shipped streaming path against the reference run with the
            # same chunking (apples-to-apples). Offline mega-shot on TTNN is
            # size-sensitive; model() now uses the chunked graph as well.
            ref_out = reference_stream(
                reference, audio, L=config.L, dec_chunk_size=config.dec_chunk_size, chunk_factor=args.chunk_factor
            )
            stream_out, metrics = model.stream(audio, chunk_factor=args.chunk_factor)
            tt_out = stream_out
            rtf, latency = metrics.rtf, metrics.latency_ms

            corr, n_voiced, n_frames = per_frame_pcc(tt_out, ref_out, hop)
            frame_corrs.append(corr)
            voiced_total += n_voiced
            frame_total += n_frames
            global_pccs.append(global_pcc(tt_out, ref_out))
            si_sdrs.append(si_sdr_db(tt_out, ref_out))
            lag, apcc, asdr = aligned_scores(tt_out, ref_out)
            lags.append(lag)
            aligned_pccs.append(apcc)
            aligned_si_sdrs.append(asdr)

            rtfs.append(rtf)
            latencies.append(latency)
            out_path = _safe_output_path(args.out_dir, os.path.basename(fname))
            _save_audio(stream_out.squeeze(0), out_path, sr)
            source_paths.append(os.path.abspath(fname))
            converted_paths.append(os.path.abspath(out_path))
            logger.info(
                "[%s] e2e_RTF=%.3f e2e_latency=%.2fms device_RTF=%.3f raw_pcc=%.4f lag=%d"
                " aligned_pcc=%.4f aligned_si_sdr=%.1fdB",
                os.path.basename(fname),
                rtf,
                latency,
                metrics.device_rtf,
                global_pccs[-1],
                lag,
                apcc,
                asdr,
            )
    finally:
        ttnn.close_device(device)

    mean_rtf = sum(rtfs) / len(rtfs)
    all_corr = torch.cat(frame_corrs)
    return {
        "num_files": len(files),
        "chunk_factor": args.chunk_factor,
        "streaming": {
            "mean_rtf": mean_rtf,
            "mean_chunk_latency_ms": sum(latencies) / len(latencies),
        },
        "decoder_throughput": decoder_throughput(mean_rtf, sr, hop),
        "token_level_accuracy_vs_reference": {
            "global_pcc_mean": sum(global_pccs) / len(global_pccs),
            "si_sdr_db_mean": sum(si_sdrs) / len(si_sdrs),
            "voiced_frame_pcc_mean": all_corr.mean().item(),
            "voiced_frame_pcc_min": all_corr.min().item(),
            "frac_voiced_frames_pcc_above_0.9": (all_corr > 0.9).float().mean().item(),
            "voiced_frames": voiced_total,
            "total_frames": frame_total,
            # Lag-corrected: if these are high while the raw numbers above are low,
            # the divergence is a per-file latency offset, not a correctness bug.
            "aligned_global_pcc_mean": sum(aligned_pccs) / len(aligned_pccs),
            "aligned_si_sdr_db_mean": sum(aligned_si_sdrs) / len(aligned_si_sdrs),
            "per_file_lag_samples": {os.path.basename(f): lag for f, lag in zip(files, lags)},
            "max_abs_lag_samples": max(abs(x) for x in lags),
        },
        "_manifest": {"sample_rate": sr, "sources": source_paths, "converted": converted_paths},
    }


def metrics_stage(args: argparse.Namespace, report: dict | None) -> dict:
    """Offline stage: WER / speaker similarity / DNSMOS on the saved wavs (no ttnn)."""
    if report is None:
        json_path = os.path.join(args.out_dir, "eval_report.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found — run `--stage convert` first.")
        report = json.loads(Path(json_path).read_text())

    manifest = report.get("_manifest")
    if not manifest:
        raise RuntimeError("eval_report.json has no manifest; re-run `--stage convert`.")
    sr = manifest["sample_rate"]
    sources = manifest["sources"]
    converted = manifest["converted"]

    if not args.target_ref:
        raise ValueError(
            "--target-ref is required for speaker similarity (bounty: cosine vs the target speaker). "
            "Pass LibriSpeech speaker-8312 wavs, e.g. --target-ref LibriSpeech/train-clean-100/8312"
        )
    wer = try_wer(sources, converted, whisper_model=args.whisper_model)
    if wer is not None:
        report["content_preservation_wer"] = wer
    spk = try_speaker_similarity(converted, args.target_ref, sources)
    if spk is not None:
        report["speaker_similarity"] = spk
    quality = try_dnsmos(converted, sr)
    if quality is not None:
        report["audio_quality_dnsmos"] = quality
    return report


# ------------------------------------------------------------------- report io
def _write_markdown(report: dict, path: str) -> None:
    thr = report["decoder_throughput"]
    tok = report["token_level_accuracy_vs_reference"]
    st = report["streaming"]
    lines = [
        "### LLVC evaluation report",
        "",
        f"Files: {report['num_files']} · chunk_factor: {report['chunk_factor']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Streaming mean RTF | {st['mean_rtf']:.3f} |",
        f"| Mean chunk latency | {st['mean_chunk_latency_ms']:.2f} ms |",
        f"| Decoder throughput | {thr['achieved_frames_per_s']:.0f} frames/s"
        f" ({thr['speed_vs_realtime']:.2f}× real-time) |",
        f"| Real-time frame rate needed | {thr['realtime_frames_per_s']:.0f} frames/s |",
        f"| Token-level accuracy (global PCC, raw) | {tok['global_pcc_mean']:.4f} |",
        f"| Token-level accuracy (SI-SDR, raw) | {tok['si_sdr_db_mean']:.1f} dB |",
        f"| Voiced-frame PCC (mean) | {tok['voiced_frame_pcc_mean']:.4f} |",
        f"| Voiced-frame PCC (min) | {tok['voiced_frame_pcc_min']:.4f} |",
        f"| Voiced frames with PCC > 0.9 | {tok['frac_voiced_frames_pcc_above_0.9'] * 100:.1f}% |",
        f"| Voiced / total frames | {tok['voiced_frames']} / {tok['total_frames']} |",
    ]
    if "aligned_global_pcc_mean" in tok:
        lines += [
            f"| Token-level accuracy (global PCC, lag-aligned) | {tok['aligned_global_pcc_mean']:.4f} |",
            f"| Token-level accuracy (SI-SDR, lag-aligned) | {tok['aligned_si_sdr_db_mean']:.1f} dB |",
            f"| Max per-file lag | {tok['max_abs_lag_samples']} samples |",
        ]
    if "content_preservation_wer" in report:
        lines.append(
            f"| Content preservation (WER, {report['content_preservation_wer'].get('whisper_model', 'whisper')}) |"
            f" {report['content_preservation_wer']['wer'] * 100:.2f}%"
            f" (target < {report['content_preservation_wer'].get('target', 0.03) * 100:.1f}%) |"
        )
    if "speaker_similarity" in report:
        spk = report["speaker_similarity"]
        if "to_target_mean" in spk:
            lines.append(f"| Speaker similarity to target (cosine) | {spk['to_target_mean']:.3f} |")
        if "target_consistency_mean" in spk:
            lines.append(f"| Target consistency (converted↔converted) | {spk['target_consistency_mean']:.3f} |")
        lines.append(
            f"| Converted↔source similarity (lower = identity changed) | {spk['converted_vs_source_mean']:.3f} |"
        )
    if "audio_quality_dnsmos" in report:
        lines.append(f"| Audio quality (DNSMOS OVRL) | {report['audio_quality_dnsmos']['dnsmos_ovrl_mean']:.2f} / 5 |")
    Path(path).write_text("\n".join(lines) + "\n")


def _write_report(report: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "eval_report.json")
    md_path = os.path.join(out_dir, "eval_report.md")
    Path(json_path).write_text(json.dumps(report, indent=2))
    _write_markdown(report, md_path)
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLVC evaluation harness")
    p.add_argument("--stage", choices=["all", "convert", "metrics"], default="all")
    p.add_argument("--config", type=str, default=None, help="KoeAI config.json path (convert stage)")
    p.add_argument("--checkpoint", type=str, default=None, help="KoeAI checkpoint .pth (convert stage)")
    p.add_argument("--input", type=str, default="test_wavs", help="Source audio file or directory")
    p.add_argument("--out-dir", type=str, default="llvc_eval_out", help="Converted wavs + report location")
    p.add_argument(
        "--target-ref",
        type=str,
        nargs="*",
        help="Target-speaker wav(s)/dir (required for metrics; LibriSpeech speaker 8312)",
    )
    p.add_argument(
        "--whisper-model", type=str, default="small.en", help="Whisper checkpoint for WER (default small.en)"
    )
    p.add_argument("--chunk-factor", type=int, default=2, help="Streaming chunk multiplier (2 meets RTF<0.3)")
    p.add_argument("--limit", type=int, default=0, help="Only evaluate the first N files (0 = all)")
    p.add_argument("--device-id", type=int, default=0)
    args = p.parse_args()
    if args.target_ref:
        expanded: list[str] = []
        for t in args.target_ref:
            expanded.extend(_glob_audio(t))
        args.target_ref = expanded
    return args


def main() -> None:
    args = parse_args()
    report: dict | None = None
    if args.stage in ("all", "convert"):
        report = convert_stage(args)
        _write_report(report, args.out_dir)
    if args.stage in ("all", "metrics"):
        report = metrics_stage(args, report)
        _write_report(report, args.out_dir)

    json_path, md_path = os.path.join(args.out_dir, "eval_report.json"), os.path.join(args.out_dir, "eval_report.md")
    logger.info("Wrote %s and %s", json_path, md_path)
    print(json.dumps({k: v for k, v in report.items() if k != "_manifest"}, indent=2))


if __name__ == "__main__":
    main()
