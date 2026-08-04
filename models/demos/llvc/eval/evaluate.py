# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LLVC quality / performance evaluation harness.

Produces the evidence a voice-conversion bring-up is expected to report:

1. Decoder throughput (latent frames per second) vs the real-time frame rate.
2. Token-level accuracy vs the PyTorch reference (per-frame correlation + SI-SDR,
   not just a single global PCC).
3. WER / content preservation (source vs converted, via Whisper) -- optional.
4. Speaker similarity (to a target speaker if given, else target-consistency of
   the converted set + contrast vs the source) -- optional.
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
    --input .../test_wavs --out-dir llvc_eval_out --target-ref .../target_speaker
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.llvc.demo.demo import _glob_audio, _load_audio, _safe_output_path, _save_audio


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


def try_wer(source_paths: list[str], converted_paths: list[str]) -> dict | None:
    """Word error rate between source and converted transcripts (content preservation)."""
    try:
        import jiwer
        import whisper
    except (ImportError, ModuleNotFoundError):
        logger.warning("WER skipped: install `openai-whisper` and `jiwer` to enable content-preservation eval.")
        return None

    asr = whisper.load_model("base.en")

    def transcribe(path: str) -> str:
        return asr.transcribe(path, fp16=False)["text"].strip().lower()

    refs, hyps = [], []
    for src, conv in zip(source_paths, converted_paths):
        refs.append(transcribe(src))
        hyps.append(transcribe(conv))
    return {"wer": float(jiwer.wer(refs, hyps)), "num_files": len(refs)}


def try_speaker_similarity(
    converted_paths: list[str],
    target_ref_paths: list[str],
    source_paths: list[str],
) -> dict | None:
    """Speaker-identity evidence via a speaker encoder.

    With ``--target-ref`` (target-speaker audio): cosine similarity of each
    converted clip to the mean target embedding. Without it (LLVC ships no target
    ground truth): report target-*consistency* — the mean pairwise similarity of
    the converted set, which should be high because every clip must map to the
    same target voice — and the converted-vs-source similarity, which should be
    lower, showing the identity actually changed.
    """
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

    if target_ref_paths:
        target = np.mean([embed(p) for p in target_ref_paths], axis=0)
        target /= np.linalg.norm(target) + 1e-8
        sims = [float(np.dot(e, target)) for e in conv]
        result["to_target_mean"] = float(np.mean(sims))
        result["to_target_min"] = float(np.min(sims))
    else:
        logger.warning(
            "No --target-ref: reporting target-consistency (pairwise similarity of converted clips) "
            "instead of similarity to a ground-truth target speaker."
        )
        pairs = [float(np.dot(a, b)) for a, b in itertools.combinations(conv, 2)]
        result["target_consistency_mean"] = float(np.mean(pairs)) if pairs else 1.0

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
        logger.warning("DNSMOS skipped ({}): install torchmetrics[audio] + onnxruntime + librosa + requests.", exc)
        return None

    import soundfile as sf

    ovrl = []
    for conv in converted_paths:
        data, _ = sf.read(conv, dtype="float32")
        score = metric(torch.from_numpy(data).reshape(-1))
        ovrl.append(float(score[-1]))  # [p808, sig, bak, ovrl]
    return {"dnsmos_ovrl_mean": sum(ovrl) / len(ovrl), "num_files": len(ovrl)}


# ------------------------------------------------------------------- stages
def convert_stage(args: argparse.Namespace) -> dict:
    """Device stage: convert wavs, time streaming, score token-level accuracy."""
    import ttnn
    from models.demos.llvc.tt.model import create_llvc
    from models.demos.llvc.tt.state_io import load_llvc_config_and_model

    if not (args.config and args.checkpoint):
        raise ValueError("--stage convert needs --config and --checkpoint")

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768, trace_region_size=23887872)
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
        voiced_total, frame_total = 0, 0

        for fname in files:
            audio = _load_audio(fname, sr)
            wav = audio.reshape(1, 1, -1)

            with torch.no_grad():
                ref_out = reference(wav)
            tt_ns = model(wav)
            corr, n_voiced, n_frames = per_frame_pcc(tt_ns, ref_out, hop)
            frame_corrs.append(corr)
            voiced_total += n_voiced
            frame_total += n_frames
            global_pccs.append(global_pcc(tt_ns, ref_out))
            si_sdrs.append(si_sdr_db(tt_ns, ref_out))

            stream_out, rtf, latency = model.stream(audio, chunk_factor=args.chunk_factor)
            rtfs.append(rtf)
            latencies.append(latency)
            out_path = _safe_output_path(args.out_dir, os.path.basename(fname))
            _save_audio(stream_out.squeeze(0), out_path, sr)
            source_paths.append(os.path.abspath(fname))
            converted_paths.append(os.path.abspath(out_path))
            logger.info(
                "[{}] RTF={:.3f} latency={:.2f}ms voiced_frame_pcc={:.4f} si_sdr={:.1f}dB",
                os.path.basename(fname),
                rtf,
                latency,
                corr.mean().item(),
                si_sdrs[-1],
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

    wer = try_wer(sources, converted)
    if wer is not None:
        report["content_preservation_wer"] = wer
    spk = try_speaker_similarity(converted, args.target_ref or [], sources)
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
        f"| Decoder throughput | {thr['achieved_frames_per_s']:.0f} frames/s ({thr['speed_vs_realtime']:.2f}× real-time) |",
        f"| Real-time frame rate needed | {thr['realtime_frames_per_s']:.0f} frames/s |",
        f"| Token-level accuracy (global PCC) | {tok['global_pcc_mean']:.4f} |",
        f"| Token-level accuracy (SI-SDR) | {tok['si_sdr_db_mean']:.1f} dB |",
        f"| Voiced-frame PCC (mean) | {tok['voiced_frame_pcc_mean']:.4f} |",
        f"| Voiced-frame PCC (min) | {tok['voiced_frame_pcc_min']:.4f} |",
        f"| Voiced frames with PCC > 0.9 | {tok['frac_voiced_frames_pcc_above_0.9'] * 100:.1f}% |",
        f"| Voiced / total frames | {tok['voiced_frames']} / {tok['total_frames']} |",
    ]
    if "content_preservation_wer" in report:
        lines.append(f"| Content preservation (WER, source vs converted) | {report['content_preservation_wer']['wer'] * 100:.1f}% |")
    if "speaker_similarity" in report:
        spk = report["speaker_similarity"]
        if "to_target_mean" in spk:
            lines.append(f"| Speaker similarity to target (cosine) | {spk['to_target_mean']:.3f} |")
        if "target_consistency_mean" in spk:
            lines.append(f"| Target consistency (converted↔converted) | {spk['target_consistency_mean']:.3f} |")
        lines.append(f"| Converted↔source similarity (lower = identity changed) | {spk['converted_vs_source_mean']:.3f} |")
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
    p.add_argument("--target-ref", type=str, nargs="*", help="Target-speaker wav(s)/dir for speaker similarity")
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
    logger.info("Wrote {} and {}", json_path, md_path)
    print(json.dumps({k: v for k, v in report.items() if k != "_manifest"}, indent=2))


if __name__ == "__main__":
    main()
