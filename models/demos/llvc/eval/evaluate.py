# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LLVC quality / performance evaluation harness.

Produces the evidence a voice-conversion bring-up is expected to report:

1. Decoder throughput (latent frames per second) vs the real-time frame rate.
2. Token-level accuracy vs the PyTorch reference (per-frame correlation, not just
   a single global PCC).
3. WER / content preservation (source vs converted, via Whisper) -- optional.
4. Speaker similarity to the target speaker (via a speaker encoder) -- optional.
5. Objective audio quality of the converted speech (DNSMOS) -- optional.

Metrics 1 and 2 are pure-torch and always run. Metrics 3-5 need extra models
(``openai-whisper`` + ``jiwer``, ``resemblyzer``, ``torchmetrics[audio]`` with
``onnxruntime``); each is imported lazily and skipped with a clear message if the
dependency is absent, so the harness stays turnkey without them.

Example
-------
python models/demos/llvc/eval/evaluate.py \
    --config /path/experiments/llvc/config.json \
    --checkpoint /path/G_500000.pth \
    --input /path/test_wavs \
    --target-ref /path/target_speaker_samples \
    --out-dir llvc_eval_out --chunk-factor 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.llvc.demo.demo import _glob_audio, _load_audio, _safe_output_path, _save_audio
from models.demos.llvc.tt.model import create_llvc
from models.demos.llvc.tt.state_io import load_llvc_config_and_model


# --------------------------------------------------------------------- metrics
def per_frame_pcc(a: torch.Tensor, b: torch.Tensor, frame_len: int) -> torch.Tensor:
    """Per-frame Pearson correlation between two waveforms, chunked into ``frame_len``.

    A single global PCC can hide localised divergence; correlating each output
    frame (one hop of ``L`` samples) gives a token-level accuracy distribution.
    """
    x = a.flatten().float()
    y = b.flatten().float()
    n = min(x.numel(), y.numel())
    n -= n % frame_len
    if n == 0:
        return torch.ones(1)
    x = x[:n].reshape(-1, frame_len)
    y = y[:n].reshape(-1, frame_len)
    xv = x - x.mean(dim=1, keepdim=True)
    yv = y - y.mean(dim=1, keepdim=True)
    denom = xv.norm(dim=1) * yv.norm(dim=1)
    corr = (xv * yv).sum(dim=1) / (denom + 1e-8)
    # Frames where both sides are silent (denom==0) are a perfect match.
    corr = torch.where(denom == 0, torch.ones_like(corr), corr)
    return corr.clamp(-1.0, 1.0)


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
    except ImportError:
        logger.warning("WER skipped: install `openai-whisper` and `jiwer` to enable content-preservation eval.")
        return None

    asr = whisper.load_model("base.en")

    def transcribe(path: str) -> str:
        return asr.transcribe(path, fp16=False)["text"].strip().lower()

    refs, hyps = [], []
    for src, conv in zip(source_paths, converted_paths):
        refs.append(transcribe(src))
        hyps.append(transcribe(conv))
    wer = jiwer.wer(refs, hyps)
    return {"wer": wer, "num_files": len(refs)}


def try_speaker_similarity(converted_paths: list[str], target_ref_paths: list[str]) -> dict | None:
    """Cosine similarity of converted-speech speaker embeddings to the target speaker."""
    if not target_ref_paths:
        logger.warning("Speaker similarity skipped: pass --target-ref with target-speaker wavs to enable.")
        return None
    try:
        import numpy as np
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        logger.warning("Speaker similarity skipped: install `resemblyzer` to enable.")
        return None

    encoder = VoiceEncoder()
    target_embed = np.mean([encoder.embed_utterance(preprocess_wav(Path(p))) for p in target_ref_paths], axis=0)
    target_embed /= np.linalg.norm(target_embed) + 1e-8

    sims = []
    for conv in converted_paths:
        emb = encoder.embed_utterance(preprocess_wav(Path(conv)))
        emb /= np.linalg.norm(emb) + 1e-8
        sims.append(float(np.dot(emb, target_embed)))
    return {"speaker_similarity_mean": float(np.mean(sims)), "speaker_similarity_min": float(np.min(sims))}


def try_dnsmos(converted_paths: list[str], sample_rate: int) -> dict | None:
    """Non-intrusive DNSMOS audio-quality score of the converted speech."""
    try:
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
    except ImportError:
        logger.warning("DNSMOS skipped: install `torchmetrics[audio]` + `onnxruntime` + `librosa` to enable.")
        return None

    metric = DeepNoiseSuppressionMeanOpinionScore(fs=sample_rate, personalized=False)
    import soundfile as sf

    ovrl = []
    for conv in converted_paths:
        data, _ = sf.read(conv, dtype="float32")
        score = metric(torch.from_numpy(data).reshape(-1))
        ovrl.append(float(score[-1]))  # [p808, sig, bak, ovrl]
    return {"dnsmos_ovrl_mean": sum(ovrl) / len(ovrl), "num_files": len(ovrl)}


# ------------------------------------------------------------------------ main
def run_eval(args: argparse.Namespace, *, device: ttnn.Device) -> dict:
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
    rtfs, latencies = [], []
    frame_corrs = []
    global_pccs = []

    for fname in files:
        audio = _load_audio(fname, sr)
        wav = audio.reshape(1, 1, -1)

        # token-level accuracy: non-streaming TTNN vs the PyTorch reference
        with torch.no_grad():
            ref_out = reference(wav)
        tt_ns = model(wav)
        corr = per_frame_pcc(tt_ns, ref_out, hop)
        frame_corrs.append(corr)
        global_pccs.append(global_pcc(tt_ns, ref_out))

        # streaming conversion: RTF/latency + saved audio for the quality evals
        stream_out, rtf, latency = model.stream(audio, chunk_factor=args.chunk_factor)
        rtfs.append(rtf)
        latencies.append(latency)
        out_path = _safe_output_path(args.out_dir, os.path.basename(fname))
        _save_audio(stream_out.squeeze(0), out_path, sr)
        source_paths.append(fname)
        converted_paths.append(out_path)
        logger.info(
            "[{}] RTF={:.3f} latency={:.2f}ms per_frame_pcc_mean={:.4f}",
            os.path.basename(fname),
            rtf,
            latency,
            corr.mean().item(),
        )

    mean_rtf = sum(rtfs) / len(rtfs)
    all_corr = torch.cat(frame_corrs)
    report = {
        "num_files": len(files),
        "chunk_factor": args.chunk_factor,
        "streaming": {
            "mean_rtf": mean_rtf,
            "mean_chunk_latency_ms": sum(latencies) / len(latencies),
        },
        "decoder_throughput": decoder_throughput(mean_rtf, sr, hop),
        "token_level_accuracy_vs_reference": {
            "global_pcc_mean": sum(global_pccs) / len(global_pccs),
            "per_frame_pcc_mean": all_corr.mean().item(),
            "per_frame_pcc_min": all_corr.min().item(),
            "frac_frames_pcc_above_0.9": (all_corr > 0.9).float().mean().item(),
            "max_abs_frame_pcc_gap": (1.0 - all_corr.min().item()),
        },
    }

    wer = try_wer(source_paths, converted_paths)
    if wer is not None:
        report["content_preservation_wer"] = wer
    spk = try_speaker_similarity(converted_paths, args.target_ref or [])
    if spk is not None:
        report["speaker_similarity"] = spk
    quality = try_dnsmos(converted_paths, sr)
    if quality is not None:
        report["audio_quality_dnsmos"] = quality

    return report


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
        f"| Token-level accuracy (per-frame PCC mean) | {tok['per_frame_pcc_mean']:.4f} |",
        f"| Per-frame PCC min | {tok['per_frame_pcc_min']:.4f} |",
        f"| Frames with PCC > 0.9 | {tok['frac_frames_pcc_above_0.9'] * 100:.1f}% |",
    ]
    if "content_preservation_wer" in report:
        lines.append(
            f"| Content preservation (WER, source vs converted) | {report['content_preservation_wer']['wer'] * 100:.1f}% |"
        )
    if "speaker_similarity" in report:
        lines.append(
            f"| Speaker similarity to target (cosine) | {report['speaker_similarity']['speaker_similarity_mean']:.3f} |"
        )
    if "audio_quality_dnsmos" in report:
        lines.append(f"| Audio quality (DNSMOS OVRL) | {report['audio_quality_dnsmos']['dnsmos_ovrl_mean']:.2f} / 5 |")
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLVC evaluation harness")
    p.add_argument("--config", type=str, required=True, help="KoeAI config.json path")
    p.add_argument("--checkpoint", type=str, required=True, help="KoeAI generator checkpoint (.pth)")
    p.add_argument("--input", type=str, required=True, help="Source audio file or directory")
    p.add_argument("--out-dir", type=str, default="llvc_eval_out", help="Where converted wavs + report are written")
    p.add_argument(
        "--target-ref", type=str, nargs="*", help="Target-speaker reference wav(s)/dir for speaker similarity"
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
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768, trace_region_size=23887872)
    device.enable_program_cache()
    try:
        report = run_eval(args, device=device)
    finally:
        ttnn.close_device(device)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "eval_report.json")
    md_path = os.path.join(args.out_dir, "eval_report.md")
    Path(json_path).write_text(json.dumps(report, indent=2))
    _write_markdown(report, md_path)
    logger.info("Wrote {} and {}", json_path, md_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
