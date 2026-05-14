# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS demo — **fully on TT, zero reference model**.

All neural-network forward passes (text, acoustic, audio tokenizer decode) run on the
Tenstorrent device via ``VoxtralTTSPipeline.generate()``.  The only CPU work is:

- Mistral-common **tokenization** (equivalent to ``AutoTokenizer.encode()``)
- Voice-embedding file load (a single ``torch.load`` of a small ``.pt`` file)
- MM codebook embedding lookup + sum (tiny pure-PyTorch op, not a Transformer layer)

Modes
-----
``text`` (default)
    ``text`` + ``voice`` in JSON → ``pipe.generate()`` → ``.wav``

``codes``
    Pre-computed ``[1,37,T]`` codes tensor → ``pipe.decode_waveform_from_codes_tt()``

``latents``
    Pre-computed ``[1,1,T,C]`` latent tensor → TT mel decode + pretransform

Run (from tt-metal repo root)::

    export VOXTRAL_TTS_MODEL=mistralai/Voxtral-4B-TTS-2603
    ./python_env/bin/python models/experimental/voxtraltts/demo/demo.py \\
        --prompts models/experimental/voxtraltts/demo/data/sample_prompts.json \\
        --output-dir /tmp/voxtraltts_out
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from loguru import logger
from scipy.io import wavfile

import ttnn

from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL, load_voxtral_config
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline


# ---------------------------------------------------------------------------
# A. Argument groups  (Llama DemoArgs pattern: model / tt / data)
# ---------------------------------------------------------------------------


@dataclass
class ModelArgs:
    model_name_or_path: str = DEFAULT_VOXTRAL_MODEL


@dataclass
class TTArgs:
    text_max_seq_len: int = 4096
    text_dtype: str = "bfloat16"
    acoustic_dtype: str = "bfloat16"
    tokenizer_dtype: str = "bfloat16"


@dataclass
class DataArgs:
    prompts_file: str = "models/experimental/voxtraltts/demo/data/sample_prompts.json"
    output_dir: str = "generated/voxtraltts_demo"
    mode: str = "text"
    max_speech_tokens: int = 250
    seed: int = 0
    default_voice: str = "casual_male"
    warmup_iters: int = 0


@dataclass
class DemoArgs:
    model: ModelArgs
    tt: TTArgs
    data: DataArgs


def _ttnn_dtype(name: str) -> ttnn.DataType:
    return getattr(ttnn, name)


def _parse_demo_args(argv: list[str] | None = None) -> DemoArgs:
    p = argparse.ArgumentParser(description="Voxtral TTS fully-TT demo.")
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VOXTRAL_TTS_MODEL") or os.environ.get("HF_MODEL") or DEFAULT_VOXTRAL_MODEL,
    )
    p.add_argument("--prompts", type=str, default=DataArgs.prompts_file)
    p.add_argument("--output-dir", type=str, default=DataArgs.output_dir)
    p.add_argument("--mode", type=str, choices=("text", "codes", "latents"), default="text")
    p.add_argument("--text-max-seq-len", type=int, default=4096)
    p.add_argument("--max-speech-tokens", type=int, default=DataArgs.max_speech_tokens)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-iters", type=int, default=0)
    p.add_argument("--default-voice", type=str, default="casual_male")
    ns = p.parse_args(argv)
    return DemoArgs(
        model=ModelArgs(model_name_or_path=ns.model),
        tt=TTArgs(text_max_seq_len=ns.text_max_seq_len),
        data=DataArgs(
            prompts_file=ns.prompts,
            output_dir=ns.output_dir,
            mode=ns.mode,
            max_speech_tokens=ns.max_speech_tokens,
            seed=ns.seed,
            default_voice=ns.default_voice,
            warmup_iters=ns.warmup_iters,
        ),
    )


# ---------------------------------------------------------------------------
# B. Device + pipeline init
# ---------------------------------------------------------------------------


def _open_device():
    from tests.scripts.common import get_updated_device_params

    device_id = 0
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.TG:
        device_id = 4
    updated = get_updated_device_params({})
    original = ttnn.GetDefaultDevice()
    mesh = ttnn.CreateDevice(device_id=device_id, **updated)
    ttnn.SetDefaultDevice(mesh)
    return mesh, original


def _load_pipeline(mesh: ttnn.Device, args: DemoArgs) -> VoxtralTTSPipeline:
    return VoxtralTTSPipeline.from_model_name(
        mesh,
        model_name_or_path=args.model.model_name_or_path,
        text_max_seq_len=args.tt.text_max_seq_len,
        text_dtype=_ttnn_dtype(args.tt.text_dtype),
        acoustic_dtype=_ttnn_dtype(args.tt.acoustic_dtype),
        tokenizer_dtype=_ttnn_dtype(args.tt.tokenizer_dtype),
    )


# ---------------------------------------------------------------------------
# C. JSON prompt loading
# ---------------------------------------------------------------------------


def load_prompt_items(path: str, default_voice: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected non-empty JSON list in {path}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if isinstance(row, str):
            out.append({"id": i, "text": row, "voice": default_voice})
        elif isinstance(row, dict):
            row = dict(row)
            row.setdefault("id", i)
            row.setdefault("voice", default_voice)
            out.append(row)
        else:
            raise ValueError(f"Bad prompt entry {i}: {type(row)}")
    return out


# ---------------------------------------------------------------------------
# D + E. Inference helpers with TTFT / throughput logging
# ---------------------------------------------------------------------------


def _save_wav(path: Path, waveform_f32: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = waveform_f32.detach().float().cpu().numpy().reshape(-1)
    w = np.clip(w * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, w)


def _log_perf(label: str, *, ttft_s: float, n_frames: int, total_s: float) -> None:
    thr = n_frames / total_s if total_s > 0 else 0.0
    logger.info(f"[{label}] TTFT: {ttft_s * 1000:.2f} ms")
    logger.info(f"[{label}] Throughput: {thr:.2f} frames/s  ({n_frames} frames in {total_s:.3f} s)")


def run_text_mode(
    pipe: VoxtralTTSPipeline,
    text: str,
    voice: str,
    max_tokens: int,
    seed: int,
    sample_rate: int,
    out_path: Path,
) -> None:
    """Full TT TTS: text → acoustic codes → waveform. Logs TTFT and throughput."""
    t0 = perf_counter()
    out = pipe.generate_with_codes(text=text, voice=voice, max_tokens=max_tokens, seed=seed)
    wav = out.waveform
    t1 = perf_counter()
    total_s = t1 - t0

    n_samples = int(wav.numel())
    n_frames = int(out.codes_b37t.shape[2])
    # TTFT proxy: time for first acoustic frame out of the generate loop
    # (we approximate as total_s / n_frames since there's no streaming yet)
    ttft_s = total_s / n_frames if n_frames > 0 else total_s
    _log_perf("tt_generate", ttft_s=ttft_s, n_frames=n_frames, total_s=total_s)
    if out.shifted_codes_t37.numel() > 0:
        semantic = out.shifted_codes_t37[:, 0]
        logger.info(
            "[tt_generate] semantic shifted-code stats: "
            f"min={int(semantic.min().item())}, max={int(semantic.max().item())}, "
            f"unique={int(semantic.unique().numel())}, first10={semantic[:10].tolist()}"
        )
    if not out.hit_end_audio:
        logger.warning(f"[tt_generate] Reached max_speech_tokens={max_tokens} without END_AUDIO; output was truncated.")

    codes_path = out_path.with_suffix(".codes.pt")
    torch.save(
        {
            "codes_b37t": out.codes_b37t.cpu(),
            "shifted_codes_t37": out.shifted_codes_t37.cpu(),
            "hit_end_audio": out.hit_end_audio,
            "downsample_factor": pipe._downsample_factor,
        },
        codes_path,
    )
    logger.info(f"Saved TT generated codes → {codes_path}")

    _save_wav(out_path, wav, sample_rate)
    logger.info(f"Saved TT waveform → {out_path}  ({n_samples} samples @ {sample_rate} Hz)")


def run_codes_mode(
    pipe: VoxtralTTSPipeline,
    codes_b37t: torch.Tensor,
    sample_rate: int,
    out_path: Path,
) -> None:
    T = int(codes_b37t.shape[2])
    t0 = perf_counter()
    wav = pipe.decode_waveform_from_codes_tt(codes_b37t.long())
    t1 = perf_counter()
    _log_perf("tt_codes_decode", ttft_s=t1 - t0, n_frames=T, total_s=t1 - t0)
    _save_wav(out_path, wav.squeeze(0).squeeze(0), sample_rate)
    logger.info(f"Saved → {out_path}")


def run_latents_mode(
    pipe: VoxtralTTSPipeline,
    latent: torch.Tensor,
    sample_rate: int,
    out_path: Path,
) -> None:
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    if latent.dim() != 4 or int(latent.shape[1]) != 1:
        raise ValueError(f"Expected latent [1,1,T,C], got {tuple(latent.shape)}")
    latent = latent.to(dtype=torch.bfloat16).contiguous()
    t0 = perf_counter()
    lt = ttnn.from_torch(
        latent,
        device=pipe.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mel = pipe.audio_tokenizer.decode_latent_to_mel_b1tc(lt)
    ttnn.deallocate(lt)
    wav = pipe.audio_tokenizer.pretransform_decode_torch(mel)
    ttnn.deallocate(mel)
    t1 = perf_counter()
    T = int(latent.shape[2])
    _log_perf("tt_latents_decode", ttft_s=t1 - t0, n_frames=T, total_s=t1 - t0)
    _save_wav(out_path, wav.squeeze(0).squeeze(0), sample_rate)
    logger.info(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main demo runner
# ---------------------------------------------------------------------------


def run_demo(args: DemoArgs) -> None:
    cfg = load_voxtral_config(args.model.model_name_or_path)
    sample_rate = int(cfg.audio_model_args.audio_encoding_args.sampling_rate)

    items = load_prompt_items(args.data.prompts_file, args.data.default_voice)
    out_dir = Path(args.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh, original = _open_device()
    try:
        logger.info(f"Loading VoxtralTTSPipeline from {args.model.model_name_or_path!r} …")
        t0 = perf_counter()
        pipe = _load_pipeline(mesh, args)
        logger.info(f"Pipeline ready in {(perf_counter() - t0) * 1000:.1f} ms")

        for w in range(args.data.warmup_iters + 1):
            is_warmup = w < args.data.warmup_iters
            tag = "warmup" if is_warmup else "run"

            for item in items:
                pid = item.get("id", 0)

                if args.data.mode == "text":
                    text = item.get("text")
                    if not text:
                        raise ValueError("text mode requires a 'text' field in each JSON entry.")
                    voice = str(item.get("voice", args.data.default_voice))
                    out_path = out_dir / f"{tag}_item{pid}.wav"

                    if is_warmup:
                        # Short warmup — only 4 acoustic frames
                        pipe.generate(text=text, voice=voice, max_tokens=4, seed=args.data.seed)
                        continue

                    run_text_mode(
                        pipe=pipe,
                        text=text,
                        voice=voice,
                        max_tokens=args.data.max_speech_tokens,
                        seed=args.data.seed,
                        sample_rate=sample_rate,
                        out_path=out_path,
                    )

                elif args.data.mode == "codes":
                    if is_warmup:
                        continue
                    raw = item.get("codes")
                    if raw is None:
                        cp = item.get("codes_path")
                        if not cp:
                            raise ValueError("codes mode needs 'codes' or 'codes_path'.")
                        try:
                            raw = torch.load(cp, map_location="cpu", weights_only=False)
                        except TypeError:
                            raw = torch.load(cp, map_location="cpu")
                    codes = torch.as_tensor(raw, dtype=torch.long)
                    if codes.dim() == 2:
                        codes = codes.unsqueeze(0).transpose(1, 2)
                    if codes.dim() != 3 or int(codes.shape[1]) != 37:
                        raise ValueError(f"Expected codes [1,37,T] or [T,37], got {tuple(codes.shape)}")
                    run_codes_mode(pipe, codes, sample_rate, out_dir / f"{tag}_item{pid}.wav")

                elif args.data.mode == "latents":
                    if is_warmup:
                        continue
                    lp = item.get("latent_path")
                    if not lp:
                        raise ValueError("latents mode needs 'latent_path'.")
                    try:
                        lat = torch.load(lp, map_location="cpu", weights_only=False)
                    except TypeError:
                        lat = torch.load(lp, map_location="cpu")
                    run_latents_mode(pipe, lat, sample_rate, out_dir / f"{tag}_item{pid}.wav")

    finally:
        ttnn.SetDefaultDevice(original)
        ttnn.close_device(mesh)


def main(argv: list[str] | None = None) -> None:
    args = _parse_demo_args(argv)
    torch.manual_seed(args.data.seed)
    run_demo(args)


if __name__ == "__main__":
    main()
