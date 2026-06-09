# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure Seamless M4T v2 demo performance across input sequence lengths.

Runs all five demo tasks (T2TT, T2ST, S2TT, S2ST, ASR) with the same warmup/timing
methodology as ``demo.py``, sweeping input sequence length by doubling: 32, 64, 128, …, 4096.

  * Text tasks (T2TT, T2ST) — source token count equals sequence length.
  * Speech tasks (S2TT, S2ST, ASR) — mel-frame count equals sequence length.

Long inputs are prepared once at startup:

  * Text — Alice in Wonderland from Project Gutenberg (repeated/trimmed per length).
  * Audio — preamble WAV downloaded online, concatenated until >= 4096 mel frames.

Results are appended to a text log with per-length headers, TT-aligned metrics, and
decoded task outputs (translated/transcribed text; speech stats and WAV files for T2ST/S2ST).

Run from repo root::

    python models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py

Weights are auto-downloaded on first run via ``ensure_seamless_m4t_v2_large_weights()`` (same as
``demo.py``). Override with ``SEAMLESS_M4T_V2_WEIGHTS=/path/to/seamless-m4t-v2-large``.

Optional::

    python .../demo_perf_sweep.py --output scripts/outputs/perf_sweep.txt --min-len 32 --max-len 512
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.demo import demo as demo_mod
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import source_text_for_enc_len
from models.experimental.seamless_m4t_v2_large.tt.common import hf_aligned_generation_kwargs
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    SeamlessGenerateTimings,
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_LOG = OUTPUT_DIR / "perf_sweep.txt"
STORY_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
STORY_FILE = OUTPUT_DIR / "alice_in_wonderland.txt"
LONG_AUDIO_WAV = OUTPUT_DIR / "long_speech_input.wav"

SRC_LANG = "eng"
TGT_TRANSLATE = "hin"
TGT_S2TT = "hin"
TGT_S2ST = "spa"
TGT_ASR = "eng"

SEQ_LEN_MIN = 32
SEQ_LEN_MAX = 4096

# Multiple speech ``generate()`` calls in one device session (warmup + timed) leave decode-trace
# state that collapses S2TT/S2ST/ASR on the timed run (notably at 2048 mel). Warm on a
# throwaway device, then time on a fresh session (same pattern as cold-start preflight).
_SPEECH_SPLIT_WARMUP_MEL = 1792


def _weights_dir() -> Path:
    """Resolve checkpoint path; download from Hugging Face Hub on first use if missing."""
    env = os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return ensure_seamless_m4t_v2_large_weights()


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _download_bytes(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        if not data:
            raise RuntimeError("response body was empty")
        tmp.write_bytes(data)
        tmp.replace(dest)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _clean_gutenberg_text(raw: str) -> str:
    """Strip Gutenberg header/footer boilerplate; keep story body."""
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw, re.IGNORECASE)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw, re.IGNORECASE)
    if start and end:
        raw = raw[start.end() : end.start()]
    raw = re.sub(r"\r\n?", "\n", raw)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def ensure_long_story(url: str = STORY_URL, dest: Path = STORY_FILE) -> str:
    """Download and cache a long English story for text-input sweeps."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest.read_text(encoding="utf-8", errors="replace")
    _download_bytes(url, dest)
    text = _clean_gutenberg_text(dest.read_text(encoding="utf-8", errors="replace"))
    if len(text) < 1000:
        raise RuntimeError(f"Downloaded story at {dest} is unexpectedly short ({len(text)} chars)")
    dest.write_text(text, encoding="utf-8")
    return text


def _mel_frame_count(processor: AutoProcessor, waveform: np.ndarray, sample_rate: int) -> int:
    audio = processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt")
    return int(audio["attention_mask"].sum().item())


def ensure_long_audio(
    processor: AutoProcessor,
    sample_rate: int,
    *,
    min_mel_frames: int = SEQ_LEN_MAX,
    url: str = demo_mod.PREAMBLE_WAV_URL,
    dest: Path = LONG_AUDIO_WAV,
) -> tuple[np.ndarray, Path]:
    """Build a mono 16 kHz waveform with at least ``min_mel_frames`` mel frames.

    Downloads ``url`` (online preamble speech by default) and concatenates copies until the
    processor timeline is long enough, then caches ``dest``.
    """
    if dest.is_file() and dest.stat().st_size > 0:
        wav, _ = demo_mod._load_mono_wav_resampled(dest, sample_rate)
        if _mel_frame_count(processor, wav, sample_rate) >= min_mel_frames:
            return wav, dest

    preamble_path = demo_mod.ensure_demo_audio(url=url, dest=demo_mod.PREAMBLE_WAV)
    chunk, _ = demo_mod._load_mono_wav_resampled(preamble_path, sample_rate)
    if chunk.size == 0:
        raise RuntimeError(f"Preamble audio at {preamble_path} is empty")

    wav = chunk.copy()
    while _mel_frame_count(processor, wav, sample_rate) < min_mel_frames:
        wav = np.concatenate([wav, chunk])

    demo_mod._save_wav(dest, wav, sample_rate=sample_rate)
    got = _mel_frame_count(processor, wav, sample_rate)
    if got < min_mel_frames:
        raise RuntimeError(f"Long audio has {got} mel frames, need >= {min_mel_frames}")
    return wav, dest


def text_inputs_for_len(
    processor: AutoProcessor,
    story: str,
    target_tokens: int,
    *,
    src_lang: str = SRC_LANG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize ``story`` to exactly ``target_tokens`` source tokens."""
    unit = story if len(story) <= 4096 else story[:4096]
    return source_text_for_enc_len(processor, target_tokens, src_lang=src_lang, unit=unit + " ")


def speech_inputs_for_len(
    processor: AutoProcessor,
    full_features: torch.Tensor,
    full_mask: torch.Tensor,
    target_mel_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate processor features to exactly ``target_mel_frames`` mel frames."""
    available = int(full_mask.sum().item())
    if available < target_mel_frames:
        raise ValueError(f"need {target_mel_frames} mel frames, only {available} available")
    feats = full_features[:, :target_mel_frames, :].contiguous()
    mask = full_mask[:, :target_mel_frames].contiguous()
    return feats, mask


def sequence_lengths(min_len: int, max_len: int) -> list[int]:
    """Return ``min_len``, then double each step until ``max_len`` (inclusive)."""
    if min_len < 1 or max_len < min_len:
        raise ValueError(f"invalid sweep: min={min_len}, max={max_len}")
    lengths: list[int] = []
    n = min_len
    while n <= max_len:
        lengths.append(n)
        n *= 2
    return lengths


# ---------------------------------------------------------------------------
# Logging / formatting
# ---------------------------------------------------------------------------


class PerfLog:
    def __init__(self, path: Path, *, also_stdout: bool = True) -> None:
        self.path = path
        self.also_stdout = also_stdout
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = path.open("w", encoding="utf-8")

    def write(self, line: str = "") -> None:
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.also_stdout:
            print(line)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> PerfLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _task_metrics_line(task: str, timings: SeamlessGenerateTimings) -> str:
    steady = timings.steady_decode_ms_per_tok
    decode_tps = timings.decode_tok_s_u
    line = (
        f"  {task} TT metrics: TTFT {timings.ttft_ms:.1f} ms, "
        f"encoder {timings.encoder_ms:.1f} ms, prefill {timings.prefill_ms:.1f} ms, "
        f"decode {decode_tps:.2f} t/s/u ({steady:.1f} ms/tok steady), "
        f"e2e {timings.e2e_ms:.1f} ms ({timings.output_tokens} out tokens)"
    )
    if timings.t2u_ms > 0 or timings.vocoder_ms > 0:
        rtf_str = f", RTF {timings.rtf:.2f}x" if timings.output_samples > 0 else ""
        line += (
            f"\n  {task} speech synth: T2U {timings.t2u_ms:.1f} ms, "
            f"vocoder {timings.vocoder_ms:.1f} ms, "
            f"TTFT audio {timings.ttft_audio_ms:.1f} ms{rtf_str}"
        )
    if timings.input_is_speech and timings.mel_frames > 0:
        line += f"\n  {task} input: {timings.mel_frames} mel frames"
    return line


def _summary_table(perf_tt: list[tuple[str, SeamlessGenerateTimings]]) -> list[str]:
    lines = [
        "-" * 78,
        "  TT-aligned runtime summary (phase-separated; decode t/s/u = 1000 / steady ms/tok)",
        "-" * 78,
        f"  {'Task':<6} {'TTFT':>8} {'Enc':>8} {'Pref':>8} " f"{'Dec t/s/u':>10} {'ms/tok':>8} {'E2E':>9} {'Out':>8}",
    ]
    for task_name, timings in perf_tt:
        out = f"{timings.output_tokens}tok"
        if timings.output_samples > 0:
            out = f"{timings.output_samples}smp"
        lines.append(
            f"  {task_name:<6} {timings.ttft_ms:>7.1f}ms {timings.encoder_ms:>7.1f}ms "
            f"{timings.prefill_ms:>7.1f}ms {timings.decode_tok_s_u:>10.2f} "
            f"{timings.steady_decode_ms_per_tok:>7.1f}ms {timings.e2e_ms:>8.1f}ms {out:>8}"
        )
        if timings.output_samples > 0:
            lines.append(
                f"         T2U {timings.t2u_ms:.0f} ms  vocoder {timings.vocoder_ms:.0f} ms  " f"RTF {timings.rtf:.2f}x"
            )
    lines.extend(
        [
            "-" * 78,
            "  decode t/s/u = 1000 / steady ms/tok (text-decoder steps 2+). "
            "TTFT includes encoder + prefill + first token. "
            "E2E includes T2U/vocoder on speech tasks.",
        ]
    )
    return lines


def _log_text_task_output(log: PerfLog, task: str, tgt_lang: str, text: str) -> None:
    log.write(f"  {task} output ({tgt_lang}): {text}")


def _speech_wav_path(task: str, seq_len: int) -> Path:
    """Per sweep-point WAV under ``scripts/outputs`` (e.g. ``t2st_seq4096.wav``)."""
    return OUTPUT_DIR / f"{task.lower()}_seq{seq_len}.wav"


def _log_speech_task_output(
    log: PerfLog,
    task: str,
    *,
    seq_len: int,
    tgt_lang: str,
    text: str,
    wav_np: np.ndarray,
    sample_rate: int,
    text_tokens: int,
    unit_frames: int,
) -> None:
    wav_path = _speech_wav_path(task, seq_len)
    demo_mod._save_wav(wav_path, wav_np, sample_rate=sample_rate)
    log.write(f"  {task} intermediate text ({tgt_lang}): {text}")
    dur_s = wav_np.size / sample_rate
    log.write(
        f"  {task} stats: text_tokens={text_tokens}, unit_frames={unit_frames}, "
        f"audio={wav_np.size} samples ({dur_s:.2f}s @ {sample_rate} Hz)"
    )
    log.write(f"  {task} saved WAV: {wav_path}")


def _seq_len_header(seq_len: int) -> list[str]:
    return [
        "",
        "=" * 78,
        f"  Sequence length: {seq_len}",
        "=" * 78,
    ]


# ---------------------------------------------------------------------------
# Per-length benchmark
# ---------------------------------------------------------------------------


def _speech_throwaway_warmups(
    *,
    session_kw: dict[str, Any],
    speech_feats: torch.Tensor,
    speech_attn: torch.Tensor,
    mel_frames: int,
    gen_common: dict[str, Any],
    iters: int,
    task: str,
) -> None:
    """Untimed speech runs on a throwaway mesh so the timed session stays decode-trace clean."""
    if iters <= 0:
        return
    with demo_mod._isolated_task_session(**session_kw) as (device, tt_model):
        demo_mod._prewarm_speech_encoder(tt_model, mel_frames)
        feats_tt = demo_mod.torch_feats_to_ttnn(device, speech_feats)
        attn_tt = demo_mod.torch_ids_to_ttnn(device, speech_attn)
        for _ in range(iters):
            if task == "S2ST":
                out = tt_model.generate(
                    input_features=feats_tt,
                    attention_mask=attn_tt,
                    generate_speech=True,
                    return_intermediate_token_ids=True,
                    tgt_lang=TGT_S2ST,
                    speaker_id=0,
                    **gen_common,
                )
                demo_mod._release_speech_out(out)
            else:
                tgt_lang = TGT_S2TT if task == "S2TT" else TGT_ASR
                out = tt_model.generate(
                    input_features=feats_tt,
                    attention_mask=attn_tt,
                    generate_speech=False,
                    tgt_lang=tgt_lang,
                    **gen_common,
                )
                ttnn.deallocate(out.sequences)
            ttnn.synchronize_device(device)
        ttnn.deallocate(feats_tt)
        ttnn.deallocate(attn_tt)


def _run_five_tasks_at_seq_len(
    seq_len: int,
    *,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    sample_rate: int,
    t2u_pad_id: int,
    story: str,
    full_speech_features: torch.Tensor,
    full_speech_mask: torch.Tensor,
    session_kw: dict[str, Any],
    log: PerfLog,
    continue_on_error: bool = False,
) -> tuple[list[tuple[str, SeamlessGenerateTimings]], list[str]]:
    gen_common = session_kw["gen_common"]
    perf_tt: list[tuple[str, SeamlessGenerateTimings]] = []
    failed_tasks: list[str] = []

    def _task_failed(task: str, exc: Exception) -> None:
        failed_tasks.append(task)
        log.write(f"  {task} FAILED: {exc}")
        if not continue_on_error:
            raise exc

    input_ids, input_text_attn = text_inputs_for_len(processor, story, seq_len)
    speech_feats, speech_attn = speech_inputs_for_len(processor, full_speech_features, full_speech_mask, seq_len)
    mel_frames = int(speech_attn.sum().item())

    # 1. T2TT
    log.write(f"  [1/5] T2TT @ {seq_len} tokens")
    try:
        with demo_mod._isolated_task_session(**session_kw) as (device, tt_model):
            ids_tt = demo_mod.torch_ids_to_ttnn(device, input_ids)
            attn_tt = demo_mod.torch_ids_to_ttnn(device, input_text_attn)
            out, _ = demo_mod._warmup_and_time(
                device,
                lambda: tt_model.generate(
                    input_ids=ids_tt,
                    attention_mask=attn_tt,
                    generate_speech=False,
                    tgt_lang=TGT_TRANSLATE,
                    **gen_common,
                ),
                release_fn=lambda o: ttnn.deallocate(o.sequences),
            )
            if not isinstance(out, TTSeamlessM4Tv2GreedySearchOutput):
                raise TypeError(f"T2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(out)}")
            if out.timings is not None:
                perf_tt.append(("T2TT", out.timings))
                log.write(_task_metrics_line("T2TT", out.timings))
            _log_text_task_output(log, "T2TT", TGT_TRANSLATE, demo_mod._decode(tokenizer, out.sequences))
            ttnn.deallocate(out.sequences)
    except Exception as exc:
        _task_failed("T2TT", exc)

    # 2. T2ST
    log.write(f"  [2/5] T2ST @ {seq_len} tokens")
    try:
        with demo_mod._isolated_task_session(**session_kw) as (device, tt_model):
            ids_tt = demo_mod.torch_ids_to_ttnn(device, input_ids)
            attn_tt = demo_mod.torch_ids_to_ttnn(device, input_text_attn)
            out, _ = demo_mod._warmup_and_time(
                device,
                lambda: tt_model.generate(
                    input_ids=ids_tt,
                    attention_mask=attn_tt,
                    generate_speech=True,
                    return_intermediate_token_ids=True,
                    tgt_lang=TGT_TRANSLATE,
                    speaker_id=0,
                    **gen_common,
                ),
                release_fn=demo_mod._release_speech_out,
                warmup_iters=demo_mod._DEMO_SPEECH_WARMUP_ITERS,
            )
            if not isinstance(out, TTSeamlessM4Tv2GenerationOutput):
                raise TypeError(f"T2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(out)}")
            if out.timings is not None:
                perf_tt.append(("T2ST", out.timings))
                log.write(_task_metrics_line("T2ST", out.timings))
            t2st_text = demo_mod._decode(tokenizer, out.sequences)
            t2st_wav = demo_mod._waveform_to_mono_fp32(out.waveform, out.waveform_lengths)
            _log_speech_task_output(
                log,
                "T2ST",
                seq_len=seq_len,
                tgt_lang=TGT_TRANSLATE,
                text=t2st_text,
                wav_np=t2st_wav,
                sample_rate=sample_rate,
                text_tokens=demo_mod._tt_row_length(out.sequences),
                unit_frames=demo_mod._valid_unit_frames(out.unit_sequences, pad_id=t2u_pad_id),
            )
            demo_mod._release_speech_out(out)
    except Exception as exc:
        _task_failed("T2ST", exc)

    speech_split_warmup = mel_frames >= _SPEECH_SPLIT_WARMUP_MEL
    speech_measure_warmups = 0 if speech_split_warmup else demo_mod._DEMO_SPEECH_WARMUP_ITERS
    if speech_split_warmup:
        _speech_throwaway_warmups(
            session_kw=session_kw,
            speech_feats=speech_feats,
            speech_attn=speech_attn,
            mel_frames=mel_frames,
            gen_common=gen_common,
            iters=demo_mod._DEMO_SPEECH_WARMUP_ITERS,
            task="S2TT",
        )

    # 3. S2TT
    log.write(f"  [3/5] S2TT @ {seq_len} mel frames")
    try:
        with demo_mod._isolated_task_session(**session_kw) as (device, tt_model):
            demo_mod._prewarm_speech_encoder(tt_model, mel_frames)
            feats_tt = demo_mod.torch_feats_to_ttnn(device, speech_feats)
            attn_tt = demo_mod.torch_ids_to_ttnn(device, speech_attn)
            out, _ = demo_mod._warmup_and_time(
                device,
                lambda: tt_model.generate(
                    input_features=feats_tt,
                    attention_mask=attn_tt,
                    generate_speech=False,
                    tgt_lang=TGT_S2TT,
                    **gen_common,
                ),
                release_fn=lambda o: ttnn.deallocate(o.sequences),
                warmup_iters=speech_measure_warmups,
            )
            if not isinstance(out, TTSeamlessM4Tv2GreedySearchOutput):
                raise TypeError(f"S2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(out)}")
            if out.timings is not None:
                perf_tt.append(("S2TT", out.timings))
                log.write(_task_metrics_line("S2TT", out.timings))
            _log_text_task_output(log, "S2TT", TGT_S2TT, demo_mod._decode(tokenizer, out.sequences))
            ttnn.deallocate(out.sequences)
    except Exception as exc:
        _task_failed("S2TT", exc)

    # 4. S2ST
    log.write(f"  [4/5] S2ST @ {seq_len} mel frames")
    try:
        s2st_measure_warmups = speech_measure_warmups
        if speech_split_warmup:
            _speech_throwaway_warmups(
                session_kw=session_kw,
                speech_feats=speech_feats,
                speech_attn=speech_attn,
                mel_frames=mel_frames,
                gen_common=gen_common,
                iters=demo_mod._DEMO_SPEECH_WARMUP_ITERS,
                task="S2ST",
            )
        with demo_mod._isolated_task_session(**session_kw) as (device, tt_model):
            demo_mod._prewarm_speech_encoder(tt_model, mel_frames)
            feats_tt = demo_mod.torch_feats_to_ttnn(device, speech_feats)
            attn_tt = demo_mod.torch_ids_to_ttnn(device, speech_attn)
            out, _ = demo_mod._warmup_and_time(
                device,
                lambda: tt_model.generate(
                    input_features=feats_tt,
                    attention_mask=attn_tt,
                    generate_speech=True,
                    return_intermediate_token_ids=True,
                    tgt_lang=TGT_S2ST,
                    speaker_id=0,
                    **gen_common,
                ),
                release_fn=demo_mod._release_speech_out,
                warmup_iters=s2st_measure_warmups,
            )
            if not isinstance(out, TTSeamlessM4Tv2GenerationOutput):
                raise TypeError(f"S2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(out)}")
            if out.timings is not None:
                perf_tt.append(("S2ST", out.timings))
                log.write(_task_metrics_line("S2ST", out.timings))
            s2st_text = demo_mod._decode(tokenizer, out.sequences)
            s2st_wav = demo_mod._waveform_to_mono_fp32(out.waveform, out.waveform_lengths)
            _log_speech_task_output(
                log,
                "S2ST",
                seq_len=seq_len,
                tgt_lang=TGT_S2ST,
                text=s2st_text,
                wav_np=s2st_wav,
                sample_rate=sample_rate,
                text_tokens=demo_mod._tt_row_length(out.sequences),
                unit_frames=demo_mod._valid_unit_frames(out.unit_sequences, pad_id=t2u_pad_id),
            )
            demo_mod._release_speech_out(out)
    except Exception as exc:
        _task_failed("S2ST", exc)

    # 5. ASR
    log.write(f"  [5/5] ASR @ {seq_len} mel frames")
    try:
        if speech_split_warmup:
            _speech_throwaway_warmups(
                session_kw=session_kw,
                speech_feats=speech_feats,
                speech_attn=speech_attn,
                mel_frames=mel_frames,
                gen_common=gen_common,
                iters=demo_mod._DEMO_SPEECH_WARMUP_ITERS,
                task="ASR",
            )
        with demo_mod._isolated_task_session(**session_kw) as (device, tt_model):
            demo_mod._prewarm_speech_encoder(tt_model, mel_frames)
            feats_tt = demo_mod.torch_feats_to_ttnn(device, speech_feats)
            attn_tt = demo_mod.torch_ids_to_ttnn(device, speech_attn)
            out, _ = demo_mod._warmup_and_time(
                device,
                lambda: tt_model.generate(
                    input_features=feats_tt,
                    attention_mask=attn_tt,
                    generate_speech=False,
                    tgt_lang=TGT_ASR,
                    **gen_common,
                ),
                release_fn=lambda o: ttnn.deallocate(o.sequences),
                warmup_iters=speech_measure_warmups,
            )
            if not isinstance(out, TTSeamlessM4Tv2GreedySearchOutput):
                raise TypeError(f"ASR expected TTSeamlessM4Tv2GreedySearchOutput, got {type(out)}")
            if out.timings is not None:
                perf_tt.append(("ASR", out.timings))
                log.write(_task_metrics_line("ASR", out.timings))
            _log_text_task_output(log, "ASR", TGT_ASR, demo_mod._decode(tokenizer, out.sequences))
            ttnn.deallocate(out.sequences)
    except Exception as exc:
        _task_failed("ASR", exc)

    return perf_tt, failed_tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seamless M4T v2 demo performance sweep")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Log file path (default: {DEFAULT_LOG})",
    )
    parser.add_argument("--min-len", type=int, default=SEQ_LEN_MIN, help="First sequence length (then double)")
    parser.add_argument("--max-len", type=int, default=SEQ_LEN_MAX, help="Last sequence length (power-of-two sweep)")
    parser.add_argument("--story-url", type=str, default=STORY_URL, help="URL for long English story text")
    parser.add_argument(
        "--audio-url",
        type=str,
        default=demo_mod.PREAMBLE_WAV_URL,
        help="URL for seed speech WAV (concatenated to reach max mel frames)",
    )
    parser.add_argument("--quiet", action="store_true", help="Write log file only (no stdout)")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Log failures and continue sweep (default: on)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first sequence-length error (disables --continue-on-error)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    lengths = sequence_lengths(args.min_len, args.max_len)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weights_dir = _weights_dir()
    path = str(weights_dir)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    hf_model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = hf_model.t2u_model.config
    sample_rate = int(getattr(cfg, "sampling_rate", 16000))

    use_decode_trace = True
    use_2cq = True
    gen_common = hf_aligned_generation_kwargs(
        hf_model.generation_config,
        use_kv_cache=True,
        use_decode_trace=use_decode_trace,
        use_2cq=use_2cq,
        return_timings=True,
    )

    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        # Default device may be unset before the first perf sweep task opens a mesh.
        original_default = None

    session_kw = dict(
        hf_model=hf_model,
        cfg=cfg,
        t2u_cfg=t2u_cfg,
        gen_common=gen_common,
        original_default=original_default,
    )

    story = ensure_long_story(url=args.story_url)
    long_wav, long_wav_path = ensure_long_audio(
        processor,
        sample_rate,
        min_mel_frames=args.max_len,
        url=args.audio_url,
    )
    full_audio = processor(audios=long_wav, sampling_rate=sample_rate, return_tensors="pt")
    full_speech_features = full_audio["input_features"].to(torch.bfloat16)
    full_speech_mask = full_audio["attention_mask"]
    max_mel_frames = int(full_speech_mask.sum().item())

    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sweep_t0 = time.perf_counter()

    with PerfLog(args.output.expanduser().resolve(), also_stdout=not args.quiet) as log:
        log.write("Seamless M4T v2 Large — performance sweep")
        log.write(f"  Started: {started}")
        log.write(f"  Log file: {args.output.resolve()}")
        log.write(f"  Sequence lengths (doubling): {', '.join(str(n) for n in lengths)}")
        log.write(
            f"  Demo mode: one mesh device open/close per task — decode: "
            f"{'trace+2CQ' if (use_decode_trace and use_2cq) else 'eager'}"
        )
        log.write(
            f"  Warmup: text={demo_mod._DEMO_WARMUP_ITERS}, "
            f"speech={demo_mod._DEMO_SPEECH_WARMUP_ITERS} untimed iter(s), "
            f"then {demo_mod._DEMO_MEASURE_ITERS} timed (report min elapsed)"
        )
        log.write(f"  Text input story: {STORY_FILE} ({len(story)} chars from {args.story_url})")
        log.write(
            f"  Speech input: {long_wav_path} ({long_wav.size} samples @ {sample_rate} Hz, "
            f"{long_wav.size / sample_rate:.2f}s, mel_frames={max_mel_frames})"
        )

        probe_ids, probe_attn = text_inputs_for_len(processor, story, lengths[0])
        demo_mod._process_jit_preflight(
            session_kw=session_kw,
            input_ids=probe_ids,
            input_text_attn=probe_attn,
            gen_common=gen_common,
        )

        continue_on_error = args.continue_on_error and not args.fail_fast
        completed = 0
        skipped: list[int] = []

        for idx, seq_len in enumerate(lengths, start=1):
            log.write("")
            log.write(f"--- sweep point {idx}/{len(lengths)} ---")
            for line in _seq_len_header(seq_len):
                log.write(line)

            perf_tt, failed_tasks = _run_five_tasks_at_seq_len(
                seq_len,
                processor=processor,
                tokenizer=tokenizer,
                sample_rate=sample_rate,
                t2u_pad_id=int(t2u_cfg.pad_token_id),
                story=story,
                full_speech_features=full_speech_features,
                full_speech_mask=full_speech_mask,
                session_kw=session_kw,
                log=log,
                continue_on_error=continue_on_error,
            )
            if failed_tasks:
                log.write(f"  Failed tasks at sequence length {seq_len}: {', '.join(failed_tasks)}")
                if continue_on_error:
                    skipped.append(seq_len)
                elif not perf_tt:
                    raise RuntimeError(f"all tasks failed at sequence length {seq_len}")

            if perf_tt:
                for line in _summary_table(perf_tt):
                    log.write(line)
            if not failed_tasks:
                completed += 1

        elapsed_min = (time.perf_counter() - sweep_t0) / 60.0
        log.write("")
        log.write("=" * 78)
        if skipped:
            log.write(
                f"  done — {completed}/{len(lengths)} sequence lengths OK, "
                f"skipped: {', '.join(str(n) for n in skipped)}"
            )
        else:
            log.write(f"  ok — performance sweep completed ({len(lengths)} sequence lengths)")
        log.write(f"  Total wall time: {elapsed_min:.1f} min")
        log.write("=" * 78)


if __name__ == "__main__":
    main()
