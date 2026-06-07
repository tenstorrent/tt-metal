# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bilingual (Japanese <-> English) speech-to-text translation demo app.

Takes one audio file that contains a conversation where Japanese and English
turns alternate.  The app:

  1. splits the audio into utterances with an energy-based VAD (``librosa``),
  2. for each utterance, identifies the spoken language (JP vs EN) using a
     **Seamless self-score LID**: the speech is encoded once, then the text
     decoder is run a few steps toward each candidate language and the mean
     per-token log-probability (decoder confidence) is compared — the model is
     more confident when it transcribes its own language than when it
     translates, so the higher-scoring language is taken as the source,
  3. translates each utterance into the *other* language with the model's
     fast KV-cache + trace ``generate`` path (JP->EN, EN->JP),
  4. prints the result to stdout and (optionally) appends it to a text file.

The expensive TT model is built **once** and reused across all utterances.

Run from repo root (inside the seamless container, single Blackhole chip):

  SEAMLESS_FORCE_1x1=1 TT_VISIBLE_DEVICES=1 \
    python models/experimental/seamless_m4t_v2_large/demo/bilingual_s2tt.py \
      --audio path/to/conversation.wav --out outputs/translation.txt

If you don't have a bilingual test file, generate one first (uses T2ST to
synthesize a Japanese turn followed by an English turn):

  ... bilingual_s2tt.py --make-test-audio outputs/jp_en_test.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse helpers already written for the full demo.
from models.experimental.seamless_m4t_v2_large.demo.demo import (
    _decode,
    _save_wav,
    _waveform_to_mono_fp32,
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import hf_aligned_generation_kwargs
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    DEVICE_PARAMS_P150_E2E_2CQ_GENERATE,
    MESH_SHAPE_P150,
    open_seamless_mesh_device,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
    _ttnn_ids_from_list,
)

SAMPLE_RATE = 16000
LANG_NAME = {"jpn": "Japanese", "eng": "English"}
LANG_SHORT = {"jpn": "JA", "eng": "EN"}

# How many decoder steps to score for LID. Short keeps the (full-vocab) lm_head
# cost low; the source/translation gap shows up within the first few tokens.
_LID_STEPS = 8


# ---------------------------------------------------------------------------
# Weights / audio I/O
# ---------------------------------------------------------------------------


def _weights_dir() -> Path:
    env = os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return ensure_seamless_m4t_v2_large_weights()


def _load_audio_16k_mono(path: Path) -> np.ndarray:
    """Load any soundfile/librosa-readable audio, resampled to 16 kHz mono fp32."""
    import librosa

    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def _download_youtube_audio(url: str, out_dir: Path) -> Path:
    """Download a YouTube URL's audio track and extract it to a 16 kHz mono WAV.

    Uses the ``yt-dlp`` CLI + ``ffmpeg`` (present in the ``*_yt`` container image).
    Returns the path to the produced WAV.
    """
    if shutil.which("yt-dlp") is None:
        raise SystemExit(
            "yt-dlp not found. Run with the YouTube-enabled image "
            "(IMAGE=tt-metalium-dev:seamless_m4t_v2_yt ./run_bili.sh ...)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "_yt_audio.%(ext)s")
    wav = out_dir / "_yt_audio.wav"
    if wav.exists():
        wav.unlink()
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--no-playlist",
        "-f",
        "bestaudio",
        "-x",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        "ffmpeg:-ar 16000 -ac 1",  # 16 kHz mono = the model's input rate
        "-o",
        template,
        url,
    ]
    print(f"  Downloading audio track from: {url}")
    subprocess.run(cmd, check=True)
    if not wav.exists():
        raise SystemExit(f"yt-dlp finished but {wav} was not produced.")
    return wav


def _fmt_ts(sec: float) -> str:
    m, s = divmod(sec, 60.0)
    return f"{int(m):02d}:{s:04.1f}"


def _pad_mel_to_long_path(feats: torch.Tensor, sattn: torch.Tensor, min_frames: int):
    """Right-pad mel features (zeros) + attention mask (zeros) to >=``min_frames``.

    The speech encoder switches to its proven long-audio path above
    ``_LONG_AUDIO_RES_DRAM_THRESHOLD`` (1024) mel frames; a mid-length sub-range
    below that crashes ("Tensor is not allocated"). Padding short segments up to
    ``min_frames`` (>1024) routes every encode through the working long path. The
    padded frames carry mask=0, so the encoder masks them out and the real-frame
    output is unchanged.
    """
    cur = int(feats.shape[1])
    if min_frames <= 0 or cur >= min_frames:
        return feats, sattn
    pad = min_frames - cur
    feats = torch.nn.functional.pad(feats, (0, 0, 0, pad))  # [B, T, F] -> pad T (dim 1)
    sattn = torch.nn.functional.pad(sattn, (0, pad))  # [B, T] -> pad T
    return feats, sattn


# ---------------------------------------------------------------------------
# VAD segmentation (energy based; no dedicated VAD lib in the container)
# ---------------------------------------------------------------------------


def segment_audio(
    y: np.ndarray,
    *,
    top_db: float = 30.0,
    min_dur: float = 0.4,
    merge_gap: float = 0.3,
    max_dur: float = 3.5,
    pad: float = 0.1,
) -> List[Tuple[int, int]]:
    """Split ``y`` into speech intervals ``[(start, end), ...]`` in samples.

    Uses ``librosa.effects.split`` (level-based) then merges intervals closer
    than ``merge_gap`` seconds, drops intervals shorter than ``min_dur``, pads
    each surviving interval by ``pad`` seconds on both sides, and finally splits
    any interval longer than ``max_dur`` seconds into equal windows.

    ``max_dur`` keeps each utterance in the speech encoder's well-tested
    short-audio regime — its mid-length path has a latent buffer bug
    ("Tensor is not allocated") that this app's per-segment encodes can trip,
    while ~3 s windows are stable.  Set ``max_dur<=0`` to disable splitting.
    """
    import librosa

    raw = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    if len(raw) == 0:
        return []

    merge_samples = int(merge_gap * SAMPLE_RATE)
    merged: List[List[int]] = [list(raw[0])]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= merge_samples:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_samples = int(min_dur * SAMPLE_RATE)
    pad_samples = int(pad * SAMPLE_RATE)
    max_samples = int(max_dur * SAMPLE_RATE) if max_dur and max_dur > 0 else 0
    out: List[Tuple[int, int]] = []
    for s, e in merged:
        if e - s < min_samples:
            continue
        s = max(0, s - pad_samples)
        e = min(len(y), e + pad_samples)
        if max_samples and (e - s) > max_samples:
            n = int(np.ceil((e - s) / max_samples))
            bounds = np.linspace(s, e, n + 1).astype(int)
            out.extend((int(bounds[i]), int(bounds[i + 1])) for i in range(n))
        else:
            out.append((s, e))
    return out


# ---------------------------------------------------------------------------
# Self-score LID
# ---------------------------------------------------------------------------


def _greedy_score_lowlevel(
    tt_model,
    enc_tt: ttnn.Tensor,
    enc_attn_tt: ttnn.Tensor,
    lang_code_id: int,
    *,
    steps: int,
    eos_ids: set,
) -> float:
    """Mean per-token log-prob of a short greedy decode toward ``lang_code_id``.

    Teacher-forced ``_decode_and_lm_head`` over the running sequence (no KV
    cache — fine for the handful of LID steps).  Reuses the already-computed
    speech-encoder output ``enc_tt`` so LID costs one encode for both languages.
    """
    ds = int(tt_model.decoder_start_token_id)
    seq = [ds, int(lang_code_id)]
    logps: List[float] = []
    for _ in range(steps):
        ids_tt = _ttnn_ids_from_list([seq], tt_model.device)
        logits = tt_model._decode_and_lm_head(enc_tt, enc_attn_tt, ids_tt, None)
        ttnn.deallocate(ids_tt)
        row = tt_model._logits_row_to_host(logits, len(seq), sharded=tt_model._tp > 1)[0, : tt_model.vocab_size]
        ttnn.deallocate(logits)
        nxt = int(torch.argmax(row).item())
        logps.append(float(torch.log_softmax(row, dim=-1)[nxt].item()))
        seq.append(nxt)
        if nxt in eos_ids:
            break
    return float(np.mean(logps)) if logps else -1e9


def detect_source_lang(tt_model, gc, feats_tt, attn_tt, eos_ids: set) -> Tuple[str, dict]:
    """Encode once, score both languages, return (source_lang, {lang: score}).

    The caller must have prewarmed the speech encoder for this segment's mel length
    (see the segment loop): the long-audio encoder path is unstable on a cold compile.
    """
    enc_tt, enc_attn_tt, _ = tt_model._encode_speech(feats_tt, attn_tt)
    try:
        scores = {
            lang: _greedy_score_lowlevel(
                tt_model,
                enc_tt,
                enc_attn_tt,
                int(gc.text_decoder_lang_to_code_id[lang]),
                steps=_LID_STEPS,
                eos_ids=eos_ids,
            )
            for lang in ("jpn", "eng")
        }
    finally:
        ttnn.deallocate(enc_tt)
        ttnn.deallocate(enc_attn_tt)
    tt_model.clear_runtime_program_cache()
    ttnn.synchronize_device(tt_model.device)
    src = max(scores, key=scores.get)
    return src, scores


# ---------------------------------------------------------------------------
# Fast full translation via generate()
# ---------------------------------------------------------------------------


def translate_to(tt_model, tokenizer, feats_tt, attn_tt, tgt_lang: str, gen_common: dict) -> str:
    out = tt_model.generate(
        input_features=feats_tt,
        attention_mask=attn_tt,
        generate_speech=False,
        tgt_lang=tgt_lang,
        **gen_common,
    )
    if not isinstance(out, TTSeamlessM4Tv2GreedySearchOutput):
        raise TypeError(f"expected TTSeamlessM4Tv2GreedySearchOutput, got {type(out)}")
    text = _decode(tokenizer, out.sequences)
    ttnn.deallocate(out.sequences)
    tt_model.clear_runtime_program_cache()
    ttnn.synchronize_device(tt_model.device)
    return text


# ---------------------------------------------------------------------------
# Device lifecycle
# ---------------------------------------------------------------------------


def _open_device():
    if os.environ.get("SEAMLESS_FORCE_1x1") == "1":
        device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*MESH_SHAPE_P150),
            **dict(DEVICE_PARAMS_P150_E2E_2CQ_GENERATE),
        )
        return device, MESH_SHAPE_P150
    return open_seamless_mesh_device(enable_decode_trace=True, enable_2cq=True)


# ---------------------------------------------------------------------------
# Test-audio generator (JP turn + EN turn) via T2ST
# ---------------------------------------------------------------------------

_TEST_JP_TEXT = "こんにちは、今日は天気がとても良いので、午後から海まで散歩に行きたいと思っています。"
_TEST_EN_TEXT = "Thanks for waiting. Let me know when you are ready and we can start the meeting together."


def make_test_audio(tt_model, processor, cfg, out_path: Path, gen_common: dict) -> None:
    """Synthesize a Japanese turn then an English turn and concatenate them."""

    def _t2st(text: str, src_lang: str) -> np.ndarray:
        ti = processor(text=text, src_lang=src_lang, return_tensors="pt")
        ids_tt = torch_ids_to_ttnn(tt_model.device, ti["input_ids"])
        attn_tt = torch_ids_to_ttnn(tt_model.device, ti["attention_mask"])
        out = tt_model.generate(
            input_ids=ids_tt,
            attention_mask=attn_tt,
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang=src_lang,  # speak in the same language as the text
            speaker_id=0,
            **gen_common,
        )
        if not isinstance(out, TTSeamlessM4Tv2GenerationOutput):
            raise TypeError(f"T2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(out)}")
        wav = _waveform_to_mono_fp32(out.waveform, out.waveform_lengths)
        ttnn.deallocate(out.waveform)
        ttnn.deallocate(out.waveform_lengths)
        if getattr(out, "sequences", None) is not None:
            ttnn.deallocate(out.sequences)
        if getattr(out, "unit_sequences", None) is not None:
            ttnn.deallocate(out.unit_sequences)
        tt_model.clear_runtime_program_cache()
        ttnn.synchronize_device(tt_model.device)
        return wav

    print(f"  Synthesizing JP turn: {_TEST_JP_TEXT}")
    jp = _t2st(_TEST_JP_TEXT, "jpn")
    print(f"  Synthesizing EN turn: {_TEST_EN_TEXT}")
    en = _t2st(_TEST_EN_TEXT, "eng")
    gap = np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32)
    combined = np.concatenate([jp, gap, en]).astype(np.float32)
    _save_wav(out_path, combined, sample_rate=SAMPLE_RATE)
    print(f"  Saved bilingual test audio ({combined.size} samples, {combined.size / SAMPLE_RATE:.2f}s) -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, help="Input audio file (WAV/FLAC/MP3 ...).")
    ap.add_argument(
        "--youtube",
        type=str,
        metavar="URL",
        help="YouTube URL: download the audio track, extract to 16 kHz mono WAV (yt-dlp+ffmpeg), and run it. Requires the *_yt image.",
    )
    ap.add_argument("--out", type=Path, help="Append the translation transcript to this text file.")
    ap.add_argument(
        "--source-lang",
        choices=["auto", "eng", "jpn"],
        default="auto",
        help="Fix the spoken language for the WHOLE file: 'eng' = translate every segment EN->JA, 'jpn' = every segment JA->EN, 'auto' (default) = per-segment LID. Use eng/jpn for monolingual audio to avoid LID direction flips (and skip the LID pass entirely = faster).",
    )
    ap.add_argument(
        "--lid",
        choices=["selfscore", "alternate"],
        default="selfscore",
        help="Per-segment language-ID method when --source-lang auto.",
    )
    ap.add_argument(
        "--first-lang", choices=["jpn", "eng"], default="jpn", help="First turn's language for --lid alternate."
    )
    ap.add_argument(
        "--lid-floor",
        type=float,
        default=-1.6,
        help="Flag utterances whose best LID score is below this (likely noise/boundary fragments).",
    )
    ap.add_argument(
        "--lid-seconds",
        type=float,
        default=2.5,
        help="Run self-score LID on this many leading seconds of each utterance (short audio = reliable LID + no encoder crash); full segment still translated. <=0 uses the whole segment.",
    )
    ap.add_argument(
        "--no-transcribe", action="store_true", help="Skip source-language transcription (translation only)."
    )
    ap.add_argument("--top-db", type=float, default=30.0, help="VAD silence threshold (dB below peak).")
    ap.add_argument("--min-dur", type=float, default=0.4, help="Drop utterances shorter than this (s).")
    ap.add_argument("--merge-gap", type=float, default=0.3, help="Merge utterances separated by less than this (s).")
    ap.add_argument(
        "--max-dur",
        type=float,
        default=0.0,
        help="Split utterances longer than this into equal windows (s); <=0 disables.",
    )
    ap.add_argument(
        "--encoder-min-frames",
        type=int,
        default=0,
        help="Pad each segment's mel frames up to this to force the encoder's long-audio path. Only needed as a fallback if the encoder's _LONG_AUDIO_RES_DRAM_THRESHOLD fix (1-C) is NOT present; with that fix the mid-length crash is gone, so 0 (no padding) is the default.",
    )
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Greedy repetition penalty (>=1.0; >1 suppresses degenerate loops).",
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=64, help="Max decoded tokens per utterance (caps runaway loops)."
    )
    ap.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search width (1=greedy). >1 (e.g. 5) reduces degenerate loops/hallucinations and improves fluency, at ~N x decode cost. Uses the beam KV path (no decode trace).",
    )
    ap.add_argument(
        "--keep-programs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="EXPERIMENTAL/UNSAFE: keep JIT/prep caches warm across segments (~1.45x faster) but it CORRUPTS results — the per-segment clear is load-bearing for correctness here (cross-segment device state leaks into the low-level LID + translate, changing LID scores/outputs). Default off (safe).",
    )
    ap.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Start processing at this offset (s); combine with --max-seconds to pick a window, or alone to process from here to the end (e.g. the last few minutes). Reported timestamps stay relative to the original file.",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Process at most N seconds from --start-seconds; <=0 = to end of file.",
    )
    ap.add_argument("--make-test-audio", type=Path, metavar="OUT", help="Generate a JP+EN test WAV and exit.")
    args = ap.parse_args()

    if not args.make_test_audio and not args.audio and not args.youtube:
        ap.error("provide --audio FILE, --youtube URL, or --make-test-audio OUT")

    # YouTube mode: fetch the audio track up-front (fail fast before loading the model).
    if args.youtube:
        out_parent = args.out.parent if args.out else Path("outputs")
        args.audio = _download_youtube_audio(args.youtube, out_parent)

    weights_dir = _weights_dir()
    path = os.fspath(weights_dir)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config

    gen_common = hf_aligned_generation_kwargs(
        model.generation_config, use_kv_cache=True, use_decode_trace=True, use_2cq=True
    )
    # Checkpoint defaults are repetition_penalty=1.0 (none) + max_new_tokens=256, which let greedy
    # decode degenerate into loops ("a little bit of a little bit of ...") on short/fragmentary audio.
    gen_common["repetition_penalty"] = max(1.0, float(args.repetition_penalty))
    gen_common["max_new_tokens"] = int(args.max_new_tokens)
    gen_common["num_beams"] = max(1, int(args.num_beams))
    eos_ids = {int(gen_common["eos_token_id"])}

    # ---- segment input audio up-front (so test-audio mode skips this) ----
    segments: List[Tuple[int, int]] = []
    y: Optional[np.ndarray] = None
    ts_offset = 0.0
    if args.audio:
        y = _load_audio_16k_mono(args.audio)
        start = int(args.start_seconds * SAMPLE_RATE) if args.start_seconds and args.start_seconds > 0 else 0
        start = min(start, y.size)
        if args.max_seconds and args.max_seconds > 0:
            y = y[start : start + int(args.max_seconds * SAMPLE_RATE)]
        else:
            y = y[start:]
        ts_offset = start / SAMPLE_RATE  # report timestamps relative to the original file
        segments = segment_audio(
            y, top_db=args.top_db, min_dur=args.min_dur, merge_gap=args.merge_gap, max_dur=args.max_dur
        )
        print(
            f"  Input: {args.audio}  (window {_fmt_ts(ts_offset)}-{_fmt_ts(ts_offset + y.size / SAMPLE_RATE)}, "
            f"{y.size / SAMPLE_RATE:.2f}s)  ->  {len(segments)} utterance(s)"
        )
        if not segments:
            raise SystemExit("No speech segments found — try lowering --top-db.")

    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        original_default = None
    device, mesh_shape = _open_device()
    ttnn.SetDefaultDevice(device)
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    print(f"  Device: MeshShape({rows}, {cols}) — TP={rows * cols}")

    gc = model.generation_config
    out_lines: List[str] = []
    tt_model = None
    try:
        tt_model = make_tt_model(device, model, cfg, t2u_cfg)

        if args.make_test_audio:
            make_test_audio(tt_model, processor, cfg, args.make_test_audio, gen_common)
            return

        if args.keep_programs:
            # The per-segment cost is dominated by rebuilding the conv1d/matmul prep caches and
            # re-capturing the decode trace, NOT by kernel (re)compiles. So replace the full
            # clear_runtime_program_cache with one that ONLY releases the decode trace — required
            # for correctness (a stale active trace corrupts the LID's low-level decode buffers) —
            # while KEEPING the device program cache and the conv1d/matmul prep caches warm across
            # same-length segments (the _encode_speech 256-frame buckets). S2TT-only never builds the
            # vocoder/T2U programs the full clear exists to evict, so this is safe here.
            tt_model.clear_runtime_program_cache = (  # type: ignore[assignment]
                lambda *a, **k: tt_model.release_text_decoder_decode_trace()
            )

        print()
        print("=" * 78)
        print("  Bilingual S2TT  (JP <-> EN)")
        print("=" * 78)

        t_loop0 = time.perf_counter()
        total_model_s = 0.0
        speech_s = 0.0

        for i, (s, e) in enumerate(segments):
            seg = y[s:e]
            t0, t1 = s / SAMPLE_RATE, e / SAMPLE_RATE
            seg_t0 = time.perf_counter()
            # Full-segment features for translation, padded onto the encoder's proven long-audio
            # path (avoids the mid-length "Tensor is not allocated" crash; pad frames masked out).
            ai = processor(audios=seg, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            feats, sattn = _pad_mel_to_long_path(ai["input_features"], ai["attention_mask"], args.encoder_min_frames)

            # ---- decide direction ----
            if args.source_lang in ("eng", "jpn"):
                # Whole-file fixed direction: no LID pass at all (faster, no direction flips).
                src = args.source_lang
                scores = None
            elif args.lid == "alternate":
                src = args.first_lang if i % 2 == 0 else ("eng" if args.first_lang == "jpn" else "jpn")
                scores = None
            else:
                # Self-score LID on a SHORT leading window only: short audio uses the encoder's
                # reliable short path (no crash) AND the "transcription is more confident than
                # translation" gap holds there — it inverts on long/complex whole turns. The full
                # segment is still translated below.
                seg_lid = (
                    seg[: int(args.lid_seconds * SAMPLE_RATE)] if args.lid_seconds and args.lid_seconds > 0 else seg
                )
                al = processor(audios=seg_lid, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                feats_l, sattn_l = al["input_features"], al["attention_mask"]  # unpadded (short path)
                tt_model.prewarm_speech_encoder([int(feats_l.shape[1])])
                tt_model.clear_runtime_program_cache()
                ttnn.synchronize_device(device)
                feats_tt = torch_feats_to_ttnn(device, feats_l)
                attn_tt = torch_ids_to_ttnn(device, sattn_l)
                src, scores = detect_source_lang(tt_model, gc, feats_tt, attn_tt, eos_ids)
                ttnn.deallocate(feats_tt)
                ttnn.deallocate(attn_tt)
            tgt = "eng" if src == "jpn" else "jpn"

            # ---- translate (and optionally transcribe) the FULL segment with the fast path ----
            tt_model.prewarm_speech_encoder([int(feats.shape[1])])
            tt_model.clear_runtime_program_cache()
            ttnn.synchronize_device(device)
            feats_tt = torch_feats_to_ttnn(device, feats)
            attn_tt = torch_ids_to_ttnn(device, sattn)
            translation = translate_to(tt_model, tokenizer, feats_tt, attn_tt, tgt, gen_common)
            transcript = None
            if not args.no_transcribe:
                feats_tt2 = torch_feats_to_ttnn(device, feats)
                attn_tt2 = torch_ids_to_ttnn(device, sattn)
                transcript = translate_to(tt_model, tokenizer, feats_tt2, attn_tt2, src, gen_common)

            seg_model_s = time.perf_counter() - seg_t0
            total_model_s += seg_model_s
            speech_s += t1 - t0

            # ---- report ----
            tag = f"{LANG_SHORT[src]}->{LANG_SHORT[tgt]}"
            ts = f"[{_fmt_ts(t0 + ts_offset)}-{_fmt_ts(t1 + ts_offset)}]"
            lid_note = ""
            low_conf = False
            if scores is not None:
                low_conf = max(scores.values()) < args.lid_floor
                lid_note = f"  (LID jpn={scores['jpn']:.3f} eng={scores['eng']:.3f})"
            flag = "  [low-confidence: likely noise/boundary]" if low_conf else ""
            header = f"{ts} {tag}  {seg_model_s:.1f}s{lid_note}{flag}"
            print(f"\n[{i + 1}/{len(segments)}] {header}")
            if transcript is not None:
                print(f"  src ({src}): {transcript}")
            print(f"  out ({tgt}): {translation}")

            out_lines.append(header)
            if transcript is not None:
                out_lines.append(f"  src ({src}): {transcript}")
            out_lines.append(f"  out ({tgt}): {translation}")
            out_lines.append("")

        wall_s = time.perf_counter() - t_loop0
        n_seg = len(segments)
        per_seg = total_model_s / n_seg if n_seg else 0.0
        rtf = total_model_s / speech_s if speech_s > 0 else 0.0
        summary = [
            "-" * 78,
            "  TIMING (model processing; excludes model build / weight load)",
            "-" * 78,
            f"  segments              : {n_seg}",
            f"  speech processed      : {speech_s:.1f}s  (input file {y.size / SAMPLE_RATE:.1f}s)",
            f"  total model time      : {total_model_s:.1f}s",
            f"  wall-clock (loop)     : {wall_s:.1f}s",
            f"  per-segment (avg)     : {per_seg:.2f}s",
            f"  real-time factor      : {rtf:.2f}x  (model-time / speech-time; <1 = faster than real time)",
            "-" * 78,
        ]
        for line in summary:
            print(line)

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text("\n".join(out_lines + summary), encoding="utf-8")
            print(f"\n  Saved transcript + timing -> {args.out}")

    finally:
        if tt_model is not None:
            try:
                tt_model.release_generation_runtime()
            except Exception:
                pass
        if original_default is not None:
            ttnn.SetDefaultDevice(original_default)
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
