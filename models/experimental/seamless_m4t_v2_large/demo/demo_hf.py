# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Seamless M4T v2 — Hugging Face (PyTorch) demo of all five inference tasks.

Mirror of ``demo.py`` but runs everything through the upstream
[`SeamlessM4Tv2Model.generate`] instead of the TT model. Useful as a reference for
sanity-checking TTNN outputs against the HF model on the same input prompt.

Tasks demonstrated (every one runs through ``model.generate``):

  1. **T2TT** Text-to-Text Translation        (English text → Hindi text)
  2. **T2ST** Text-to-Speech Translation      (English text → Hindi speech)
  3. **S2TT** Speech-to-Text Translation      (Hindi speech → English text)
  4. **S2ST** Speech-to-Speech Translation    (Hindi speech → Spanish speech)
  5. **ASR**  Automatic Speech Recognition    (Hindi speech → Hindi text)

The Hindi speech used as the input to tasks 3–5 is produced by task 2. Output audio is
written next to this file under ``outputs/``:

  * ``outputs/t2st_hindi_speech_hf.wav``    — task 2 output (re-used as input for tasks 3-5)
  * ``outputs/s2st_spanish_speech_hf.wav``  — task 4 output

Run from repo root:

  python models/experimental/seamless_m4t_v2_large/demo/demo_hf.py

Optional: ``SEAMLESS_M4T_V2_WEIGHTS=/path/to/seamless-m4t-v2-large`` if not using the default tree.
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2GenerationOutput

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
T2ST_WAV = OUTPUT_DIR / "t2st_hindi_speech_hf.wav"
S2ST_WAV = OUTPUT_DIR / "s2st_spanish_speech_hf.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weights_dir() -> Path:
    env = os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return ensure_seamless_m4t_v2_large_weights()


def _waveform_to_mono_fp32(waveform: torch.Tensor, lengths: Optional[torch.Tensor]) -> np.ndarray:
    """HF vocoder waveform → 1-D fp32 numpy array, trimmed to valid length.

    HF returns ``[B, T_max]`` (right-padded with zeros to the batch max) and per-row valid lengths.
    """
    arr = waveform.detach().float().squeeze().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if lengths is not None:
        valid_len = int(lengths.detach().long().reshape(-1)[0].item())
        if 0 < valid_len <= arr.size:
            arr = arr[:valid_len]
    return arr


def _save_wav(path: Path, waveform_np: np.ndarray, sample_rate: int) -> None:
    """Save a mono fp32 waveform to ``path`` as a 16-bit PCM WAV (stdlib ``wave`` only)."""
    arr = np.clip(waveform_np, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def _decode(tokenizer: Any, sequences: torch.Tensor) -> str:
    ids = sequences.detach().to(torch.int64).cpu()
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def _valid_unit_frames(unit_sequences: torch.Tensor, *, pad_id: int) -> int:
    u = unit_sequences.detach().long().reshape(-1)
    return int((u != int(pad_id)).sum().item())


def _row_length(sequences: torch.Tensor) -> int:
    return int(sequences.detach().long().reshape(-1).numel())


def _print_header(idx: int, name: str, abbrev: str, src: str, tgt: str) -> None:
    print()
    print("=" * 78)
    print(f"  {idx}. {abbrev}  —  {name}  ({src} → {tgt})")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    weights_dir = _weights_dir()
    path = os.fspath(weights_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    sample_rate = int(getattr(cfg, "sampling_rate", 16000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  HF demo device: {device}")

    # ---- Single English prompt drives the entire demo chain ----
    src_text = """Maya lived in a small coastal town where every morning began with the sound of fishing boats leaving the harbor. She worked at her grandfather’s old bookstore, a narrow shop filled with dusty shelves, handwritten notes, and the smell of paper that had aged for decades. Most customers came looking for schoolbooks or travel guides, but Maya loved recommending forgotten stories hidden in the back corners of the store.

One rainy evening, while organizing a stack of returned books, she discovered a small blue journal tucked between two novels. The cover had no title, only a silver compass symbol that shimmered faintly under the light. Curious, she opened it and found detailed sketches of places around the town along with cryptic messages about a hidden lighthouse path that only appeared during storms.

At first, Maya thought someone was playing a prank. But the next night, as heavy clouds gathered over the sea, she noticed something unusual from the bookstore window. A narrow trail of lantern lights stretched along the cliffs where no road existed before. Holding the journal tightly, she followed the glowing path through the rain until she reached an abandoned lighthouse overlooking the crashing waves.

Inside the lighthouse, she found old maps, letters, and photographs belonging to sailors who had once protected ships during dangerous storms. Among the papers was a letter written by her grandfather many years earlier. He explained that the journal was meant for the next person curious enough to search for the truth hidden in ordinary places. Maya smiled as thunder echoed outside. For the first time, she realized the bookstore had never only been about selling books. It had always been about discovering stories waiting quietly for someone brave enough to follow them.
"""
    src_lang = "eng"
    tgt_translate = "hin"  # task 1, 2: translate eng → hin
    tgt_back_text = "eng"  # task 3: speech in hin → text in eng (back-translation)
    tgt_speech_other = "spa"  # task 4: speech in hin → speech in spa
    tgt_asr = "hin"  # task 5: transcribe the hin audio in hin

    # SeamlessM4T v2 emits EOS at sentence/short-utterance boundaries (training distribution),
    # so multi-paragraph inputs get truncated regardless of ``max_new_tokens``. Chunk the source
    # by paragraph and join outputs. Same pattern as ``demo.py``.
    paragraphs = [p.strip() for p in src_text.split("\n\n") if p.strip()]

    gen_max_new = int(
        getattr(cfg, "max_new_tokens", None) or getattr(model.generation_config, "max_new_tokens", 128) or 128
    )
    rep_penalty = float(getattr(model.generation_config, "repetition_penalty", 1.0) or 1.0)
    if rep_penalty == 1.0:
        rep_penalty = 1.1
    gen_common = dict(
        max_new_tokens=gen_max_new,
        do_sample=False,
        num_beams=1,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=cfg.eos_token_id,
        repetition_penalty=rep_penalty,
    )

    # =========================================================================
    # 1. T2TT — Text-to-Text Translation (English → Hindi)
    # =========================================================================
    _print_header(1, "Text-to-Text Translation", "T2TT", "eng", tgt_translate)
    print(f"  Input text  ({src_lang}): {src_text}")
    t2tt_chunks: list[str] = []
    for para in paragraphs:
        para_inputs = processor(text=para, src_lang=src_lang, return_tensors="pt")
        with torch.no_grad():
            t2tt_seq = model.generate(
                input_ids=para_inputs["input_ids"].to(device),
                attention_mask=para_inputs["attention_mask"].to(device),
                generate_speech=False,
                tgt_lang=tgt_translate,
                **gen_common,
            )
        # HF returns either a Tensor or a ModelOutput depending on settings; normalize.
        t2tt_sequences = t2tt_seq if isinstance(t2tt_seq, torch.Tensor) else t2tt_seq.sequences
        t2tt_chunks.append(_decode(tokenizer, t2tt_sequences))
    t2tt_joined = "\n\n".join(t2tt_chunks)
    print(f"  Output text ({tgt_translate}): {t2tt_joined}")

    # =========================================================================
    # 2. T2ST — Text-to-Speech Translation (English text → Hindi speech)
    # =========================================================================
    _print_header(2, "Text-to-Speech Translation", "T2ST", "eng", tgt_translate)
    print(f"  Input text  ({src_lang}): {src_text}")
    t2u_pad = int(t2u_cfg.pad_token_id)
    # 0.4s of silence between paragraphs for natural pacing in the joined audio.
    inter_para_silence = np.zeros(int(sample_rate * 0.4), dtype=np.float32)
    t2st_texts: list[str] = []
    chunk_wavs: list[np.ndarray] = []
    per_para_wavs: list[np.ndarray] = []  # paragraph audio (no silence) — reused by tasks 3-5
    total_text_tokens = 0
    total_unit_frames = 0
    for idx, para in enumerate(paragraphs):
        para_inputs = processor(text=para, src_lang=src_lang, return_tensors="pt")
        with torch.no_grad():
            t2st_out = model.generate(
                input_ids=para_inputs["input_ids"].to(device),
                attention_mask=para_inputs["attention_mask"].to(device),
                generate_speech=True,
                return_intermediate_token_ids=True,
                tgt_lang=tgt_translate,
                speaker_id=0,
                **gen_common,
            )
        if not isinstance(t2st_out, SeamlessM4Tv2GenerationOutput):
            raise TypeError(f"T2ST expected SeamlessM4Tv2GenerationOutput, got {type(t2st_out)}")
        t2st_texts.append(_decode(tokenizer, t2st_out.sequences))
        para_wav = _waveform_to_mono_fp32(t2st_out.waveform, t2st_out.waveform_lengths)
        chunk_wavs.append(para_wav)
        per_para_wavs.append(para_wav)
        if idx < len(paragraphs) - 1:
            chunk_wavs.append(inter_para_silence)
        total_text_tokens += _row_length(t2st_out.sequences)
        total_unit_frames += _valid_unit_frames(t2st_out.unit_sequences, pad_id=t2u_pad)
    t2st_text_joined = "\n\n".join(t2st_texts)
    print(f"  Intermediate text ({tgt_translate}): {t2st_text_joined}")
    hindi_wav_np = np.concatenate(chunk_wavs) if chunk_wavs else np.zeros(0, dtype=np.float32)
    _save_wav(T2ST_WAV, hindi_wav_np, sample_rate=sample_rate)
    print(
        f"  T2ST stats: text_tokens={total_text_tokens}, "
        f"unit_frames={total_unit_frames}, "
        f"audio={hindi_wav_np.size} samples ({hindi_wav_np.size / sample_rate:.2f}s)"
    )
    print(f"  Saved to: {T2ST_WAV}")

    # Tasks 3-5 process the T2ST output per paragraph (same EOS truncation applies on the
    # decoder side regardless of how much audio is fed in).
    def _audio_inputs_for_chain(wav: np.ndarray):
        a = processor(audios=wav, sampling_rate=sample_rate, return_tensors="pt")
        return (
            a["input_features"].to(device=device, dtype=torch.bfloat16),
            a["attention_mask"].to(device),
        )

    # =========================================================================
    # 3. S2TT — Speech-to-Text Translation (Hindi speech → English text)
    # =========================================================================
    _print_header(3, "Speech-to-Text Translation", "S2TT", tgt_translate, tgt_back_text)
    print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
    s2tt_chunks: list[str] = []
    for para_wav in per_para_wavs:
        in_feats, in_attn = _audio_inputs_for_chain(para_wav)
        with torch.no_grad():
            s2tt_seq = model.generate(
                input_features=in_feats,
                attention_mask=in_attn,
                generate_speech=False,
                tgt_lang=tgt_back_text,
                **gen_common,
            )
        s2tt_sequences = s2tt_seq if isinstance(s2tt_seq, torch.Tensor) else s2tt_seq.sequences
        s2tt_chunks.append(_decode(tokenizer, s2tt_sequences))
    s2tt_joined = "\n\n".join(s2tt_chunks)
    print(f"  Output text ({tgt_back_text}): {s2tt_joined}")

    # =========================================================================
    # 4. S2ST — Speech-to-Speech Translation (Hindi speech → Spanish speech)
    # =========================================================================
    _print_header(4, "Speech-to-Speech Translation", "S2ST", tgt_translate, tgt_speech_other)
    print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
    s2st_texts: list[str] = []
    s2st_wavs: list[np.ndarray] = []
    for idx, para_wav in enumerate(per_para_wavs):
        in_feats, in_attn = _audio_inputs_for_chain(para_wav)
        with torch.no_grad():
            s2st_out = model.generate(
                input_features=in_feats,
                attention_mask=in_attn,
                generate_speech=True,
                return_intermediate_token_ids=True,
                tgt_lang=tgt_speech_other,
                speaker_id=0,
                **gen_common,
            )
        if not isinstance(s2st_out, SeamlessM4Tv2GenerationOutput):
            raise TypeError(f"S2ST expected SeamlessM4Tv2GenerationOutput, got {type(s2st_out)}")
        s2st_texts.append(_decode(tokenizer, s2st_out.sequences))
        s2st_wavs.append(_waveform_to_mono_fp32(s2st_out.waveform, s2st_out.waveform_lengths))
        if idx < len(per_para_wavs) - 1:
            s2st_wavs.append(inter_para_silence)
    s2st_text_joined = "\n\n".join(s2st_texts)
    print(f"  Intermediate text ({tgt_speech_other}): {s2st_text_joined}")
    spanish_wav_np = np.concatenate(s2st_wavs) if s2st_wavs else np.zeros(0, dtype=np.float32)
    _save_wav(S2ST_WAV, spanish_wav_np, sample_rate=sample_rate)
    print(f"  Output audio ({tgt_speech_other}, {sample_rate} Hz, {spanish_wav_np.size} samples)")
    print(f"  Saved to: {S2ST_WAV}")

    # =========================================================================
    # 5. ASR — Automatic Speech Recognition (Hindi speech → Hindi text)
    # =========================================================================
    # Same-language transcription — disable repetition penalty (matches demo.py).
    gen_common_asr = {**gen_common, "repetition_penalty": 1.0}
    _print_header(5, "Automatic Speech Recognition", "ASR", tgt_translate, tgt_asr)
    print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
    asr_chunks: list[str] = []
    for para_wav in per_para_wavs:
        in_feats, in_attn = _audio_inputs_for_chain(para_wav)
        with torch.no_grad():
            asr_seq = model.generate(
                input_features=in_feats,
                attention_mask=in_attn,
                generate_speech=False,
                tgt_lang=tgt_asr,
                **gen_common_asr,
            )
        asr_sequences = asr_seq if isinstance(asr_seq, torch.Tensor) else asr_seq.sequences
        asr_chunks.append(_decode(tokenizer, asr_sequences))
    asr_joined = "\n\n".join(asr_chunks)
    print(f"  Output text ({tgt_asr}): {asr_joined}")

    print()
    print("=" * 78)
    print("  ok — all five tasks completed (HF)")
    print("=" * 78)
    print(f"  Audio outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
