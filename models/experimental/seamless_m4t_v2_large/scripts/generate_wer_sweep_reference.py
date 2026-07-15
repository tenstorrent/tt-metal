# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Offline HF references for Seamless M4T v2 E2E WER sweep tests (T2ST, S2ST).

Runs HuggingFace ``generate(generate_speech=True)`` once per sweep point and saves the
decoded intermediate translation text (``sequences``) for ``jiwer.wer`` comparison against TT.

Usage::

    python models/experimental/seamless_m4t_v2_large/scripts/generate_wer_sweep_reference.py --task t2st --seq_len 128
    python models/experimental/seamless_m4t_v2_large/scripts/generate_wer_sweep_reference.py --sweep --task all
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.demo_perf_sweep import (
    S2ST_PREAMBLE_MAX_MEL,
    SEQ_LEN_MAX,
    SRC_LANG,
    ensure_long_audio,
    ensure_long_story,
    sequence_lengths,
    speech_inputs_for_len,
    text_inputs_for_wer_len,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    TEXT_INPUT_TASKS,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_wer_helpers import (
    WER_SWEEP_TASKS,
    WER_TASK_TGT_LANG,
    transcribe_waveform_whisper,
    wer_refpt_path,
    whisper_language_for_tgt,
)
from models.experimental.seamless_m4t_v2_large.tt.common import hf_aligned_generation_kwargs


@contextmanager
def _capture_hf_vocoder_input(model):
    """Capture the exact unit ids HF feeds to its vocoder during ``generate(generate_speech=True)``.

    These are HF's ``unit_ids`` right before ``self.vocoder(input_ids=...)`` — eos/pad already mapped
    to pad_id and real units shifted by ``-vocoder_offset``. Feeding this same tensor to the TT vocoder
    teacher-forces the T2U (text→units), isolating vocoder fidelity from the free-running T2U divergence.
    """
    captured: dict = {}

    def _pre_hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None:
            captured["input_ids"] = ids.detach().cpu()

    handle = model.vocoder.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        yield captured
    finally:
        handle.remove()


def _hf_reference_text(model, tokenizer, *, out) -> str:
    if hasattr(out, "sequences"):
        ids = out.sequences
    else:
        ids = out[0] if isinstance(out, (tuple, list)) else out
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def _hf_waveform_whisper_text(model, out, tgt_lang: str) -> str:
    """Whisper transcription of HF's generated waveform (empty if no waveform, e.g. ASR text output)."""
    wav = getattr(out, "waveform", None)
    if wav is None:
        return ""
    arr = wav.detach().float().reshape(-1).cpu().numpy()
    lengths = getattr(out, "waveform_lengths", None)
    if lengths is not None:
        n = int(torch.as_tensor(lengths).reshape(-1)[0].item())
        if 0 < n <= arr.size:
            arr = arr[:n]
    sr = int(getattr(model.config, "sampling_rate", 16000))
    return transcribe_waveform_whisper(arr, sr, whisper_language_for_tgt(tgt_lang))


def generate_text_wer_sweep_reference(
    *,
    weights_dir: str,
    output_file: Path,
    task: str,
    seq_len: int,
    src_lang: str = SRC_LANG,
    story: str | None = None,
) -> None:
    """HF speech-generate reference for T2ST at exactly ``seq_len`` source tokens."""
    if task != "t2st":
        raise ValueError(f"generate_text_wer_sweep_reference expects t2st, got {task!r}")
    model, processor, tokenizer = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
    story_text = story if story is not None else ensure_long_story()
    # Long T2ST: mixed story windows (not one contiguous Dickens span) so HF Spanish speech stays
    # non-degenerate for the whisper round-trip — see ``text_inputs_for_wer_len``.
    src_ids, src_mask = text_inputs_for_wer_len(processor, story_text, seq_len, src_lang=src_lang)
    tgt_lang = WER_TASK_TGT_LANG[task]
    gen_kwargs = hf_aligned_generation_kwargs(model.generation_config)

    with torch.no_grad(), _capture_hf_vocoder_input(model) as voc_cap:
        out = model.generate(
            input_ids=src_ids,
            attention_mask=src_mask,
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang=tgt_lang,
            speaker_id=0,
            **gen_kwargs,
        )
    ref_text = _hf_reference_text(model, tokenizer, out=out)
    hf_whisper_text = _hf_waveform_whisper_text(model, out, tgt_lang)
    # HF intermediate translation tokens — fed to the TT speech pipeline for the TEACHER-FORCED
    # whisper round-trip (isolates T2U+vocoder from the text-decode cascade).
    hf_intermediate_ids = out.sequences.detach().cpu() if hasattr(out, "sequences") else None
    # HF's exact (offset-applied) vocoder input — fed straight to the TT vocoder to ALSO teacher-force
    # the T2U, leaving a pure vocoder-fidelity whisper metric.
    hf_vocoder_input_ids = voc_cap.get("input_ids")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "seq_len": seq_len,
        "tgt_lang": tgt_lang,
        "src_lang": src_lang,
        "src_ids": src_ids.cpu(),
        "src_mask": src_mask.cpu(),
        "reference_text": ref_text,
        "hf_whisper_text": hf_whisper_text,
        "hf_intermediate_ids": hf_intermediate_ids,
        "hf_vocoder_input_ids": hf_vocoder_input_ids,
    }
    torch.save(payload, output_file)
    logger.info(
        f"Saved {task.upper()} WER reference len={seq_len} ({len(ref_text.split())} words; "
        f"whisper {len(hf_whisper_text.split())} words) to {output_file}"
    )


def generate_speech_wer_sweep_reference(
    *,
    weights_dir: str,
    output_file: Path,
    task: str,
    mel_frames: int,
) -> None:
    """HF reference for a speech-INPUT WER task at ``mel_frames`` mel frames.

    ``s2st`` generates speech (WER on the intermediate translation text); ``asr`` generates text
    directly (WER on the English transcription). Both consume the same length-dependent audio.
    """
    if task not in ("s2st", "asr"):
        raise ValueError(f"generate_speech_wer_sweep_reference expects s2st/asr, got {task!r}")
    model, processor, tokenizer = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
    sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16_000))
    # Short mel: preamble (its truncated <1 s opening translates cleanly). Longer mel: LibriSpeech
    # (coherent, non-repeating; the repeated preamble produced short degenerate references).
    if mel_frames <= S2ST_PREAMBLE_MAX_MEL:
        wav, _ = ensure_long_audio(
            processor, sample_rate, min_mel_frames=max(mel_frames, S2ST_PREAMBLE_MAX_MEL), source="preamble"
        )
    else:
        wav, _ = ensure_long_audio(
            processor, sample_rate, min_mel_frames=max(mel_frames, SEQ_LEN_MAX), source="librispeech"
        )
    audio = processor(audio=wav, sampling_rate=sample_rate, return_tensors="pt")
    input_features, mel_mask = speech_inputs_for_len(
        processor,
        audio["input_features"],
        audio["attention_mask"],
        mel_frames,
    )
    input_features = input_features.to(dtype=next(model.parameters()).dtype)
    tgt_lang = WER_TASK_TGT_LANG[task]
    gen_kwargs = hf_aligned_generation_kwargs(model.generation_config)

    # s2st → speech output (compare the intermediate translation); asr → text output (transcription).
    if task == "s2st":
        gen_extra = dict(generate_speech=True, return_intermediate_token_ids=True, speaker_id=0)
    else:  # asr
        gen_extra = dict(generate_speech=False)
    with torch.no_grad(), _capture_hf_vocoder_input(model) as voc_cap:
        out = model.generate(
            input_features=input_features.float(),
            attention_mask=mel_mask,
            tgt_lang=tgt_lang,
            **gen_extra,
            **gen_kwargs,
        )
    ref_text = _hf_reference_text(model, tokenizer, out=out)
    hf_whisper_text = _hf_waveform_whisper_text(model, out, tgt_lang)  # "" for asr (no waveform)
    # HF intermediate translation tokens + exact vocoder input for the teacher-forced whisper round-trip
    # (s2st only; asr generate_speech=False has no speech pipeline / vocoder to teacher-force).
    is_speech = task == "s2st"
    hf_intermediate_ids = out.sequences.detach().cpu() if (is_speech and hasattr(out, "sequences")) else None
    hf_vocoder_input_ids = voc_cap.get("input_ids") if is_speech else None

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "seq_len": mel_frames,
        "tgt_lang": tgt_lang,
        "input_features": input_features.cpu(),
        "mel_attention_mask": mel_mask.cpu(),
        "reference_text": ref_text,
        "hf_whisper_text": hf_whisper_text,
        "hf_intermediate_ids": hf_intermediate_ids,
        "hf_vocoder_input_ids": hf_vocoder_input_ids,
    }
    torch.save(payload, output_file)
    logger.info(
        f"Saved {task.upper()} WER reference mel={mel_frames} " f"({len(ref_text.split())} words) to {output_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Seamless M4T v2 WER sweep .refpt files")
    parser.add_argument(
        "--task",
        choices=["t2st", "s2st", "asr", "all"],
        default="t2st",
        help="Speech-output task reference to generate (default: t2st)",
    )
    parser.add_argument("--output_file", type=Path, default=None, help="Override output .refpt path")
    parser.add_argument("--weights_dir", type=str, default=None)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Generate sequence-length sweep references (32→4096, same inputs as demo_perf_sweep.py)",
    )
    parser.add_argument("--seq_len", type=int, default=None, help="Single sweep length (requires --sweep)")
    parser.add_argument("--min_len", type=int, default=32, help="Sweep minimum sequence length (default: 32)")
    parser.add_argument(
        "--max_len", type=int, default=SEQ_LEN_MAX, help="Sweep maximum sequence length (default: 4096)"
    )
    args = parser.parse_args()
    weights_dir = args.weights_dir or ensure_seamless_m4t_v2_large_weights()

    if args.sweep:
        if args.seq_len is not None:
            lengths = [args.seq_len]
        else:
            lengths = sequence_lengths(args.min_len, args.max_len)
        tasks = list(WER_SWEEP_TASKS) if args.task == "all" else [args.task]
        for task in tasks:
            for seq_len in lengths:
                out = (
                    args.output_file
                    if args.output_file and len(tasks) == 1 and len(lengths) == 1
                    else wer_refpt_path(task, seq_len)
                )
                if task in TEXT_INPUT_TASKS:
                    generate_text_wer_sweep_reference(
                        weights_dir=weights_dir,
                        output_file=out,
                        task=task,
                        seq_len=seq_len,
                    )
                else:
                    generate_speech_wer_sweep_reference(
                        weights_dir=weights_dir,
                        output_file=out,
                        task=task,
                        mel_frames=seq_len,
                    )
        return

    task = "t2st" if args.task == "all" else args.task
    seq_len = args.seq_len if args.seq_len is not None else 128
    out = args.output_file or wer_refpt_path(task, seq_len)
    if task == "t2st":
        generate_text_wer_sweep_reference(
            weights_dir=weights_dir,
            output_file=out,
            task=task,
            seq_len=seq_len,
        )
    else:
        generate_speech_wer_sweep_reference(
            weights_dir=weights_dir,
            output_file=out,
            task=task,
            mel_frames=seq_len,
        )


if __name__ == "__main__":
    main()
