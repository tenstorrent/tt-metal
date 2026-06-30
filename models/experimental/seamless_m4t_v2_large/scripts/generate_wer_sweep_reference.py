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
from pathlib import Path

import torch
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.demo_perf_sweep import (
    SEQ_LEN_MAX,
    SRC_LANG,
    ensure_long_audio,
    ensure_long_story,
    sequence_lengths,
    speech_inputs_for_len,
    text_inputs_for_len,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    SPEECH_OUTPUT_TASKS,
    TASK_TGT_LANG,
    TEXT_INPUT_TASKS,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_wer_helpers import wer_refpt_path
from models.experimental.seamless_m4t_v2_large.tt.common import hf_aligned_generation_kwargs


def _hf_reference_text(model, tokenizer, *, out) -> str:
    if hasattr(out, "sequences"):
        ids = out.sequences
    else:
        ids = out[0] if isinstance(out, (tuple, list)) else out
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


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
    src_ids, src_mask = text_inputs_for_len(processor, story_text, seq_len, src_lang=src_lang)
    tgt_lang = TASK_TGT_LANG[task]
    gen_kwargs = hf_aligned_generation_kwargs(model.generation_config)

    with torch.no_grad():
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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "seq_len": seq_len,
        "tgt_lang": tgt_lang,
        "src_lang": src_lang,
        "src_ids": src_ids.cpu(),
        "src_mask": src_mask.cpu(),
        "reference_text": ref_text,
    }
    torch.save(payload, output_file)
    logger.info(
        f"Saved {task.upper()} WER reference len={seq_len} " f"({len(ref_text.split())} words) to {output_file}"
    )


def generate_speech_wer_sweep_reference(
    *,
    weights_dir: str,
    output_file: Path,
    task: str,
    mel_frames: int,
) -> None:
    """HF speech-generate reference for S2ST at ``mel_frames`` mel frames."""
    if task != "s2st":
        raise ValueError(f"generate_speech_wer_sweep_reference expects s2st, got {task!r}")
    model, processor, tokenizer = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
    sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16_000))
    wav, _ = ensure_long_audio(processor, sample_rate, min_mel_frames=max(mel_frames, SEQ_LEN_MAX))
    audio = processor(audio=wav, sampling_rate=sample_rate, return_tensors="pt")
    input_features, mel_mask = speech_inputs_for_len(
        processor,
        audio["input_features"],
        audio["attention_mask"],
        mel_frames,
    )
    input_features = input_features.to(dtype=next(model.parameters()).dtype)
    tgt_lang = TASK_TGT_LANG[task]
    gen_kwargs = hf_aligned_generation_kwargs(model.generation_config)

    with torch.no_grad():
        out = model.generate(
            input_features=input_features.float(),
            attention_mask=mel_mask,
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang=tgt_lang,
            speaker_id=0,
            **gen_kwargs,
        )
    ref_text = _hf_reference_text(model, tokenizer, out=out)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "seq_len": mel_frames,
        "tgt_lang": tgt_lang,
        "input_features": input_features.cpu(),
        "mel_attention_mask": mel_mask.cpu(),
        "reference_text": ref_text,
    }
    torch.save(payload, output_file)
    logger.info(
        f"Saved {task.upper()} WER reference mel={mel_frames} " f"({len(ref_text.split())} words) to {output_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Seamless M4T v2 WER sweep .refpt files")
    parser.add_argument(
        "--task",
        choices=["t2st", "s2st", "all"],
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
        tasks = list(SPEECH_OUTPUT_TASKS) if args.task == "all" else [args.task]
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
