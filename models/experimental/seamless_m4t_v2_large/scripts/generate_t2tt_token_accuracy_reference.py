# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Offline HF references for Seamless M4T v2 E2E token-matching tests.

Runs HuggingFace once per task with teacher-forced greedy decode and saves per-step top-5
predictions to ``.refpt`` files consumed by ``test_seamless_e2e_token_matching.py``.

Usage::

    python models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py
    python models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py --task all
    python models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py --sweep --task all
    python models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py --sweep --task t2tt --seq_len 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.scripts.demo_perf_sweep import (
    SEQ_LEN_MAX,
    SRC_LANG,
    ensure_long_audio,
    ensure_long_story,
    speech_inputs_for_len,
    text_inputs_for_len,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import (
    DEFAULT_SRC_LANG,
    TextDecoderPccInputs,
    _hf_speech_encoder_hidden_and_mask,
    _hf_text_encoder_hidden,
    decoder_seed_ids,
    load_hf_model_and_processor,
    tokenize_source_text,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import align_case_for_tt_prefill
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_logit_pcc_helpers import (
    MAX_ENC_SEQ,
    make_speech_e2e_inputs,
    resolve_preamble_wav_for_tests,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    ALL_E2E_TASKS,
    TASK_TGT_LANG,
    TEXT_INPUT_TASKS,
    T2TT_REF_SOURCE_TEXT,
    T2TT_REF_TGT_LANG,
    default_refpt_path,
)

_DEFAULT_MAX_STEPS = 128
_SWEEP_REF_DIR = Path(__file__).resolve().parent.parent / "tests" / "teacher_forced_sweep_outputs" / "references"


def sweep_refpt_path(task: str, seq_len: int, max_decode_steps: int = _DEFAULT_MAX_STEPS) -> Path:
    return _SWEEP_REF_DIR / f"seamless_m4t_v2_{task}_len{seq_len}_eval{max_decode_steps}.refpt"


def _hf_teacher_forced_top5_reference(
    model,
    *,
    encoder_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    seed_ids: torch.Tensor,
    max_decode_steps: int,
) -> tuple[list[int], list[torch.Tensor]]:
    decoder = model.text_decoder
    lm_head = model.lm_head
    pad_id = int(model.config.pad_token_id)
    eos_id = int(getattr(model.generation_config, "eos_token_id", pad_id))
    seed_mask = torch.ones_like(seed_ids)

    p0 = next(decoder.parameters())
    enc_hidden = encoder_hidden.to(device=p0.device, dtype=p0.dtype)
    enc_mask_dev = enc_mask.to(device=p0.device, dtype=torch.long)
    seed_ids_dev = seed_ids.to(device=p0.device)

    teacher_tokens: list[int] = []
    top5_rows: list[torch.Tensor] = []

    with torch.no_grad():
        prefill = decoder(
            input_ids=seed_ids_dev,
            attention_mask=seed_mask.to(p0.device),
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask_dev,
            use_cache=True,
            return_dict=True,
        )
        past = prefill.past_key_values
        tok = int(lm_head(prefill.last_hidden_state[:, -1:, :]).float().argmax(dim=-1).item())

        for _ in range(max_decode_steps):
            teacher_tokens.append(tok)
            step_ids = torch.full((1, 1), tok, dtype=torch.long, device=p0.device)
            step_out = decoder(
                input_ids=step_ids,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask_dev,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = lm_head(step_out.last_hidden_state).float()
            probs = torch.softmax(logits, dim=-1)
            _, top5 = torch.topk(probs, k=5, dim=-1)
            top5_rows.append(top5[0, 0].to(torch.int64).cpu())
            past = step_out.past_key_values
            tok = int(logits[0, 0].argmax().item())
            if tok == eos_id:
                break

    return teacher_tokens, top5_rows


def generate_t2tt_reference(
    *,
    weights_dir: str,
    output_file: Path,
    task: str = "t2tt",
    src_text: str = T2TT_REF_SOURCE_TEXT,
    src_lang: str = DEFAULT_SRC_LANG,
    tgt_lang: str = T2TT_REF_TGT_LANG,
    max_decode_steps: int = _DEFAULT_MAX_STEPS,
) -> None:
    model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)

    src_ids, src_mask = tokenize_source_text(processor, src_text, src_lang)
    encoder_hidden = _hf_text_encoder_hidden(model, src_ids, src_mask)
    enc_mask = src_mask.to(device=encoder_hidden.device, dtype=torch.long)
    seed_ids = decoder_seed_ids(model, tgt_lang)

    teacher_tokens, top5_rows = _hf_teacher_forced_top5_reference(
        model,
        encoder_hidden=encoder_hidden,
        enc_mask=enc_mask,
        seed_ids=seed_ids,
        max_decode_steps=max_decode_steps,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "src_ids": src_ids.cpu(),
        "src_mask": src_mask.cpu(),
        "seed_ids": seed_ids.cpu(),
        "teacher_tokens": torch.tensor(teacher_tokens, dtype=torch.int64),
        "top5_tokens": torch.stack(top5_rows, dim=0),
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src_text": src_text,
        "max_decode_steps": max_decode_steps,
    }
    torch.save(payload, output_file)
    n = len(teacher_tokens)
    logger.info(f"Saved {task.upper()} token-accuracy reference ({n} decode steps) to {output_file}")


def generate_speech_reference(
    *,
    weights_dir: str,
    output_file: Path,
    task: str,
    tgt_lang: str,
    max_decode_steps: int = _DEFAULT_MAX_STEPS,
    enc_seq_len: int = MAX_ENC_SEQ,
    wav_path: Path | None = None,
) -> None:
    model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
    speech = make_speech_e2e_inputs(
        model,
        processor,
        tgt_lang=tgt_lang,
        enc_seq_len=enc_seq_len,
        wav_path=wav_path,
    )
    case = speech.case
    aligned = align_case_for_tt_prefill(case, int(model.config.pad_token_id))

    teacher_tokens, top5_rows = _hf_teacher_forced_top5_reference(
        model,
        encoder_hidden=case.encoder_hidden_states,
        enc_mask=case.encoder_attention_mask,
        seed_ids=case.input_ids,
        max_decode_steps=max_decode_steps,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "input_features": speech.input_features.cpu(),
        "mel_attention_mask": speech.mel_attention_mask.cpu(),
        "seed_ids": case.input_ids.cpu(),
        "encoder_hidden_states": aligned.encoder_hidden_states.cpu(),
        "encoder_attention_mask": aligned.encoder_attention_mask.cpu(),
        "teacher_tokens": torch.tensor(teacher_tokens, dtype=torch.int64),
        "top5_tokens": torch.stack(top5_rows, dim=0),
        "tgt_lang": tgt_lang,
        "max_decode_steps": max_decode_steps,
    }
    torch.save(payload, output_file)
    n = len(teacher_tokens)
    logger.info(f"Saved {task.upper()} token-accuracy reference ({n} decode steps) to {output_file}")


def generate_text_sweep_reference(
    *,
    weights_dir: str,
    output_file: Path,
    task: str,
    seq_len: int,
    src_lang: str = SRC_LANG,
    max_decode_steps: int = _DEFAULT_MAX_STEPS,
    story: str | None = None,
) -> None:
    """Teacher-forced HF reference for text tasks at exactly ``seq_len`` source tokens."""
    if task not in TEXT_INPUT_TASKS:
        raise ValueError(f"generate_text_sweep_reference expects a text task, got {task!r}")
    model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
    story_text = story if story is not None else ensure_long_story()
    src_ids, src_mask = text_inputs_for_len(processor, story_text, seq_len, src_lang=src_lang)
    tgt_lang = TASK_TGT_LANG[task]
    encoder_hidden = _hf_text_encoder_hidden(model, src_ids, src_mask)
    enc_mask = src_mask.to(device=encoder_hidden.device, dtype=torch.long)
    seed_ids = decoder_seed_ids(model, tgt_lang)

    teacher_tokens, top5_rows = _hf_teacher_forced_top5_reference(
        model,
        encoder_hidden=encoder_hidden,
        enc_mask=enc_mask,
        seed_ids=seed_ids,
        max_decode_steps=max_decode_steps,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "seq_len": seq_len,
        "src_ids": src_ids.cpu(),
        "src_mask": src_mask.cpu(),
        "seed_ids": seed_ids.cpu(),
        "teacher_tokens": torch.tensor(teacher_tokens, dtype=torch.int64),
        "top5_tokens": torch.stack(top5_rows, dim=0),
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "max_decode_steps": max_decode_steps,
    }
    torch.save(payload, output_file)
    n = len(teacher_tokens)
    logger.info(f"Saved {task.upper()} sweep reference len={seq_len} ({n} decode steps) to {output_file}")


def generate_speech_sweep_reference(
    *,
    weights_dir: str,
    output_file: Path,
    task: str,
    mel_frames: int,
    max_decode_steps: int = _DEFAULT_MAX_STEPS,
) -> None:
    """Teacher-forced HF reference for speech tasks at exactly ``mel_frames`` mel frames."""
    if task not in {"s2tt", "s2st", "asr"}:
        raise ValueError(f"generate_speech_sweep_reference expects a speech task, got {task!r}")
    model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
    sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16_000))
    wav, _ = ensure_long_audio(processor, sample_rate, min_mel_frames=max(mel_frames, SEQ_LEN_MAX))
    audio = processor(audios=wav, sampling_rate=sample_rate, return_tensors="pt")
    input_features, mel_mask = speech_inputs_for_len(
        processor,
        audio["input_features"],
        audio["attention_mask"],
        mel_frames,
    )
    input_features = input_features.to(dtype=next(model.parameters()).dtype)
    encoder_hidden, enc_attn = _hf_speech_encoder_hidden_and_mask(model, input_features, mel_mask)
    tgt_lang = TASK_TGT_LANG[task]
    seed_ids = decoder_seed_ids(model, tgt_lang)
    aligned = align_case_for_tt_prefill(
        TextDecoderPccInputs(
            input_ids=seed_ids,
            attention_mask=torch.ones_like(seed_ids),
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=enc_attn,
        ),
        int(model.config.pad_token_id),
    )

    teacher_tokens, top5_rows = _hf_teacher_forced_top5_reference(
        model,
        encoder_hidden=aligned.encoder_hidden_states,
        enc_mask=aligned.encoder_attention_mask,
        seed_ids=seed_ids,
        max_decode_steps=max_decode_steps,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "seq_len": mel_frames,
        "input_features": input_features.cpu(),
        "mel_attention_mask": mel_mask.cpu(),
        "seed_ids": seed_ids.cpu(),
        "encoder_hidden_states": aligned.encoder_hidden_states.cpu(),
        "encoder_attention_mask": aligned.encoder_attention_mask.cpu(),
        "teacher_tokens": torch.tensor(teacher_tokens, dtype=torch.int64),
        "top5_tokens": torch.stack(top5_rows, dim=0),
        "tgt_lang": tgt_lang,
        "max_decode_steps": max_decode_steps,
    }
    torch.save(payload, output_file)
    n = len(teacher_tokens)
    logger.info(f"Saved {task.upper()} sweep reference mel={mel_frames} ({n} decode steps) to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Seamless M4T v2 token-accuracy .refpt files")
    parser.add_argument(
        "--task",
        choices=["t2tt", "t2st", "s2tt", "s2st", "asr", "all"],
        default="t2tt",
        help="Which task reference to generate (default: t2tt)",
    )
    parser.add_argument("--output_file", type=Path, default=None, help="Override output .refpt path")
    parser.add_argument("--max_decode_steps", type=int, default=_DEFAULT_MAX_STEPS)
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
        from models.experimental.seamless_m4t_v2_large.scripts.demo_perf_sweep import sequence_lengths

        if args.seq_len is not None:
            lengths = [args.seq_len]
        else:
            lengths = sequence_lengths(args.min_len, args.max_len)
        tasks = list(ALL_E2E_TASKS) if args.task == "all" else [args.task]
        for task in tasks:
            for seq_len in lengths:
                out = sweep_refpt_path(task, seq_len, args.max_decode_steps)
                if task in TEXT_INPUT_TASKS:
                    generate_text_sweep_reference(
                        weights_dir=weights_dir,
                        output_file=out,
                        task=task,
                        seq_len=seq_len,
                        max_decode_steps=args.max_decode_steps,
                    )
                else:
                    generate_speech_sweep_reference(
                        weights_dir=weights_dir,
                        output_file=out,
                        task=task,
                        mel_frames=seq_len,
                        max_decode_steps=args.max_decode_steps,
                    )
        return

    tasks = list(ALL_E2E_TASKS) if args.task == "all" else [args.task]
    for task in tasks:
        out = args.output_file if args.output_file and len(tasks) == 1 else default_refpt_path(task)
        if task in TEXT_INPUT_TASKS:
            generate_t2tt_reference(
                weights_dir=weights_dir,
                output_file=out,
                task=task,
                max_decode_steps=args.max_decode_steps,
            )
        elif task == "s2tt":
            generate_speech_reference(
                weights_dir=weights_dir,
                output_file=out,
                task="s2tt",
                tgt_lang="eng",
                max_decode_steps=args.max_decode_steps,
            )
        elif task == "s2st":
            generate_speech_reference(
                weights_dir=weights_dir,
                output_file=out,
                task="s2st",
                tgt_lang="spa",
                max_decode_steps=args.max_decode_steps,
                wav_path=resolve_preamble_wav_for_tests(),
            )
        else:
            generate_speech_reference(
                weights_dir=weights_dir,
                output_file=out,
                task="asr",
                tgt_lang="eng",
                max_decode_steps=args.max_decode_steps,
            )


if __name__ == "__main__":
    main()
