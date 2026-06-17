# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Offline HF references for Seamless M4T v2 E2E token-matching tests.

Runs HuggingFace once per task with teacher-forced greedy decode and saves per-step top-5
predictions to ``.refpt`` files consumed by ``test_seamless_e2e_token_matching.py``.

Usage::

    python models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py
    python models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py --task all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import (
    DEFAULT_SRC_LANG,
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
    TEXT_INPUT_TASKS,
    T2TT_REF_SOURCE_TEXT,
    T2TT_REF_TGT_LANG,
    default_refpt_path,
)

_DEFAULT_MAX_STEPS = 128


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
    args = parser.parse_args()
    weights_dir = args.weights_dir or ensure_seamless_m4t_v2_large_weights()

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
