# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for recaption stage params and cot_text decode."""

from __future__ import annotations

import pytest
import torch

from models.experimental.hunyuan_image_3_0.ref.recaption import (
    build_recaption_stage_params,
    decode_cot_text,
    is_meager_recaption_cot,
    run_recaption_ar,
    sanitize_recaption_cot_text,
)
from models.experimental.hunyuan_image_3_0.ref.generate import SamplingConfig
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs


@pytest.fixture(scope="module")
def tok():
    return HunyuanTokenizer.from_pretrained(sequence_template="instruct")


def test_recaption_stage_params_fixed_size(tok):
    params = build_recaption_stage_params(tok, "recaption", image_size=1024, sequence_template="instruct")
    assert params.first_bot_task == "recaption"
    assert params.need_ratio is False
    assert params.final_stop_tokens == [tok.special.end_recaption_token_id]
    assert params.stage_transitions == []


def test_think_recaption_stage_has_think_to_recaption_transition(tok):
    params = build_recaption_stage_params(tok, "think_recaption", image_size=1024, sequence_template="instruct")
    assert params.first_bot_task == "think"
    assert len(params.stage_transitions) == 1
    stop_id, append_ids = params.stage_transitions[0]
    assert stop_id == tok.special.end_think_token_id
    assert append_ids == [tok.special.recaption_token_id]


def test_auto_image_size_adds_ratio_stage(tok):
    params = build_recaption_stage_params(tok, "recaption", image_size="auto", sequence_template="instruct")
    assert params.need_ratio is True
    assert params.stage_transitions
    transition_id, append_ids = params.stage_transitions[-1]
    assert transition_id == tok.special.end_recaption_token_id
    assert tok.special.answer_token_id in append_ids
    assert tok.special.boi_token_id in append_ids
    assert tok.special.size_token_id(1024) in append_ids
    assert tok.special.ratio_token_id(0) in params.final_stop_tokens


def test_decode_cot_text_recaption(tok):
    sp = tok.special
    recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    body_ids = tok.encode("a dramatic sunset over mountains")
    gen_ids = [sp.recaption_token_id, *body_ids, sp.end_recaption_token_id]
    prefix = [1, 2, 3]
    sequences = torch.tensor([prefix + gen_ids], dtype=torch.long)
    cot = decode_cot_text(tok, sequences, len(prefix), "recaption")
    assert cot[0].startswith(recaption_str)
    assert end_recaption_str in cot[0]
    assert "dramatic sunset" in cot[0]


def test_sanitize_recaption_cot_text_strips_garbage(tok):
    sp = tok.special
    recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    messy = (
        f"{recaption_str}<quad><pos_x_294><quad><pos_x_269><pos_y_399>"
        f"<pos_x_700><quad><pos_y_437></quad><|endoftext|>"
        "18．某同学在用显微镜观察菠菜叶徒手切片"
    )
    clean = sanitize_recaption_cot_text(
        messy,
        recaption_open=recaption_str,
        recaption_close=end_recaption_str,
    )
    assert clean.endswith(end_recaption_str)
    assert "菠菜" not in clean
    assert "<quad>" in clean
    assert "<|endoftext|>" not in clean


def test_is_meager_recaption_cot_quad_only(tok):
    sp = tok.special
    recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    quad_only = f"{recaption_str}<quad><pos_x_-1><pos_y_4><pos_x_989><pos_y_995></quad>{end_recaption_str}"
    assert is_meager_recaption_cot(
        quad_only,
        recaption_open=recaption_str,
        recaption_close=end_recaption_str,
    )


def test_is_meager_recaption_cot_with_prose(tok):
    sp = tok.special
    recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    good = f"{recaption_str}a studio photograph of a fluffy cat on a cushion{end_recaption_str}"
    assert not is_meager_recaption_cot(
        good,
        recaption_open=recaption_str,
        recaption_close=end_recaption_str,
    )


def test_cot_text_in_i2i_template(tok):
    out = tok.apply_chat_template(
        "a cat",
        image_size=1024,
        cot_text="<recaption>enhanced prompt</recaption>",
        system_prompt="test",
        sequence_template="instruct",
    )
    decoded = tok.decode(out["output"].tokens[0].tolist(), skip_special_tokens=False)
    assert "enhanced prompt" in decoded


def test_run_recaption_ar_greedy_recaption(tok):
    bundle = prepare_recaption_inputs(tok, "a cat on a mat", bot_task="recaption")
    input_length = bundle.input_ids.shape[1]
    vocab = tok.config.vocab_size
    body = tok.encode("enhanced caption text")
    forced = body + [tok.special.end_recaption_token_id]

    step = [0]

    def forward_logits_fn(ids):
        idx = step[0]
        step[0] += 1
        logits = torch.full((1, vocab), -20.0)
        if idx < len(forced):
            logits[0, forced[idx]] = 10.0
        else:
            logits[0, tok.special.end_recaption_token_id] = 10.0
        return logits

    cfg = SamplingConfig(do_sample=False, max_new_tokens=len(forced) + 4)
    result = run_recaption_ar(
        forward_logits_fn,
        bundle,
        tok,
        "recaption",
        image_size=1024,
        config=cfg,
    )
    assert result.input_length == input_length
    assert result.cot_text[0].startswith(tok.tokenizer.convert_ids_to_tokens(tok.special.recaption_token_id))
    assert tok.special.end_recaption_token_id in result.sequences[0].tolist()
