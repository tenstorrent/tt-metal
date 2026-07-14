# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated recaption input tests:
#   prepare_recaption_inputs parity, prepare_recaption_ar_bundle, recaption stage params.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/tokenizer/test_recaption_inputs.py -v

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import pytest

TOKENIZER_DIR = Path(__file__).resolve().parent
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))

from models.experimental.hunyuan_image_3_0.ref.generate import SamplingConfig
from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import (
    load_aligner,
    load_patch_embed,
    load_siglip2_vision,
    load_timestep_embedder,
)
from models.experimental.hunyuan_image_3_0.ref.recaption import (
    build_recaption_stage_params,
    decode_cot_text,
    is_meager_recaption_cot,
    run_recaption_ar,
    sanitize_recaption_cot_text,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    enrich_bundle_attention,
    prepare_recaption_ar_bundle,
    prepare_recaption_inputs,
    print_recaption_inputs_report,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import build_i2i_inputs_embeds
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR
from tokenizer_helpers import (
    HAS_INSTRUCT,
    HAS_UPSTREAM,
    MAX_LENGTH,
    RECAPTION_PROMPT,
    VIT_LAYERS,
    ensure_upstream_in_path,
)


def _upstream_recaption_ids(hf_model, *, prompt, system_prompt, image=None, bot_task="recaption"):
    out = hf_model.preprocess_inputs(
        prompt=prompt,
        image=image,
        mode="gen_text",
        bot_task=bot_task,
        system_prompt=system_prompt,
        cfg_factor=1,
        max_length=MAX_LENGTH,
    )
    return out["output"].tokens


def _manual_ar_bundle(
    tok, prompt, proc, wte, *, cond_images=None, bot_task="recaption", system_prompt=None, generator=None
):
    bundle = prepare_recaption_inputs(
        tok,
        prompt,
        cond_images=cond_images,
        bot_task=bot_task.split("_")[0] if bot_task == "think_recaption" else bot_task,
        system_prompt=system_prompt,
        sequence_template=None,
    )
    if cond_images is not None:
        bundle = build_i2i_inputs_embeds(
            bundle,
            wte,
            patch_embed=load_patch_embed(INSTRUCT_MODEL_DIR),
            time_embed=load_timestep_embedder("time_embed", INSTRUCT_MODEL_DIR),
            timestep_emb=load_timestep_embedder("timestep_emb", INSTRUCT_MODEL_DIR),
            vision_model=load_siglip2_vision(INSTRUCT_MODEL_DIR, num_layers=VIT_LAYERS),
            aligner=load_aligner(INSTRUCT_MODEL_DIR),
            model_dir=INSTRUCT_MODEL_DIR,
            generator=generator,
            vit_num_layers=VIT_LAYERS,
        )
    else:
        bundle.inputs_embeds = F.embedding(bundle.input_ids, wte.float())
    bundle.bot_task = bot_task
    return enrich_bundle_attention(bundle, proc)


# ---------------------------------------------------------------------------
# prepare_recaption_inputs parity vs upstream
# ---------------------------------------------------------------------------
def test_recaption_text_only_parity(bundled_tok, hf_preprocess_model):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_recaption", "recaption")
    bundle = prepare_recaption_inputs(
        bundled_tok,
        RECAPTION_PROMPT,
        bot_task="recaption",
        system_prompt=system_prompt,
    )
    upstream = _upstream_recaption_ids(
        hf_preprocess_model,
        prompt=RECAPTION_PROMPT,
        system_prompt=system_prompt,
        bot_task="recaption",
    )
    report = print_recaption_inputs_report(bundle, bundled_tok, upstream_ids=upstream, label="text_only_recaption")
    assert torch.equal(bundle.input_ids[0], upstream[0])
    assert report["recaption_inputs_ok"]
    assert bundle.mode == "gen_text"
    assert bundle.bot_task == "recaption"
    assert bundle.input_ids[0, -1].item() == bundled_tok.special.recaption_token_id


def test_recaption_i2i_with_cond_parity(instruct_tok, hf_preprocess_model, processor, rgb_image):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_unified", "image")
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")

    bundle = prepare_recaption_inputs(
        instruct_tok,
        RECAPTION_PROMPT,
        cond_images=cond,
        bot_task="recaption",
        system_prompt=system_prompt,
    )
    upstream = _upstream_recaption_ids(
        hf_preprocess_model,
        prompt=RECAPTION_PROMPT,
        system_prompt=system_prompt,
        image=rgb_image,
        bot_task="recaption",
    )
    report = print_recaption_inputs_report(bundle, instruct_tok, upstream_ids=upstream, label="i2i_recaption")
    assert torch.equal(bundle.input_ids[0], upstream[0])
    assert report["recaption_inputs_ok"]
    assert report["vit_placeholder_count"] > 0
    assert report["vae_placeholder_count"] > 0


def test_think_prefix_ends_with_think_token(bundled_tok, hf_preprocess_model):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_think_recaption", "think_recaption")
    bundle = prepare_recaption_inputs(
        bundled_tok,
        RECAPTION_PROMPT,
        bot_task="think",
        system_prompt=system_prompt,
    )
    upstream = _upstream_recaption_ids(
        hf_preprocess_model,
        prompt=RECAPTION_PROMPT,
        system_prompt=system_prompt,
        bot_task="think",
    )
    report = print_recaption_inputs_report(bundle, bundled_tok, upstream_ids=upstream, label="think_prefix")
    assert torch.equal(bundle.input_ids[0], upstream[0])
    assert bundle.input_ids[0, -1].item() == bundled_tok.special.think_token_id
    assert report["recaption_inputs_ok"]


# ---------------------------------------------------------------------------
# prepare_recaption_ar_bundle
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_INSTRUCT, reason="Instruct checkpoint required")
def test_recaption_ar_bundle_text_only_matches_manual(instruct_tok, processor, wte):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_recaption", "recaption")
    manual = _manual_ar_bundle(
        instruct_tok, RECAPTION_PROMPT, processor, wte, bot_task="recaption", system_prompt=system_prompt
    )
    bundle = prepare_recaption_ar_bundle(
        instruct_tok,
        RECAPTION_PROMPT,
        processor,
        wte,
        bot_task="recaption",
        system_prompt=system_prompt,
        sequence_template=None,
    )
    assert torch.equal(bundle.input_ids, manual.input_ids)
    assert torch.allclose(bundle.inputs_embeds, manual.inputs_embeds, atol=1e-4, rtol=1e-4)
    assert bundle.full_attn_slices == manual.full_attn_slices
    assert torch.equal(bundle.attention_mask, manual.attention_mask)


@pytest.mark.skipif(not HAS_INSTRUCT, reason="Instruct checkpoint required")
def test_recaption_ar_bundle_i2i_cond_matches_manual(instruct_tok, processor, wte, rgb_image):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_unified", "image")
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    gen_manual = torch.Generator().manual_seed(42)
    gen_bundle = torch.Generator().manual_seed(42)
    manual = _manual_ar_bundle(
        instruct_tok,
        RECAPTION_PROMPT,
        processor,
        wte,
        cond_images=cond,
        bot_task="recaption",
        system_prompt=system_prompt,
        generator=gen_manual,
    )
    bundle = prepare_recaption_ar_bundle(
        instruct_tok,
        RECAPTION_PROMPT,
        processor,
        wte,
        cond_images=cond,
        bot_task="recaption",
        system_prompt=system_prompt,
        sequence_template=None,
        patch_embed=load_patch_embed(INSTRUCT_MODEL_DIR),
        time_embed=load_timestep_embedder("time_embed", INSTRUCT_MODEL_DIR),
        timestep_emb=load_timestep_embedder("timestep_emb", INSTRUCT_MODEL_DIR),
        vision_model=load_siglip2_vision(INSTRUCT_MODEL_DIR, num_layers=VIT_LAYERS),
        aligner=load_aligner(INSTRUCT_MODEL_DIR),
        model_dir=INSTRUCT_MODEL_DIR,
        generator=gen_bundle,
    )
    report = print_recaption_inputs_report(bundle, instruct_tok, label="ar_bundle_i2i")
    assert report["vit_placeholder_count"] > 0
    assert report["vae_placeholder_count"] > 0
    assert torch.equal(bundle.input_ids, manual.input_ids)
    assert torch.allclose(bundle.inputs_embeds, manual.inputs_embeds, atol=1e-3, rtol=1e-3)
    assert bundle.rope_image_info == manual.rope_image_info
    assert torch.equal(bundle.attention_mask, manual.attention_mask)


@pytest.mark.skipif(not (HAS_INSTRUCT and HAS_UPSTREAM), reason="Instruct + upstream required")
def test_recaption_ar_bundle_ids_match_upstream_preprocess(instruct_tok, processor, wte, rgb_image):
    ensure_upstream_in_path()
    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_unified", "image")
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    config = HunyuanImage3Config.from_pretrained(str(INSTRUCT_MODEL_DIR))
    hf_model = HunyuanImage3ForCausalMM(config, skip_load_module={"all"})
    hf_model.load_tokenizer(str(INSTRUCT_MODEL_DIR))
    upstream_ids = hf_model.preprocess_inputs(
        prompt=RECAPTION_PROMPT,
        image=rgb_image,
        mode="gen_text",
        bot_task="recaption",
        system_prompt=system_prompt,
        cfg_factor=1,
        max_length=MAX_LENGTH,
    )["output"].tokens

    bundle = prepare_recaption_ar_bundle(
        instruct_tok,
        RECAPTION_PROMPT,
        processor,
        wte,
        cond_images=cond,
        bot_task="recaption",
        system_prompt=system_prompt,
        sequence_template=None,
        patch_embed=load_patch_embed(INSTRUCT_MODEL_DIR),
        time_embed=load_timestep_embedder("time_embed", INSTRUCT_MODEL_DIR),
        timestep_emb=load_timestep_embedder("timestep_emb", INSTRUCT_MODEL_DIR),
        vision_model=load_siglip2_vision(INSTRUCT_MODEL_DIR, num_layers=VIT_LAYERS),
        aligner=load_aligner(INSTRUCT_MODEL_DIR),
        model_dir=INSTRUCT_MODEL_DIR,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(bundle.input_ids[0], upstream_ids[0])
    assert bundle.inputs_embeds is not None
    assert bundle.full_attn_slices is not None


# ---------------------------------------------------------------------------
# Recaption stage params and cot_text decode
# ---------------------------------------------------------------------------
def test_recaption_stage_params_fixed_size(recaption_tok):
    params = build_recaption_stage_params(recaption_tok, "recaption", image_size=1024, sequence_template="instruct")
    assert params.first_bot_task == "recaption"
    assert params.need_ratio is False
    assert params.final_stop_tokens == [recaption_tok.special.end_recaption_token_id]
    assert params.stage_transitions == []


def test_think_recaption_stage_has_think_to_recaption_transition(recaption_tok):
    params = build_recaption_stage_params(
        recaption_tok, "think_recaption", image_size=1024, sequence_template="instruct"
    )
    assert params.first_bot_task == "think"
    assert len(params.stage_transitions) == 1
    stop_id, append_ids = params.stage_transitions[0]
    assert stop_id == recaption_tok.special.end_think_token_id
    assert append_ids == [recaption_tok.special.recaption_token_id]


def test_auto_image_size_adds_ratio_stage(recaption_tok):
    params = build_recaption_stage_params(recaption_tok, "recaption", image_size="auto", sequence_template="instruct")
    assert params.need_ratio is True
    assert params.stage_transitions
    transition_id, append_ids = params.stage_transitions[-1]
    assert transition_id == recaption_tok.special.end_recaption_token_id
    assert recaption_tok.special.answer_token_id in append_ids
    assert recaption_tok.special.boi_token_id in append_ids
    assert recaption_tok.special.size_token_id(1024) in append_ids
    assert recaption_tok.special.ratio_token_id(0) in params.final_stop_tokens


def test_decode_cot_text_recaption(recaption_tok):
    sp = recaption_tok.special
    recaption_str = recaption_tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = recaption_tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    body_ids = recaption_tok.encode("a dramatic sunset over mountains")
    gen_ids = [sp.recaption_token_id, *body_ids, sp.end_recaption_token_id]
    prefix = [1, 2, 3]
    sequences = torch.tensor([prefix + gen_ids], dtype=torch.long)
    cot = decode_cot_text(recaption_tok, sequences, len(prefix), "recaption")
    assert cot[0].startswith(recaption_str)
    assert end_recaption_str in cot[0]
    assert "dramatic sunset" in cot[0]


def test_sanitize_recaption_cot_text_strips_garbage(recaption_tok):
    sp = recaption_tok.special
    recaption_str = recaption_tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = recaption_tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
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


def test_is_meager_recaption_cot_quad_only(recaption_tok):
    sp = recaption_tok.special
    recaption_str = recaption_tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = recaption_tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    quad_only = f"{recaption_str}<quad><pos_x_-1><pos_y_4><pos_x_989><pos_y_995></quad>{end_recaption_str}"
    assert is_meager_recaption_cot(
        quad_only,
        recaption_open=recaption_str,
        recaption_close=end_recaption_str,
    )


def test_is_meager_recaption_cot_with_prose(recaption_tok):
    sp = recaption_tok.special
    recaption_str = recaption_tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = recaption_tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    good = f"{recaption_str}a studio photograph of a fluffy cat on a cushion{end_recaption_str}"
    assert not is_meager_recaption_cot(
        good,
        recaption_open=recaption_str,
        recaption_close=end_recaption_str,
    )


def test_cot_text_in_i2i_template(recaption_tok):
    out = recaption_tok.apply_chat_template(
        "a cat",
        image_size=1024,
        cot_text="<recaption>enhanced prompt</recaption>",
        system_prompt="test",
        sequence_template="instruct",
    )
    decoded = recaption_tok.decode(out["output"].tokens[0].tolist(), skip_special_tokens=False)
    assert "enhanced prompt" in decoded


def test_run_recaption_ar_greedy_recaption(recaption_tok):
    bundle = prepare_recaption_inputs(recaption_tok, "a cat on a mat", bot_task="recaption")
    input_length = bundle.input_ids.shape[1]
    vocab = recaption_tok.config.vocab_size
    body = recaption_tok.encode("enhanced caption text")
    forced = body + [recaption_tok.special.end_recaption_token_id]

    step = [0]

    def forward_logits_fn(ids):
        idx = step[0]
        step[0] += 1
        logits = torch.full((1, vocab), -20.0)
        if idx < len(forced):
            logits[0, forced[idx]] = 10.0
        else:
            logits[0, recaption_tok.special.end_recaption_token_id] = 10.0
        return logits

    cfg = SamplingConfig(do_sample=False, max_new_tokens=len(forced) + 4)
    result = run_recaption_ar(
        forward_logits_fn,
        bundle,
        recaption_tok,
        "recaption",
        image_size=1024,
        config=cfg,
    )
    assert result.input_length == input_length
    assert result.cot_text[0].startswith(
        recaption_tok.tokenizer.convert_ids_to_tokens(recaption_tok.special.recaption_token_id)
    )
    assert recaption_tok.special.end_recaption_token_id in result.sequences[0].tolist()
