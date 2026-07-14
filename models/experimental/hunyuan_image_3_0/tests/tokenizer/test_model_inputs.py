# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated model-input tests:
#   chat template layout, prepare_model_inputs parity, attention-mask bundle.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/tokenizer/test_model_inputs.py -v

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

TOKENIZER_DIR = Path(__file__).resolve().parent
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))

from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask
from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import (
    load_aligner,
    load_patch_embed,
    load_siglip2_vision,
    load_timestep_embedder,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    build_attention_mask_for_bundle,
    build_full_attn_slices,
    build_i2i_inputs_embeds,
    bundle_to_denoise_cond,
    enrich_bundle_attention,
    prepare_gen_image_inputs,
    prepare_i2i_inputs,
    scatter_distill_step_embeds,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, MODEL_DIR, load_tensors
from tokenizer_helpers import (
    HAS_UPSTREAM,
    HAS_WEIGHTS,
    HF_IMAGE_SIZE,
    IMAGE_SIZE,
    MAX_LENGTH,
    PROMPT,
    rope_pairs,
)


def _upstream_attention_mask(model, tokenizer_output, seq_len, bsz):
    batch_full_attn_slices = [model.image_processor.prepare_full_attn_slices(tokenizer_output, i) for i in range(bsz)]
    attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0).repeat(bsz, 1, 1)
    for i in range(bsz):
        for image_slice in batch_full_attn_slices[i]:
            attention_mask[i, image_slice, image_slice] = True
    return attention_mask.unsqueeze(1), batch_full_attn_slices


def _bidirectional_block(mask: torch.Tensor, span: slice) -> torch.Tensor:
    block = mask[0, 0, span, span]
    return torch.all(block)


# ---------------------------------------------------------------------------
# Chat template / host preprocess bundle
# ---------------------------------------------------------------------------
def test_cfg_uncond_replaces_text_not_image(hunyuan_tokenizer):
    out = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cfg_factor=2)["output"]
    assert out.tokens.shape[0] == 2
    cfg_id = hunyuan_tokenizer.special.cfg_token_id
    boi_id = hunyuan_tokenizer.special.boi_token_id

    cond, uncond = out.tokens[0], out.tokens[1]
    boi_idx = (cond == boi_id).nonzero(as_tuple=False)[0].item()
    assert (uncond[:boi_idx] == cfg_id).any()
    assert torch.equal(cond[boi_idx:], uncond[boi_idx:])
    assert torch.equal(out.gen_image_mask[0], out.gen_image_mask[1])
    assert torch.equal(out.gen_timestep_scatter_index[0], out.gen_timestep_scatter_index[1])


def test_host_preprocess_bundle(hunyuan_tokenizer):
    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=2)
    assert bundle.input_ids.shape == (2, bundle.seq_len)
    assert bundle.position_ids.shape == bundle.input_ids.shape
    assert bundle.rope_image_info is not None
    assert len(bundle.rope_image_info) == 2
    slice_i, (th, tw) = bundle.rope_image_info[0][0]
    assert th * tw == 4096
    assert slice_i.stop - slice_i.start == 4096
    assert bundle.vae_image_mask is None
    assert bundle.cond_timestep_scatter_index is None


def test_i2i_cond_joint_layout(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    t2i = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cfg_factor=1)["output"]
    i2i = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cond_images=cond, cfg_factor=1)["output"]

    assert i2i.tokens.shape[1] > t2i.tokens.shape[1]
    assert i2i.vae_image_mask is not None
    assert i2i.vit_image_mask is not None
    assert i2i.joint_image_slices is not None
    assert len(i2i.joint_image_slices[0]) == 1
    assert i2i.cond_timestep_scatter_index is not None
    assert len(i2i.cond_timestep_scatter_index[0]) == 1

    vae_len = int(i2i.vae_image_mask[0].sum())
    vit_len = int(i2i.vit_image_mask[0].sum())
    joint_len = i2i.joint_image_slices[0][0].stop - i2i.joint_image_slices[0][0].start
    assert joint_len == vae_len + vit_len + 1

    sep_id = hunyuan_tokenizer.special.joint_img_sep_token_id
    tokens = i2i.tokens[0].tolist()
    assert tokens.count(sep_id) == 1


def test_i2i_host_preprocess_bundle(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)

    assert bundle.batch_cond_images is not None
    assert len(bundle.batch_cond_images) == 1
    assert bundle.vae_image_mask is not None
    assert bundle.vit_image_mask is not None
    assert bundle.cond_timestep_scatter_index is not None
    assert bundle.rope_image_info is not None
    assert len(bundle.rope_image_info[0]) == 3


def test_i2i_cfg_preserves_cond_and_gen_blocks(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    out = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cond_images=cond, cfg_factor=2)["output"]
    assert out.tokens.shape[0] == 2
    cfg_id = hunyuan_tokenizer.special.cfg_token_id
    cond_row, uncond_row = out.tokens[0], out.tokens[1]
    assert (uncond_row == cfg_id).any()
    vae_slice = out.vae_image_slices[0][0]
    vit_slice = out.vit_image_slices[0][0]
    gen_slice = out.gen_image_slices[0][0]
    assert torch.equal(cond_row[vae_slice], uncond_row[vae_slice])
    assert torch.equal(cond_row[vit_slice], uncond_row[vit_slice])
    assert torch.equal(cond_row[gen_slice], uncond_row[gen_slice])
    assert len((cond_row == hunyuan_tokenizer.special.boi_token_id).nonzero(as_tuple=False)) == 2
    assert torch.equal(out.vae_image_mask[0], out.vae_image_mask[1])
    assert torch.equal(out.vit_image_mask[0], out.vit_image_mask[1])
    assert torch.equal(out.gen_image_mask[0], out.gen_image_mask[1])
    assert torch.equal(out.cond_timestep_scatter_index[0], out.cond_timestep_scatter_index[1])


# ---------------------------------------------------------------------------
# Upstream prepare_model_inputs parity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_UPSTREAM, reason="HF upstream repo required (set HUNYUAN_UPSTREAM)")
def test_t2i_prepare_model_inputs_tokenizer_parity(hf_preprocess_model, hunyuan_tokenizer, processor):
    hf_out = hf_preprocess_model.preprocess_inputs(
        prompt=PROMPT,
        mode="gen_image",
        image_size=HF_IMAGE_SIZE,
        cfg_factor=2,
        max_length=MAX_LENGTH,
    )
    hf_output = hf_out["output"]

    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=2)
    enrich_bundle_attention(bundle, processor)

    assert torch.equal(bundle.input_ids, hf_output.tokens)
    assert torch.equal(bundle.gen_image_mask, hf_output.gen_image_mask)

    hf_rope = hf_preprocess_model.build_batch_rope_image_info(hf_output, hf_out["sections"])
    assert rope_pairs(bundle.rope_image_info[0]) == rope_pairs(hf_rope[0])

    hf_mask, hf_spans = _upstream_attention_mask(hf_preprocess_model, hf_output, bundle.seq_len, 2)
    assert torch.equal(bundle.attention_mask, hf_mask)
    assert bundle.full_attn_slices[0] == hf_spans[0]


@pytest.mark.skipif(not HAS_UPSTREAM, reason="HF upstream repo required (set HUNYUAN_UPSTREAM)")
def test_i2i_prepare_model_inputs_tokenizer_parity(hf_preprocess_model, hunyuan_tokenizer, processor, rgb_image):
    hf_out = hf_preprocess_model.preprocess_inputs(
        prompt=PROMPT,
        image=rgb_image,
        mode="gen_image",
        image_size=HF_IMAGE_SIZE,
        cfg_factor=2,
        max_length=MAX_LENGTH,
    )
    hf_output = hf_out["output"]

    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=2)
    enrich_bundle_attention(bundle, processor)

    assert torch.equal(bundle.input_ids, hf_output.tokens)
    assert torch.equal(bundle.vae_image_mask, hf_output.vae_image_mask)
    assert torch.equal(bundle.vit_image_mask, hf_output.vit_image_mask)
    assert torch.equal(bundle.cond_timestep_scatter_index, hf_output.cond_timestep_scatter_index)

    hf_rope = hf_preprocess_model.build_batch_rope_image_info(hf_output, hf_out["sections"])
    assert rope_pairs(bundle.rope_image_info[0]) == rope_pairs(hf_rope[0])

    hf_mask, hf_spans = _upstream_attention_mask(hf_preprocess_model, hf_output, bundle.seq_len, 2)
    assert torch.equal(bundle.attention_mask, hf_mask)
    assert bundle.full_attn_slices[0] == hf_spans[0]


@pytest.mark.skipif(not (HAS_WEIGHTS and HAS_UPSTREAM), reason="HF upstream + checkpoint required")
def test_i2i_prepare_model_inputs_encode_parity(hf_encode_model, hunyuan_tokenizer, processor, rgb_image):
    hf_inputs = hf_encode_model.prepare_model_inputs(
        prompt=PROMPT,
        image=rgb_image,
        mode="gen_image",
        image_size=HF_IMAGE_SIZE,
        max_length=MAX_LENGTH,
    )
    hf_output = hf_inputs["tokenizer_output"]

    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=2)
    wte = load_tensors(MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    gen = torch.Generator().manual_seed(int(hf_inputs["generator"][0].initial_seed()))

    bundle = build_i2i_inputs_embeds(
        bundle,
        wte,
        patch_embed=load_patch_embed(MODEL_DIR),
        time_embed=load_timestep_embedder("time_embed", MODEL_DIR),
        timestep_emb=load_timestep_embedder("timestep_emb", MODEL_DIR),
        vision_model=load_siglip2_vision(MODEL_DIR, num_layers=27),
        aligner=load_aligner(MODEL_DIR),
        generator=gen,
    )
    enrich_bundle_attention(bundle, processor)

    assert torch.equal(bundle.input_ids, hf_inputs["input_ids"].cpu())
    assert torch.equal(bundle.position_ids, hf_inputs["position_ids"].cpu())

    hf_mask, _ = _upstream_attention_mask(
        hf_encode_model, hf_output, bundle.seq_len, int(hf_inputs["input_ids"].shape[0])
    )
    assert torch.equal(bundle.attention_mask, hf_mask.cpu())

    assert isinstance(bundle.cond_vae_images, torch.Tensor)
    assert bundle.cond_vae_images.shape == hf_inputs["cond_vae_images"].shape
    assert torch.allclose(
        bundle.cond_vae_images.float(),
        hf_inputs["cond_vae_images"].cpu().float(),
        atol=1e-4,
        rtol=1e-4,
    )

    with torch.no_grad():
        hf_hidden = torch.nn.functional.embedding(hf_inputs["input_ids"], wte.float())
        hf_hidden = hf_encode_model.instantiate_vae_image_tokens(
            hf_hidden,
            hf_inputs["cond_timesteps"],
            hf_inputs["cond_vae_images"],
            hf_inputs["cond_vae_image_mask"],
        )
        hf_hidden = hf_encode_model.instantiate_continuous_tokens(
            hf_hidden,
            hf_inputs["cond_timesteps"],
            hf_inputs["cond_timesteps_index"],
        )
        hf_hidden = hf_encode_model.instantiate_vit_image_tokens(
            hf_hidden,
            hf_inputs["cond_vit_images"],
            hf_inputs["cond_vit_image_mask"],
            hf_inputs["cond_vit_image_kwargs"],
        )

    assert torch.allclose(
        bundle.inputs_embeds.float(),
        hf_hidden.cpu().float(),
        atol=1e-3,
        rtol=1e-3,
    )


# ---------------------------------------------------------------------------
# Attention mask bundle
# ---------------------------------------------------------------------------
def test_t2i_full_attn_single_gen_span(hunyuan_tokenizer, processor):
    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=1)
    spans = build_full_attn_slices(bundle, processor)
    assert len(spans) == 1
    assert len(spans[0]) == 1
    gen_slice = bundle.gen_image_slices[0][0]
    assert spans[0][0] == gen_slice

    mask = build_attention_mask_for_bundle(bundle, processor)
    assert mask.shape == (1, 1, bundle.seq_len, bundle.seq_len)
    assert _bidirectional_block(mask, gen_slice)


def test_i2i_full_attn_joint_plus_gen(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)

    spans = build_full_attn_slices(bundle, processor)
    assert len(spans) == 1
    assert len(spans[0]) == 2

    joint_slice = bundle.joint_image_slices[0][0]
    gen_slice = bundle.gen_image_slices[0][0]
    assert spans[0][0] == joint_slice
    assert spans[0][1] == gen_slice

    mask = build_attention_mask_for_bundle(bundle, processor)
    assert _bidirectional_block(mask, joint_slice)
    assert _bidirectional_block(mask, gen_slice)

    cross = mask[0, 0, joint_slice.start, gen_slice.start]
    assert not bool(cross)


def test_enrich_bundle_attention_populates_fields(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)
    assert bundle.full_attn_slices is None
    assert bundle.attention_mask is None

    enrich_bundle_attention(bundle, processor)
    assert bundle.full_attn_slices is not None
    assert bundle.attention_mask is not None
    assert bundle.attention_mask.dtype == torch.bool


def test_bundle_to_denoise_cond_i2i_shape(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)
    wte = load_tensors(MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    cond_dict = bundle_to_denoise_cond(bundle, wte, processor)

    assert "text_pre" in cond_dict
    assert cond_dict["text_post"] is None or cond_dict["text_post"].shape[1] >= 0
    assert cond_dict["attention_mask"].shape == (1, 1, bundle.seq_len, bundle.seq_len)
    assert len(cond_dict["image_infos"][0]) == 3
    assert cond_dict["gen_slice"].stop - cond_dict["gen_slice"].start == 4096


def test_bundle_to_denoise_cond_matches_manual_build(hunyuan_tokenizer, processor):
    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=1)
    spans = build_full_attn_slices(bundle, processor)
    manual = build_attention_mask(bundle.seq_len, spans, bsz=1)
    auto = build_attention_mask_for_bundle(bundle, processor)
    assert torch.equal(manual, auto)


@pytest.mark.skipif(
    not (INSTRUCT_MODEL_DIR / "tokenizer.json").is_file(),
    reason="Instruct tokenizer (with <timestep_r>) not available",
)
def test_distill_i2i_bundle_has_guidance_and_timestep_r_indices(processor, rgb_image):
    base_tok = HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template="instruct")
    distill_tok = HunyuanTokenizer(
        replace(base_tok.config, cfg_distilled=True, use_meanflow=True),
        base_tok.tokenizer,
        base_tok.special,
        sequence_template="instruct",
    )
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(distill_tok, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)
    assert bundle.guidance_scatter_index is not None
    assert bundle.gen_timestep_r_scatter_index is not None
    assert bundle.guidance_scatter_index.numel() == 1
    assert bundle.gen_timestep_r_scatter_index.numel() == 1


def test_scatter_distill_step_embeds_updates_three_slots():
    from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder

    hidden = 8
    emb = TimestepEmbedder(hidden_size=hidden)
    bsz, seq, _ = 1, 6, hidden
    base = torch.randn(bsz, seq, hidden)
    ref = base.clone()
    idx_t = torch.tensor([[2]], dtype=torch.long)
    idx_g = torch.tensor([[3]], dtype=torch.long)
    idx_r = torch.tensor([[4]], dtype=torch.long)

    out = scatter_distill_step_embeds(
        base,
        t_scalar=100.0,
        gen_timestep_scatter_index=idx_t,
        timestep_emb=emb,
        guidance_scalar=2500.0,
        guidance_scatter_index=idx_g,
        guidance_emb=emb,
        t_r_scalar=50.0,
        gen_timestep_r_scatter_index=idx_r,
        timestep_r_emb=emb,
    )

    with torch.no_grad():
        assert not torch.allclose(out[0, 2], ref[0, 2])
        assert not torch.allclose(out[0, 3], ref[0, 3])
        assert not torch.allclose(out[0, 4], ref[0, 4])
        assert torch.allclose(out[0, 0], ref[0, 0])
        assert torch.allclose(out[0, 1], ref[0, 1])
        assert torch.allclose(out[0, 5], ref[0, 5])
