# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where does the image path diverge from HF? Tower+embedder PCC on a real image, then the first token.

    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=0,1,2,3 \\
    GEMMA4_VISION_REF=glx_bench/vision_ref/gemma-4-31B-it_dog.pt pytest vision_e2e_probe.py -sv

The reference file comes from glx_bench/vision_ref/hf_vision_ref.py (HF CPU run on the same
image + prompt): input_ids, pixel_values, image_position_ids, image_features [n_img, text_hidden],
image_token_id, generated text.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.demo.vision_demo import _device_params, create_tt_page_table
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.common import PagedAttentionConfig


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_vision_e2e_probe(mesh_device):
    ref_path = os.environ["GEMMA4_VISION_REF"]
    ref = torch.load(ref_path)
    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    input_ids, pixel_values, pos = ref["input_ids"], ref["pixel_values"], ref["image_position_ids"]
    prompt_len = int(input_ids.shape[1])
    max_seq_len = 4096
    page_params = {"page_block_size": 64, "page_max_num_blocks": max_seq_len // 64}
    pac = PagedAttentionConfig(block_size=64, max_num_blocks=page_params["page_max_num_blocks"])
    page_table = create_tt_page_table(1, pac)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        paged_attention_config=pac,
        multimodal=True,
    )
    assert generator.is_multimodal

    # 1. Tower + embedder vs HF image features on the real image.
    embeds_tt = generator.encode_vision(pixel_values.to(torch.bfloat16), pos.to(torch.long))
    tt_feats = (
        ttnn.to_torch(ttnn.get_device_tensors(embeds_tt)[0]).float()
        if hasattr(mesh_device, "shape")
        else ttnn.to_torch(embeds_tt).float()
    )
    hf_feats = ref["image_features"].float()
    logger.info(
        f"TT image features {tuple(tt_feats.shape)} abs mean {tt_feats.abs().mean():.4f} | HF {tuple(hf_feats.shape)} abs mean {hf_feats.abs().mean():.4f}"
    )
    assert tt_feats.shape == hf_feats.shape, "soft-token count differs from HF"
    ok, pcc = comp_pcc(hf_feats, tt_feats, 0.95)
    per_tok = torch.nn.functional.cosine_similarity(hf_feats, tt_feats, dim=-1)
    logger.info(f"image features PCC vs HF: {pcc}; per-token cosine min/mean {per_tok.min():.3f}/{per_tok.mean():.3f}")

    # 2. First token after the multimodal prefill vs HF's first generated token.
    out = generator.prefill_forward_multimodal(
        input_ids,
        pixel_values.to(torch.bfloat16),
        pos.to(torch.long),
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[prompt_len],
    )
    logits = out[0] if isinstance(out, tuple) else out
    tt_tok = int(torch.argmax(logits.reshape(1, -1).float(), dim=-1))
    hf_text = ref["generated"]
    hf_first = tokenizer.encode(hf_text, add_special_tokens=False)[0]
    logger.info(
        f"first token: TT {tt_tok} ({tokenizer.decode([tt_tok])!r}) vs HF {hf_first} ({tokenizer.decode([hf_first])!r}); HF text: {hf_text[:80]!r}"
    )
    assert ok, f"image features PCC {pcc} < 0.95"
