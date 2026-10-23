# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does the on-device merge put the image soft tokens where the text stream expects them?

Replicates the merge of ``_prefill_single_user_multimodal`` step by step and reads
the merged embeddings back: rows at image_token positions must equal the image
soft tokens (PCC), rows elsewhere must equal the text token embeddings.

    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=0,1,2,3 \\
    GEMMA4_VISION_REF=glx_bench/vision_ref/gemma-4-31B-it_dog.pt pytest vision_merge_probe.py -sv
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.demo.vision_demo import _device_params
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len


def _host(t, mesh_device):
    return (
        ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        if hasattr(mesh_device, "shape")
        else ttnn.to_torch(t).float()
    )


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
def test_vision_merge_probe(mesh_device):
    ref = torch.load(os.environ["GEMMA4_VISION_REF"])
    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    input_ids, pixel_values, pos = ref["input_ids"], ref["pixel_values"], ref["image_position_ids"]
    pac = PagedAttentionConfig(block_size=64, max_num_blocks=4096 // 64)
    generator, tt_kv_cache, _tok = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=4096,
        paged_attention_config=pac,
        multimodal=True,
    )
    model = generator.model[0]
    text_hidden = model.hidden_size
    prompt_len = int(input_ids.shape[1])
    prefill_seq_len = get_padded_prefill_len(prompt_len)

    image_embeds_tt = generator.encode_vision(pixel_values.to(torch.bfloat16), pos.to(torch.long))
    image_embeds = _host(image_embeds_tt, mesh_device)  # [num_valid, hidden]
    num_valid = int(image_embeds.shape[0])

    ids = input_ids[0].to(torch.long)
    image_mask = ids == generator.image_token_id
    image_positions = torch.where(image_mask)[0].to(torch.int32)
    assert int(image_mask.sum()) == num_valid, (int(image_mask.sum()), num_valid)
    llm_ids = ids.clone()
    llm_ids[image_mask] = generator.pad_token_id
    llm_ids = torch.cat(
        [llm_ids, torch.full((prefill_seq_len - prompt_len,), generator.pad_token_id, dtype=torch.long)]
    )
    llm_ids_tt = ttnn.from_torch(
        llm_ids.view(1, prefill_seq_len),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=generator._replicate_mapper(),
    )
    text_embeds_tt = model.embed_tokens(llm_ids_tt)
    text_embeds_tt = ttnn.reshape(text_embeds_tt, (prefill_seq_len, text_hidden))
    text_embeds_tt = ttnn.to_layout(text_embeds_tt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    text_embeds = _host(text_embeds_tt, mesh_device)[:prefill_seq_len]

    scatter_index = image_positions.view(num_valid, 1).expand(num_valid, text_hidden).contiguous()
    scatter_index_tt = ttnn.from_torch(
        scatter_index,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=generator._replicate_mapper(),
    )
    merged_tt = ttnn.scatter(text_embeds_tt, 0, scatter_index_tt, image_embeds_tt)
    merged = _host(merged_tt, mesh_device)[:prefill_seq_len]

    img_rows = merged[image_positions.long()]
    _, pcc_img = comp_pcc(image_embeds, img_rows, 0.99)
    text_pos = torch.where(~image_mask)[0]
    _, pcc_txt = comp_pcc(text_embeds[text_pos], merged[text_pos], 0.99)
    unchanged_img = (img_rows - text_embeds[image_positions.long()]).abs().max().item()
    logger.info(
        f"merge: image rows vs soft tokens PCC {pcc_img}; text rows PCC {pcc_txt}; "
        f"image rows differ from pad embeds by max {unchanged_img:.4f}; "
        f"|image| {image_embeds.abs().mean():.3f} |text| {text_embeds[text_pos].abs().mean():.3f} |pad| {text_embeds[image_positions.long()].abs().mean():.3f}"
    )
    assert float(pcc_img) > 0.99, f"image rows PCC {pcc_img}: scatter did not place the soft tokens"
    assert float(pcc_txt) > 0.99, f"text rows PCC {pcc_txt}: scatter disturbed the text rows"
