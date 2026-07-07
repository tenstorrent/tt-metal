# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Hybrid end-to-end PCC test for Janus-Pro.

The TT vision tower + aligner run on the device; the LLaMA-style decoder runs as a
torch/CPU reference. This isolates the vision port's error from the decoder's ttnn
port: a single HF ``JanusForConditionalGeneration`` instance provides the decoder for
BOTH the golden path and the hybrid path, so the only difference between them is the
vision features (TT vs torch). We reuse HF Janus' own image-token fusion
(``masked_scatter`` on ``image_token_id``), the same pattern gemma's e2e model uses.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.janus_pro.tt.janus_pro_vision_model import TtJanusProTransformerVision
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_e2e_hybrid(mesh_device, reset_seeds):
    pcc_required = 0.95
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device)
    model_args.WEIGHTS_DTYPE = dtype
    state_dict = model_args.load_state_dict()

    # Single HF instance shared by both paths -> identical decoder, so the PCC delta
    # is purely TT vision+aligner vs torch vision+aligner.
    hf_model = model_args.reference_vision_transformer(wrap=False)
    hf_model.eval()

    image_token_id = hf_model.config.image_token_id
    num_image_tokens = model_args.mm_tokens_per_image

    # ---- inputs: random image + a small prompt with exactly num_image_tokens placeholders ----
    bsz = 1
    in_channels = model_args.vision_in_channels
    image_size = model_args.vision_chunk_size
    pixel_values = torch.rand((bsz, in_channels, image_size, image_size))

    text_prefix = torch.tensor([[1, 100, 200]], dtype=torch.long)
    image_block = torch.full((bsz, num_image_tokens), image_token_id, dtype=torch.long)
    text_suffix = torch.tensor([[300, 400]], dtype=torch.long)
    input_ids = torch.cat([text_prefix, image_block, text_suffix], dim=1)
    assert (
        int((input_ids == image_token_id).sum()) == num_image_tokens
    ), "Number of image placeholder tokens must equal the number of image features."

    # ---- reference (full torch): HF runs vision + aligner + masked_scatter + decoder ----
    with torch.no_grad():
        ref_logits = hf_model(input_ids=input_ids, pixel_values=pixel_values.float()).logits

    # ---- hybrid: TT vision tower (vision encoder + aligner) on device ----
    # The wrapper runs vision_model -> aligner on device, so there is no host
    # round-trip between them; the aligner input stays in tile layout.
    tt_ccl = TT_CCL(mesh_device)
    tt_tower = TtJanusProTransformerVision(
        mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        state_dict_prefix="model.",
        dtype=dtype,
        configuration=model_args,
    )
    image_feat_tt = tt_tower(pixel_values)
    image_features = ttnn.to_torch(
        ttnn.from_device(image_feat_tt), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0]
    image_features = image_features.reshape(-1, model_args.vision_projection_dim)

    # ---- fuse TT image features into text embeddings, run torch decoder (HF/gemma pattern) ----
    with torch.no_grad():
        inputs_embeds = hf_model.get_input_embeddings()(input_ids)
        mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, image_features.to(inputs_embeds.dtype))
        hybrid_logits = hf_model(inputs_embeds=inputs_embeds, pixel_values=None).logits

    passing, pcc_message = comp_pcc(ref_logits, hybrid_logits, pcc_required)
    logger.info(comp_allclose(ref_logits, hybrid_logits))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required}. Check Warnings!"
