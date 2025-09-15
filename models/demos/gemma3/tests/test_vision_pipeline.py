# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from models.demos.gemma3.tt.gemma_vision_model import TtSiglipGemmaVisionModel
from models.demos.gemma3.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("bsz", [1])
def test_gemma_vision(
    mesh_device,
    reset_seeds,
    bsz,
):
    pcc_required = 0.95
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "model.vision_tower.vision_model."

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    input_tensor = torch.rand((bsz, in_channels, image_size, image_size))

    reference_model = model_args.reference_vision_model()
    reference_output = reference_model(input_tensor).last_hidden_state

    test_gemma_vision = TtSiglipGemmaVisionModel(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )
    test_output = test_gemma_vision(input_tensor)

    logger.info("Checking outputs")
    out = ttnn.from_device(test_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :, :]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("bsz", [1])
def test_gemma_vision_piecewise(
    mesh_device,
    reset_seeds,
    bsz,
):
    from models.demos.gemma3.tt.gemma_image_transformer import TtGemmaImageTransformer
    from models.demos.gemma3.tt.siglip_vision_embedding import TtSiglipVisionEmbeddings
    from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm

    pcc_required = 0.99
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "model.vision_tower.vision_model."

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    input_tensor = torch.rand((bsz, in_channels, image_size, image_size))

    reference_model = model_args.reference_vision_model()
    reference_output = reference_model(input_tensor).last_hidden_state
    tt_ccl = TT_CCL(mesh_device)
    test_gemma_vision = TtSiglipGemmaVisionModel(
        mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )
    test_output = test_gemma_vision(input_tensor)

    test_gemma_vision_embeddings = TtSiglipVisionEmbeddings(
        mesh_device,
        state_dict=state_dict,
        state_dict_prefix=f"{first_layer_prefix}embeddings.",
        dtype=dtype,
        image_size=model_args.vision_chunk_size,
        patch_size=model_args.vision_patch_size,
        num_channels=model_args.vision_in_channels,
        hidden_dim=model_args.vision_dim,
        bias=True,
    )
    reference_vision_embeddings_output = reference_model.embeddings(input_tensor)
    test_vision_embeddings_output = test_gemma_vision_embeddings(input_tensor)

    test_gemma_vision_encoder = TtGemmaImageTransformer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        tt_ccl=tt_ccl,
        state_dict_prefix=f"{first_layer_prefix}encoder.",
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        layers=model_args.vision_n_layers,
        block_key="layers",
    )
    attention_mask = torch.zeros(bsz, 1, test_vision_embeddings_output.shape[1], test_vision_embeddings_output.shape[1])

    tt_mask = ttnn.from_torch(
        attention_mask,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    reference_vision_encoder_output = reference_model.encoder(reference_vision_embeddings_output)
    test_vision_encoder_output = test_gemma_vision_encoder(test_vision_embeddings_output, mask=tt_mask)

    test_gemma_vision_ln_post = TtLayerNorm(
        device=mesh_device,
        dim=model_args.vision_dim,
        state_dict=state_dict,
        state_dict_prefix=f"{first_layer_prefix}ln_post.",
        weight_dtype=dtype,
        eps=model_args.norm_eps,
    )
    reference_vision_ln_post_output = reference_model.post_layernorm(reference_vision_encoder_output.last_hidden_state)
    test_vision_ln_post_output = test_gemma_vision_ln_post(test_vision_encoder_output)

    logger.info("Checking outputs")

    def compare_outputs(reference_output, test_output, test_name):
        out = ttnn.from_device(test_output)
        tt_output_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"{test_name} PCC: {pcc_message}")

    compare_outputs(reference_vision_embeddings_output, test_vision_embeddings_output, "embeddings")
    compare_outputs(reference_vision_encoder_output.last_hidden_state, test_vision_encoder_output, "encoder")
    compare_outputs(reference_vision_ln_post_output, test_vision_ln_post_output, "ln_post")
    compare_outputs(reference_output, test_output, "gemma_vision")
    # assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
