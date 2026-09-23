# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torchvision.transforms as T
from loguru import logger
from transformers import Gemma4ImageProcessor

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.gemma4.tests.unit.test_vision_attention import convert_vision_block_hf_to_meta
from models.demos.gemma4.tt.vision.vision_encoder import VisionTransformer
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import standardize_hf_keys_multimodal


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_layers",
    [None, 1, 2],  # None means all layers, specific numbers will run fewer layers
    ids=["all_layers", "single_layer", "two_layers"],
)
@pytest.mark.parametrize(
    # image_grid_thw: The temporal, height and width of feature shape of each image in LLM.
    "token_budget, image_grid_chw",
    [
        (
            1120 * 9,
            [3, 110, 85],
        ),  # 300 DPI scanned doc with Letter paper (8.5x11 inches) has resolution around 2550x3300
        (
            280 * 9,
            [3, 50, 50],
        ),  # 240 DPI scanned doc with Letter paper (8.5x11 inches) has resolution around 2048x1300
    ],
    ids=["300dpi", "240dpi"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_model_inference(
    mesh_device,
    reset_seeds,
    ensure_gc,
    num_layers,
    is_ci_env,
    request,
    token_budget,
    image_grid_chw,
):
    test_id = request.node.callspec.id
    if is_ci_env and "two_layers" not in test_id:
        pytest.skip("CI only runs the two_layers test")

    dtype = ttnn.bfloat8_b
    # Set from measurement, not from another repo's number. Full depth against the
    # correctly masked reference measures ~0.99, so 0.98 is earned with margin.
    # The previous 0.91 was justified as "Llama 3 repo allows 0.91 for prefill,
    # vision probably even less sensitive to pcc" -- a guess, and it sat below a
    # real defect (no bidirectional padding mask) rather than above a result.
    pcc = 0.99 if num_layers and num_layers <= 3 else 0.98
    batch_size = 1  # For prefill we only support batch_size = 1
    ref_seq_len = image_grid_chw[1] * image_grid_chw[2]
    seq_len = ((token_budget // 2048) + 1) * 2048

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    if num_layers:
        model_args.hf_config.vision_config.num_hidden_layers = num_layers
        from transformers import logging as transformers_logging

        # Set logging level to ERROR to suppress warnings about unexpected keys
        transformers_logging.set_verbosity_error()
    else:
        num_layers = model_args.hf_config.vision_config.num_hidden_layers

    # Create reference model
    reference_model = model_args.reference_vision_model(depth=model_args.hf_config.vision_config.num_hidden_layers)
    state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
    state_dict = convert_vision_block_hf_to_meta(
        state_dict, model_args.n_heads, model_args.n_kv_heads, model_args.head_dim
    )
    state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}
    print(state_dict.keys())

    random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
    img = T.ToPILImage()(random_img)

    image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
    processed = image_processor(images=[img], max_soft_tokens=token_budget // 9, return_tensors="pt")
    pixel_position_ids = processed["image_position_ids"]  # [batch, num_patches, 2]
    pt_pixel_values = processed["pixel_values"]  # [batch, num_patches, in_dim]
    padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [batch, num_patches]
    ref_seq_len = pixel_position_ids.shape[1]
    logger.info(f"num_patches={ref_seq_len}, valid_patches={(~padding_positions).sum().item()}")

    # Reference: patch embed -> encoder. Patch embedding lives in VisionTower now, so the encoder
    # (both reference and TT) is fed already-embedded patch hidden states.
    inputs_embeds = reference_model.patch_embedder(pt_pixel_values, pixel_position_ids, padding_positions)
    reference_output = reference_model.encoder(
        inputs_embeds=inputs_embeds,
        pixel_position_ids=pixel_position_ids,
        # As Gemma4VisionModel does. Previously None, to match a TT encoder that
        # applied no padding mask; that handicap hid the defect (see the tower test).
        attention_mask=~padding_positions,
    ).last_hidden_state

    # Initialize TT model (rotary + transformer blocks, all on device)
    tt_model = VisionTransformer(
        args=model_args,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
    )

    # Upload the (host-computed) patch embeddings + position metadata as ttnn tensors. The encoder
    # no longer performs patch embedding, so it is fed already-embedded hidden states directly.
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_inputs_embeds = ttnn.from_torch(
        inputs_embeds.unsqueeze(0),  # [1, batch, num_patches, hidden]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    tt_position_ids = ttnn.from_torch(
        pixel_position_ids.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )

    # Run TT model
    tt_out = tt_model(
        tt_inputs_embeds,
        tt_position_ids,
        unpadded_seq_len=ref_seq_len,
        seq_len=seq_len,
        # Real patch count, so the encoder applies the same bidirectional padding
        # mask the reference gets above. Note this is a DIFFERENT padding from
        # ``unpadded_seq_len``: that is tile padding (num_patches -> seq_len),
        # this is the (-1,-1) image padding inside num_patches.
        num_valid_patches=int((~padding_positions).sum().item()),
    )

    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
    )

    tt_output_torch = tt_out[:, 0:1, :, : model_args.hf_config.vision_config.hidden_size].squeeze(0).squeeze(0)

    # Compare the VALID patches only. Padded rows are masked out of attention on
    # both sides, so what they contain is undefined and differs between the two
    # implementations -- and ``Gemma4VisionModel`` discards them anyway via the
    # pooler mask, exactly as the tower test does. Including them would compare
    # values no consumer ever reads.
    valid = ~padding_positions.squeeze(0)  # [num_patches], True = real patch
    ref_valid = reference_output.squeeze()[valid]
    tt_valid = tt_output_torch[valid]
    passing, pcc_message = comp_pcc(ref_valid, tt_valid, pcc)
    logger.info(comp_allclose(ref_valid, tt_valid))
    logger.info(f"PCC of output: {pcc_message}")

    # Generate test summary message
    test_desc = f"Vision Transformer Model ({num_layers} layers)"

    if passing:
        logger.info(f"{test_desc} Passed!")
    else:
        logger.warning(f"{test_desc} Failed! PCC value is lower than {pcc} for some of the outputs. Check Warnings!")
    assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
