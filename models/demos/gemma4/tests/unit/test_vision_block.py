# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torchvision.transforms as T
from loguru import logger
from transformers import Gemma4ImageProcessor

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.gemma4.tests.unit.test_vision_attention import (
    convert_rope_style_hf_to_meta_md,
    convert_vision_block_hf_to_meta,
)
from models.demos.gemma4.tt.vision.vision_block import VisionBlock
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_rot_transformation_mat
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
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_block_inference(
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    n_layers = 27
    dtype = ttnn.bfloat16
    pccs = [0.97] * n_layers
    pccs[24:] = [0.85] * (n_layers - 24)
    print(pccs)
    batch_size = 1  # For prefill we only support batch_size = 1

    # for this test assume 1 image of size 84 x 112
    image_grid_chw = [3, 30, 40]
    ref_seq_len = 1120 * 9
    # pad seq_len to be divisible by base_model_args.MAX_QKV_MM_SEQ_LEN from the tt_transformers model
    seq_len = ((ref_seq_len // 2048) + 1) * 2048  # Using 128 as MAX_QKV_MM_SEQ_LEN

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    reference_whole_model = model_args.reference_vision_model()

    all_passing = True
    for layer_num in range(n_layers):
        logger.info(f"Testing layer {layer_num}")
        pcc = pccs[layer_num]

        reference_model = reference_whole_model.encoder.layers[layer_num]

        # Get the state dict of the reference model.
        # Use the vision-specific block converter: it applies the per-block (2D) RoPE
        # permute to the attention q/k weights and norms and renames only the projection
        # submodules, preserving the HF container/norm names (self_attn, input_layernorm,
        # post_attention_layernorm, ...) that the Gemma-4 vision modules read. The stock
        # convert_hf_to_meta would (a) skip the q/k permute and (b) apply LLM renames
        # (self_attn->attention, input_layernorm->attention_norm), breaking key lookups.
        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
        state_dict = convert_vision_block_hf_to_meta(
            state_dict,
            model_args.n_heads,
            model_args.n_kv_heads,
            model_args.head_dim,
        )
        state_dict_prefix = model_args.get_state_dict_prefix("VisionBlock", layer_num)
        state_dict = {f"{state_dict_prefix}{k}": v for k, v in state_dict.items()}
        print(state_dict.keys())

        # Example inputs and preprocessing
        pt_input = torch.randn(1, 1, ref_seq_len, model_args.dim, dtype=torch.bfloat16)
        random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
        img = T.ToPILImage()(random_img)

        print("model_args.head_dim", model_args.head_dim)
        image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
        processed = image_processor(images=[img], max_soft_tokens=1120, return_tensors="pt")
        pixel_position_ids = processed["image_position_ids"]
        position_embeddings = model_args.reference_vision_model().encoder.rotary_emb(pt_input, pixel_position_ids)

        # pre-compute the rotational embedding matrix and send to device
        cos, sin = position_embeddings
        cos, sin = convert_rope_style_hf_to_meta_md(cos, sin)
        cos = torch.nn.functional.pad(cos, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0)
        sin = torch.nn.functional.pad(sin, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0)
        cos = ttnn.from_torch(
            cos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        sin = ttnn.from_torch(
            sin,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        rot_mats = [cos, sin]

        transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)

        transformation_mats_prefill = ttnn.as_tensor(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        transformation_mats = {"prefill": transformation_mats_prefill}

        # Initialize TT model
        tt_model = VisionBlock(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=None,  # Don't cache random weights
            layer_num=layer_num,
            tt_ccl=TT_CCL(mesh_device),
            dtype=dtype,
            transformation_mats=transformation_mats,
            args=model_args,
        )

        # Prepare input tensor for the TT model
        tt_input = pt_input.clone()
        tt_input = torch.nn.functional.pad(tt_input, (0, 0, 0, seq_len - ref_seq_len))
        attention_input = model_args.prepare_residual_tensor_prefill(tt_input.view(1, seq_len, -1))

        # Run our model
        print(cos.shape, sin.shape)
        tt_out = tt_model(
            attention_input,
            rot_mats=rot_mats,
        )

        # Process the output
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
        )
        tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(batch_size, seq_len, -1)  # [batch, seq, hidden_dim]

        # Remove sequence padding
        tt_output_torch = tt_output_torch[0, :ref_seq_len, :]

        # Run reference model
        print(pt_input.shape, position_embeddings[0].shape, pixel_position_ids.shape)
        reference_output = reference_model(
            pt_input.squeeze(0),
            position_embeddings=position_embeddings,
            position_ids=pixel_position_ids,
        )

        # Compare outputs
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"Vision Block Layer {layer_num} Passed!")
        else:
            logger.warning(f"Vision Block Layer {layer_num} Failed! PCC: {pcc_message} lower than {pcc}")
            all_passing = False

    assert all_passing, "PCC value is lower than expected for some of the outputs. Check Layer-specific Warnings!"
