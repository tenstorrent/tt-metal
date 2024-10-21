# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
import importlib

llama_reference_mod = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.model"
)
encoder_utils = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.encoder_utils"
)
from models.demos.llama3.tt.multimodal.llama_image_transformer import TtLlamaImageTransformer
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.llama3.tt.multimodal.llama_image_vision_encoder import pad_seq_one_tile, mask_tile_padding
from models.demos.llama3.tt.llama_common import (
    prepare_inputs_ttnn_prefill,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch, num_chunks, ntok",
    ((1, 4, 1024),),
)
@pytest.mark.parametrize(
    "is_global",
    (True, False),
)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(os.environ.get("FAKE_DEVICE"), None)],
    indirect=True,
)
def test_llama_image_transformer_inference(
    batch, num_chunks, ntok, mesh_device, is_global, use_program_cache, reset_seeds, ensure_gc
):
    dtype = ttnn.bfloat16
    pcc = 0.86

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    if is_global:
        first_layer_prefix = "vision_model.vision_encoder."
        gated = True
        n_layers = model_args.vision_n_global_layers
        return_intermediate = None
    else:
        # first_layer_prefix = "vision_model.vision_encoder.transformer."
        first_layer_prefix = "vision_model.vision_encoder."
        gated = False
        n_layers = model_args.vision_n_layers
        # return_intermediate = [int(l) for l in "3,7,15,23,30".split(",")]
        return_intermediate = list(range(n_layers))

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.vision_dim
    heads = model_args.vision_attn_n_heads

    reference_model = llama_reference_mod.VisionEncoder(
        max_num_tiles=4,
        image_size=model_args.vision_chunk_size,
        patch_size=model_args.vision_patch_size,
        n_global_layers=8,
        global_model=True,
        return_intermediate=return_intermediate,
    )
    reference_model.load_state_dict(partial_state_dict, strict=False)

    callable_reference = reference_model.transformer if not is_global else reference_model.global_transformer

    all_tests_pass = True

    tt_model = TtLlamaImageTransformer(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix + ("transformer." if not is_global else "global_transformer."),
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        layers=n_layers,
        gated=gated,
    )

    # Create PT input
    ar = torch.tensor([[2, 2]])
    pt_block_input = (torch.rand(batch, num_chunks, ntok, dim) * 2) - 1
    pt_block_input = pt_block_input.reshape(batch * num_chunks, ntok, dim)
    pt_block_input = reference_model.apply_class_embedding(pt_block_input)
    ntok += 1
    pt_block_input = pt_block_input.reshape(batch, num_chunks, ntok, dim)
    pt_block_input = reference_model.apply_positional_embedding(pt_block_input, ar)

    pt_block_input = reference_model.ln_pre(pt_block_input)

    tt_attention_input = pt_block_input.clone()
    # Do PT padding
    npad = 0
    pt_block_input, npad = encoder_utils.expand_num_tokens_to_mult8(pt_block_input)
    # Create PT attention mask
    mask = encoder_utils.build_encoder_attention_mask(pt_block_input, ar, ntok, num_chunks, 1)
    pt_block_input = pt_block_input.reshape(batch, -1, dim)

    attention_input = prepare_inputs_ttnn_prefill(
        tt_attention_input.view(num_chunks, ntok, dim),
        mesh_device,
    )
    # Pad TT input to multipple of 32
    attention_input, npadtt = pad_seq_one_tile(attention_input, mesh_device)
    # Create attention mask, assuming padding of 32
    fake_x = torch.zeros(
        attention_input.shape[0], attention_input.shape[1], attention_input.shape[2], attention_input.shape[3]
    )
    tt_attn_mask = encoder_utils.build_encoder_attention_mask(fake_x, ar, ntok, num_chunks, 1)
    # Make striped attention mask to mask out our padding between 8 and 32
    # Striped mask doesn't affect PCC
    tt_attn_mask = mask_tile_padding(tt_attn_mask, ntok, npadtt, num_chunks)

    attention_input = attention_input.reshape(1, batch, -1, dim)

    tt_mask = ttnn.from_torch(
        tt_attn_mask,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    with torch.no_grad():
        tt_out = tt_model(attention_input, return_intermediate=return_intermediate, mask=tt_mask)
        if return_intermediate:
            tt_out, tt_intermediates = tt_out
            tt_intermediates = [tt.reshape(batch, num_chunks, ntok + npadtt, dim) for tt in tt_intermediates]
            tt_intermediates = [ttnn.slice(tt, (0, 0, 0, 0), (batch, num_chunks, ntok, dim)) for tt in tt_intermediates]
            tt_intermed_torch = [
                ttnn.to_torch(tt_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
                for tt_intermediate in tt_intermediates
            ]

        tt_out = tt_out.reshape(batch, num_chunks, ntok + npadtt, dim)
        tt_out = ttnn.slice(tt_out, (0, 0, 0, 0), (batch, num_chunks, ntok, dim))
        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :, :]

        reference_output = callable_reference(pt_block_input, return_intermediate=return_intermediate, mask=mask)
        if return_intermediate:
            reference_output, intermediates = reference_output
            intermediates = intermediates.reshape(batch, num_chunks, ntok + npad, dim, -1)
            intermediates = intermediates[..., :ntok, :, :]
            intermediates = torch.chunk(intermediates, intermediates.shape[-1], dim=-1)
            intermediates = [i.squeeze(-1) for i in intermediates]
        reference_output = reference_output.reshape(batch, num_chunks, ntok + npad, dim)
        reference_output = encoder_utils.contract_num_tokens_from_mult8(reference_output, npad)
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)
        # Check mse

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if return_intermediate:
            for idx, (pt_interm, tt_interm) in enumerate(zip(intermediates, tt_intermed_torch)):
                passing, pcc_message = comp_pcc(pt_interm, tt_interm, pcc)
                logger.info(f"Intermediate {idx}: {pcc_message}")
                logger.info(comp_allclose(pt_interm, tt_interm))

        if passing:
            logger.info(f"Llama_Attention Passed!")
        else:
            logger.warning(f"Llama_Attention Failed!")
            all_tests_pass = False

        if all_tests_pass:
            logger.info("Llama Attention output Passed!")
        else:
            logger.warning("Llama Attention output Failed!")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
