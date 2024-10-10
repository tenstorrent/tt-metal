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
from models.demos.wormhole.llama31_8b_N300.tt.llama_image_transformer import TtLlamaImageTransformer
from models.demos.wormhole.llama31_8b_N300.tt.model_config import TtModelArgs
from models.demos.wormhole.llama31_8b_N300.tt.llama_common import (
    prepare_inputs_ttnn_prefill,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (5 * 1024,),
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
def test_llama_image_transformer_inference(seq_len, mesh_device, is_global, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16
    pcc = 0.99

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    if is_global:
        first_layer_prefix = "vision_model.vision_encoder.global_transformer."
        gated = True
        n_layers = model_args.vision_n_global_layers
        return_intermediate = None
    else:
        first_layer_prefix = "vision_model.vision_encoder.transformer."
        gated = False
        n_layers = model_args.vision_n_layers
        # return_intermediate = [int(l) for l in "3,7,15,23,30".split(",")]
        return_intermediate = list(range(n_layers))

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.vision_dim
    heads = model_args.vision_attn_n_heads

    reference_model = llama_reference_mod.ImageTransformer(
        width=dim, layers=n_layers, heads=heads, mlp_ratio=model_args.vision_mlp_ratio, gated=gated
    )
    reference_model.load_state_dict(partial_state_dict)

    batch = 1

    all_tests_pass = True

    tt_model = TtLlamaImageTransformer(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        layers=n_layers,
        gated=gated,
    )

    # pt_block_input = (torch.rand(batch, seq_len, dim) * 2) - 1
    pt_block_input = torch.load(
        "/home/cglagovich/tt-metal/models/demos/t3000/llama2_70b/reference/llama-models/image_transformer_32L_x.pt"
    )
    # pt_block_input = pt_block_input[..., :seq_len, :].bfloat16().float()
    pt_block_input = pt_block_input.bfloat16().float()
    pt_block_input = torch.nn.functional.pad(pt_block_input, (0, 0, 0, seq_len - pt_block_input.shape[-2]))
    mask = torch.load(
        "/home/cglagovich/tt-metal/models/demos/t3000/llama2_70b/reference/llama-models/image_transformer_32L_mask.pt"
    )
    # mask = mask[..., :seq_len, :seq_len]
    mask = torch.nn.functional.pad(mask, (0, seq_len - mask.shape[-1], 0, seq_len - mask.shape[-2]), value=-1e9)
    tt_block_input = pt_block_input.clone()
    block_input = prepare_inputs_ttnn_prefill(
        tt_block_input,
        mesh_device,
    )

    # mask = torch.bernoulli(
    #     torch.full(
    #         (
    #             batch,
    #             seq_len,
    #             seq_len,
    #         ),
    #         0.25,
    #     )
    # )
    # mask = mask.unsqueeze(1)
    # mask = mask * -1e9

    tt_mask = ttnn.from_torch(
        mask,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_model(block_input, return_intermediate=return_intermediate, mask=tt_mask)
    if return_intermediate:
        tt_out, tt_intermediates = tt_out
        tt_intermed_torch = [
            ttnn.to_torch(tt_intermediate, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[:, 0, :, :].view(
                batch, seq_len, -1
            )
            for tt_intermediate in tt_intermediates
        ]

    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[:, 0, :, :].view(
        batch, seq_len, -1
    )  # [ batch, seq, hidden_dim]

    reference_output = reference_model(pt_block_input, return_intermediate=return_intermediate, mask=mask)
    if return_intermediate:
        reference_output, intermediates = reference_output
        intermediates = torch.chunk(intermediates, intermediates.shape[-1], dim=-1)
        intermediates = [i.squeeze(-1) for i in intermediates]
    passing, pcc_message = comp_pcc(reference_output[..., :4120, :], tt_output_torch[..., :4120, :], pcc)

    logger.info(comp_allclose(reference_output[..., :4120, :], tt_output_torch[..., :4120, :]))
    logger.info(pcc_message)

    if return_intermediate:
        for idx, (pt_interm, tt_interm) in enumerate(zip(intermediates, tt_intermed_torch)):
            passing, pcc_message = comp_pcc(pt_interm[..., :4120, :], tt_interm[..., :4120, :], pcc)
            logger.info(f"Intermediate {idx}: {pcc_message}")
            logger.info(comp_allclose(pt_interm[..., :4120, :], tt_interm[..., :4120, :]))
            # if not passing:
            # break
            if idx == 31:
                torch.save(pt_interm, "layer_31_intermediate.pt")
                torch.save(mask, "mask.pt")
            # if idx == 31:
            #     breakpoint()

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
