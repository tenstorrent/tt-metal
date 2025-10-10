# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from llama_models.llama3.reference_impl.multimodal import encoder_utils
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc  # , skip_for_grayskull
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_image_transformer import TtLlamaImageTransformer
from models.tt_transformers.tt.multimodal.llama_vision_encoder import mask_tile_padding, pad_seq_one_tile


# @skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch, num_chunks",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "is_global",
    (True, False),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_image_transformer_inference(batch, num_chunks, mesh_device, is_global):
    pcc_required = 0.75
    model_args = ModelArgs(mesh_device)
    dtype = ttnn.bfloat16
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    n_layers = model_args.vision_n_layers
    n_global_layers = model_args.vision_n_global_layers
    first_layer_prefix = "vision_model.vision_encoder."
    if is_global:
        gated = True
        return_intermediate = None
    else:
        gated = False
        # Checks all intermediates
        return_intermediate = list(range(n_layers))

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.vision_dim
    ntok = model_args.vision_chunk_ntok - 1  # NOTE: -1 to remove class embedding

    from transformers import MllamaForConditionalGeneration

    weights_path = os.getenv("HF_MODEL")
    # the next two lines are memoery intensive and create big overhead
    # model = MllamaForConditionalGeneration(config).to(torch.bfloat16)
    # model.model.vision_model.load_state_dict(partial_state_dict, strict=False)

    model = MllamaForConditionalGeneration.from_pretrained(
        weights_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, use_safetensors=True
    )  # config=config,
    reference_model = model.model.vision_model.eval()
    callable_reference = reference_model.transformer if not is_global else reference_model.global_transformer
    all_tests_pass = True

    # The next weight assingment increases pcc similarity metric
    for id_b, block in enumerate(callable_reference.layers):
        callable_reference.layers[id_b].self_attn.q_proj.weight = torch.nn.Parameter(
            partial_state_dict["transformer.resblocks.{}.attn.wq.weight".format(id_b)]
        )
        callable_reference.layers[id_b].self_attn.k_proj.weight = torch.nn.Parameter(
            partial_state_dict["transformer.resblocks.{}.attn.wk.weight".format(id_b)]
        )

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtLlamaImageTransformer(
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix=first_layer_prefix + ("transformer." if not is_global else "global_transformer."),
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        layers=n_layers if not is_global else n_global_layers,
        gated=gated,
    )

    ar = torch.tensor([[2, 2]] * batch)
    pt_block_input = (torch.rand(batch, num_chunks, ntok, dim, dtype=torch.bfloat16) - 0.5) / 100
    tt_attention_input = pt_block_input.clone()

    # Create PT attention mask
    mask = encoder_utils.build_encoder_attention_mask(pt_block_input, ar, ntok, num_chunks, 1)
    pt_block_input = pt_block_input.reshape(batch, -1, dim)

    attention_input = model_args.prepare_residual_tensor_prefill(
        tt_attention_input.view(num_chunks, ntok, dim),
        force_replicated=True,
    )
    # Pad TT input to multipple of 32
    attention_input, npadtt = pad_seq_one_tile(attention_input, mesh_device)
    # Create attention mask, assuming padding of 32
    fake_x = torch.zeros(
        attention_input.shape[0], attention_input.shape[1], attention_input.shape[2], attention_input.shape[3]
    )
    tt_attn_mask = encoder_utils.build_encoder_attention_mask(fake_x, ar, ntok, num_chunks, 1)
    # Make striped attention mask to mask out our padding between 8 and 32
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
        tt_out = tt_model(attention_input, return_intermediate=return_intermediate, mask=None)
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

        tens_input = ttnn.to_torch(attention_input, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
            :, :, :, :
        ].reshape(batch * num_chunks, ntok + npadtt, dim)

        feats = callable_reference(tens_input[:, :ntok, :], attention_mask=None, output_hidden_states=False)
        reference_output = feats.last_hidden_state
        if return_intermediate != None:
            intermediates = [tens_input[:, :ntok, :]]
            for l in range(n_layers):
                intermediates.append(callable_reference.layers[l](tt_intermed_torch[l])[0])
            reference_output = intermediates[n_layers]
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        if not passing:
            logger.warning(f"PCC value -- {pcc_message} -- is lower than {pcc_required} for the output.")
        else:
            logger.info(f"PCC: {pcc_message}")
        logger.info(comp_allclose(reference_output, tt_output_torch))

        all_tests_pass = all_tests_pass and passing
        if return_intermediate:
            for idx, (pt_interm, tt_interm) in enumerate(zip(intermediates, tt_intermed_torch)):
                passing, pcc_message = comp_pcc(pt_interm, tt_interm, pcc_required)
                logger.info(f"Intermediate {idx}: {pcc_message}")
                logger.info(comp_allclose(pt_interm, tt_interm))
                all_tests_pass = all_tests_pass and passing

        assert all_tests_pass, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
