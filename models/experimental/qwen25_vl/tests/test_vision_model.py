"""Test for Qwen 2.5 VL Vision Transformer Pretrained Model Inference"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from models.tt_transformers.tt.model_config import ModelArgs

from models.experimental.qwen25_vl.tt.vision_model import TtQwen2_5_VisionTransformerPretrainedModel
from models.utility_functions import comp_pcc, skip_for_grayskull, comp_allclose


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch, num_chunks",
    ((1, 4),),
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
def test_vision_inference(batch, num_chunks, mesh_device, reset_seeds):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "visual."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.vision_dim

    reference_model = model_args.reference_vision_model()
    reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    n_layers = model_args.vision_n_layers

    tt_model = TtQwen2_5_VisionTransformerPretrainedModel(
        mesh_device,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        model_args=model_args,
        dtype=dtype,
        layers=n_layers,
    )

    pt_input = torch.randn([32, 1176])  # no batch dim
    grid_thw = torch.tensor([[1, 4, 8]])

    reference_output = reference_model(
        pt_input,
        grid_thw,
    )

    tt_attention_input = model_args.prepare_residual_tensor_prefill(pt_input.unsqueeze(0), force_replicated=True)
    tt_out = tt_model(tt_attention_input, grid_thw)

    tt_output_torch = ttnn.to_torch(tt_out, device=mesh_device)

    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
