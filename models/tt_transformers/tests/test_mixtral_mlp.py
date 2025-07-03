import torch
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.reference.model import FeedForward, RMSNorm
from models.tt_transformers.tt.mixtral_mlp import TtMixtralMLP
from models.tt_transformers.tt.model_config import ModelArgs
from ttnn import ConcatMeshToTensor

# pytest -svv models/tt_transformers/tests/test_mixtral_mlp.py::test_mixtral_mlp_inference[wormhole_b0-True]


def test_mixtral_mlp_inference(t3k_mesh_device, use_program_cache, reset_seeds):
    dtypes = {
        "w1": ttnn.bfloat4_b,
        "w2": ttnn.bfloat8_b,
        "w3": ttnn.bfloat4_b,
    }

    model_args = ModelArgs(t3k_mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)

    tt_model = TtMixtralMLP(
        mesh_device=t3k_mesh_device, state_dict=state_dict, args=model_args, layer_num=0, dtypes=dtypes
    )

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k: v for k, v in state_dict.items() if (k.startswith("layers.0.") and "attention" not in k and "norm" not in k)
    }
    partial_state_dict_ref = {k[32:]: v for k, v in partial_state_dict.items() if f"experts.{0}" in k}

    reference_model = FeedForward(model_args)
    reference_model.load_state_dict(partial_state_dict_ref)

    rms_state_dict = {k[18:]: v for k, v in state_dict.items() if (k.startswith("layers.0.ffn_norm."))}
    rms = RMSNorm(dim=model_args.dim)
    rms.load_state_dict(rms_state_dict)

    torch_input = (torch.rand(1, 1, 32, model_args.dim) * 2) - 1
    torch_input = rms(torch_input)  # apply rmsnorm to input

    reference_output = reference_model(torch_input)
    breakpoint()
    tt_input = ttnn.from_torch(
        torch_input,
        device=t3k_mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_mesh_device, dim=0),
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)
    if passing:
        logger.info("Mixtral_MLP Passed!")
    else:
        logger.warning("Mixtral_MLP Failed!")

    assert passing, f"Mixtral_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
