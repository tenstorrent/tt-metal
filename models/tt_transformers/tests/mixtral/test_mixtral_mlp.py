# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.reference.model import FeedForward, RMSNorm
from models.tt_transformers.tt.mixtral_mlp import TtMixtralMLP
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc
from ttnn import ConcatMeshToTensor

# pytest models/tt_transformers/tests/mixtral/test_mixtral_mlp.py::test_mixtral_mlp_inference[wormhole_b0-True-prefill]


def convert2ref(state_dict):
    out = {}
    for key, value in state_dict.items():
        out[key[4:]] = value
    return out


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_mixtral_mlp_inference(t3k_mesh_device, reset_seeds, mode):
    seqlen = 32
    t3k_mesh_device.disable_and_clear_program_cache()
    dtypes = {
        "w1": ttnn.bfloat8_b,
        "w2": ttnn.bfloat8_b,
        "w3": ttnn.bfloat8_b,
    }

    model_args = ModelArgs(t3k_mesh_device)
    model_args.n_layers = 32
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", 0)

    tt_model = TtMixtralMLP(
        mesh_device=t3k_mesh_device, state_dict=state_dict, args=model_args, layer_num=0, dtypes=dtypes
    )

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k: v for k, v in state_dict.items() if (k.startswith("layers.0.") and "attention" not in k and "norm" not in k)
    }
    partial_state_dict_ref = {k[32:]: v for k, v in partial_state_dict.items() if f"experts.{0}" in k}
    reference_model = FeedForward(model_args)
    reference_model.load_state_dict(convert2ref(partial_state_dict_ref))

    rms_state_dict = {k[18:]: v for k, v in state_dict.items() if (k.startswith("layers.0.ffn_norm."))}
    rms = RMSNorm(dim=model_args.dim)
    rms.load_state_dict(rms_state_dict)

    original_input = (torch.rand(1, 1, seqlen, model_args.dim) * 2) - 1
    torch_input = rms(original_input)  # apply rmsnorm to input

    reference_output = reference_model(torch_input)

    if mode == "decode":
        shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}  # 8 cores: x=0 to x=7, y=0
        )

        shard_shape = [seqlen, 512]

        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )

        tt_input = ttnn.as_tensor(
            torch_input,
            dtype=ttnn.bfloat16,  # or your desired dtype
            layout=ttnn.TILE_LAYOUT,
            device=t3k_mesh_device,
            memory_config=width_sharded_mem_config,
        )
    else:
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,  # or your desired dtype
            layout=ttnn.TILE_LAYOUT,
            device=t3k_mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt_output = tt_model(tt_input, mode)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)
    if passing:
        logger.info(f"Mixtral_MLP {mode} Passed!")
    else:
        logger.warning(f"Mixtral_MLP {mode} Failed!")

    assert passing, f"Mixtral_MLP {mode} output does not meet PCC requirement {pcc_required}: {pcc_message}."
