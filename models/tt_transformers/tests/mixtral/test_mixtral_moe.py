# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.mixtral_mlp import TtMixtralMLP
from models.tt_transformers.tt.mixtral_moe import TtMoeLayer
from models.tt_transformers.tt.model_config import ModelArgs

from .utils import load_hf_mixtral_config

# pytest models/tt_transformers/tests/mixtral/test_mixtral_moe.py


def convert2ref(state_dict):
    out = {}
    for key, value in state_dict.items():
        if "moe.experts" in key:
            new_key = key.replace("moe.experts", "experts")
            out[new_key] = value
        elif "moe.gate" in key:  # ensure we don’t duplicate/overwrite
            new_key = key.replace("moe.gate", "gate")
            out[new_key] = value
    return out


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_mixtral_moe_inference(mesh_device, reset_seeds, mode, device_params):
    pcc = 0.99
    iterations = 1
    dtype = ttnn.bfloat8_b
    mesh_device.disable_and_clear_program_cache()
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()
    model_args.n_layers = 1
    layer_num = 0

    hf_config = load_hf_mixtral_config()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k: v for k, v in state_dict.items() if (k.startswith("layers.0.") and "attention" not in k and "norm" not in k)
    }
    partial_state_dict_ref = {k[32:]: v for k, v in partial_state_dict.items() if f"experts.{0}" in k}

    partial_state_dict_ref = {k[22:]: v for k, v in partial_state_dict.items()}

    reference_model = MixtralSparseMoeBlock(hf_config)
    reference_model.load_state_dict(convert2ref(partial_state_dict_ref))
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtMoeLayer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        experts=TtMixtralMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            args=model_args,
            layer_num=layer_num,
            dtypes={
                "w1": dtype,
                "w2": dtype,
                "w3": dtype,
            },
        ),
        args=model_args,
        layer_num=layer_num,
        dtype=dtype,
        tt_ccl=tt_ccl,
    )

    all_tests_pass = True

    seqlen = 1
    batch = 32

    for i in range(iterations):
        logger.info(f"[Decoder] Generating token {i}")

        pt_decode_input = (torch.rand(seqlen, batch, model_args.dim) * 2) - 1

        if mode == "decode":
            memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                    [32, 512],  # shard shape
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            tt_decode_input = ttnn.as_tensor(
                pt_decode_input,
                dtype=ttnn.bfloat16,  # or your desired dtype
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=memory_config,
            )

        else:
            tt_decode_input = ttnn.from_torch(
                pt_decode_input,
                dtype=ttnn.bfloat16,  # or your desired dtype
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # TT Model Output
        logger.info(f"Starting TT Mixtral MOE {mode}")
        tt_out = tt_model(tt_decode_input, mode)
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))[0]
            .squeeze(0)
            .view(seqlen, batch, -1)
        )
        # Reference Model Output
        logger.info(f"Starting Reference MOE {mode}")
        ref_output, _ = reference_model(pt_decode_input)
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"TT Mixtral MOE {mode} Passed!")
        else:
            logger.warning(f"TT Mixtral MOE {mode} Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {iterations} Mixtral MOE {mode} iterations Passed!")
    else:
        logger.warning(f"One or more iterations of Mixtral MOE {mode} Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
