# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.reference.model import FeedForward
from models.demos.t3000.mixtral8x7b.reference.moe import MoeLayer
from models.demos.t3000.mixtral8x7b.tt.mixtral_mlp import TtMixtralMLP
from models.demos.t3000.mixtral8x7b.tt.mixtral_moe import TtMoeLayer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_allclose, comp_pcc
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh


def test_mixtral_moe_inference(t3k_mesh_device, use_program_cache, reset_seeds):
    pcc = 0.99
    iterations = 1
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[9:]: v
        for k, v in state_dict.items()
        if (k.startswith("layers.0.") and "attention" not in k and "norm" not in k)
    }

    partial_state_dict_ref = {k[13:]: v for k, v in partial_state_dict.items()}
    reference_model = MoeLayer(
        experts=[FeedForward(args=model_args) for _ in range(8)],
        gate=torch.nn.Linear(model_args.dim, 8, bias=False),
        moe_args=model_args,
    )
    reference_model.load_state_dict(partial_state_dict_ref)
    # Initialize TT models

    experts = TtMixtralMLP(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtypes={
            "w1": ttnn.bfloat4_b,
            "w2": ttnn.bfloat8_b,
            "w3": ttnn.bfloat4_b,
        },
    )

    tt_model = TtMoeLayer(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        experts=experts,
        args=model_args,
        layer_num=0,
        dtype=dtype,
    )

    all_tests_pass = True

    seqlen = 1
    batch = 32

    # TODO Update start_pos (check llama test for reference)
    for i in range(iterations):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = ttnn.from_torch(
            pt_decode_input.clone().unsqueeze(1).view(1, 1, 32, 4096),
            device=t3k_mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
        )

        # Run TT model
        tt_out = tt_model(tt_decode_input)
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
            .squeeze(2)
            .view(batch, 1, -1)
        )

        # Reference model
        ref_output = reference_model(pt_decode_input)
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral MOE Passed!")
        else:
            logger.warning("Mistral MOE Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {iterations} Mistral MOE iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral MOE Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
