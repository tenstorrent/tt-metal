# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Grok flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["GROK_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.experimental.grok.tt.grok_mlp import TtGrokMLP
from models.experimental.grok.tt.grok_moe import TtMoeLayer
from models.experimental.grok.reference.model import MoeBlock
from models.experimental.grok.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.timeout(600)
def test_grok_moe_inference(t3k_mesh_device, use_program_cache, reset_seeds):
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)
    pcc = 0.87  # real weights = 0.99
    iterations = 1
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_mesh_device.get_device(0), dummy_weights=os.getenv("CI") == "true")
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    key_start = "model.layers.0.moe_block."
    partial_state_dict = {k[len(key_start) :]: v for k, v in state_dict.items() if (k.startswith(key_start))}
    reference_model = MoeBlock(
        hidden_dim=model_args.hidden_size,
        ffn_dim=model_args.intermediate_size,
        num_experts=model_args.num_experts,
        top_k=model_args.num_experts_per_tok,
    )
    reference_model.load_state_dict(partial_state_dict)

    # Initialize TT models
    experts = TtGrokMLP(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtypes={
            "linear": ttnn.bfloat4_b,
            "linear_1": ttnn.bfloat8_b,
            "linear_v": ttnn.bfloat4_b,
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

        # input = torch.randn(1, 32, 6144)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.hidden_size) * 2) - 1
        tt_decode_input = ttnn.from_torch(
            pt_decode_input.clone().unsqueeze(1).view(1, 1, 32, model_args.hidden_size),
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
        ref_output, unused_router_logits = reference_model(pt_decode_input)
        print(f"ref_output: {ref_output.shape}")
        print(f"tt_output_torch: {tt_output_torch.shape}")
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Grok MOE Passed!")
        else:
            logger.warning("Grok MOE Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {iterations} Grok MOE iterations Passed!")
    else:
        logger.warning("One or more iterations of Grok MOE Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
