# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_decoder_inference(mesh_device, reset_seeds):
    model_args = ModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = model_args.reference_decoder()
    reference_model.load_state_dict(partial_state_dict)

    generation_length = 10

    seqlen = 1
    batch = model_args.max_batch_size

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            ttnn.L1_MEMORY_CONFIG,
        )

        dim = 2048

        attn_input_grid = ttnn.CoreGrid(y=2, x=8)
        mem_cfg = ttnn.create_sharded_memory_config(
            (
                32,
                dim // attn_input_grid.num_cores,
            ),
            attn_input_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Run TT model
        tt_out = ttnn.all_gather(
            decode_input, dim=3, num_links=1, topology=model_args.ccl_topology(), memory_config=mem_cfg
        )

        debug_max = lambda t: ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(model_args.mesh_device, dim=-1)
        ).max()
        logger.info(f"decode_input max: {debug_max(decode_input)=}, {decode_input.memory_config()=}")
        logger.info(f"tt_out max: {debug_max(tt_out)=}, {tt_out.memory_config()=}")
