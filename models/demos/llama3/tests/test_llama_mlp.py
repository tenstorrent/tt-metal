# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward
from models.utility_functions import comp_pcc, comp_allclose, skip_for_parallelism, skip_for_batch_parallism
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (
        64 * 1024,
        32 * 1024,
        32,
    ),
    ids=lambda seq_len: f"seq_len_{seq_len}_",
)
@pytest.mark.parametrize(
    "batch_dp_tp",
    [
        (1, 1, 8),
        (8, 8, 1),
        (64, 8, 1),
        (1, 1, 2),
        (2, 2, 1),
    ],
    ids=lambda args: "batch_{}_dp_{}_tp_{}".format(*args),
)
def test_llama_mlp_inference(seq_len, batch_dp_tp, mesh_device, use_program_cache, reset_seeds, ensure_gc):
    batch_size, data_parallel, tensor_parallel = batch_dp_tp

    skip, reason = skip_for_batch_parallism(batch_size, data_parallel)
    if skip:
        pytest.skip(reason)

    skip, reason = skip_for_parallelism(
        mesh_device.get_num_devices() if mesh_device else 0, data_parallel, tensor_parallel
    )
    if skip:
        pytest.skip(reason)

    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    mesh_device.enable_async(True)

    model_args = TtModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        data_parallel=data_parallel,
        tensor_parallel=tensor_parallel,
        max_seq_len=128,
    )
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaMLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(
        dim=model_args.dim,
        hidden_dim=4 * model_args.dim,
        multiple_of=model_args.multiple_of,
        ffn_dim_multiplier=model_args.ffn_dim_multiplier,
    )
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtLlamaMLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )
    torch_input = torch.randn(model_args.per_chip_batch_dim, 1, seq_len, model_args.dim)
    reference_output = reference_model(torch_input)

    if data_parallel > 1:  # if data parallel, use dim 1 to shard input
        assert tensor_parallel == 1, "Hybrid parallelism not supported"
        input_shard_dims = (0, None)
    else:  # tensor parallel
        assert data_parallel == 1, "Hybrid parallelism not supported"
        if model_args.is_galaxy:
            input_shard_dims = (None, 3)
        else:
            input_shard_dims = (None, None)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=input_shard_dims, mesh_shape=model_args.cluster_shape
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        memory_config=(
            (
                tt_model.model_config["MLP_ACT_MEMCFG"]
                if model_args.is_galaxy
                else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
            )
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG
        ),
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run Llama_MLP")
    tt_output = tt_model(tt_input, mode)

    if data_parallel > 1:
        assert tensor_parallel == 1, "Hybrid parallelism not supported"
        output_shard_dims = (0, 3)  # second parameter is not relevant for data parallel
    else:  # tensor parallel
        assert data_parallel == 1, "Hybrid parallelism not supported"
        output_shard_dims = (1, 3)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=output_shard_dims, mesh_shape=model_args.cluster_shape
        ),
    )

    tt_output_torch = tt_output_torch[:, :1, :, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Llama_MLP Passed!")
    else:
        logger.warning("Llama_MLP Failed!")

    assert passing, f"Llama_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
