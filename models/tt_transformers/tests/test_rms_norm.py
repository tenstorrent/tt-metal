# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm as RMSNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


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
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_rms_norm_inference(
    max_seq_len,
    batch_size,
    mode,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)

    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", 0)
    first_layer_prefix = state_dict_prefix + "attention_norm."

    # Create the inner RMSNormxw
    tt_inner_norm = RMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key="attention_norm",
        weight_dtype=dtype,
        add_unit_offset=model_args.rms_norm_add_unit_offset,
        is_distributed=model_args.is_distributed_norm,
        sharded_program_config=model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
        sharded_output_config=model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
    )

    # Wrap it in DistributedNorm
    tt_model = DistributedNorm(tt_inner_norm, model_args, TG=model_args.is_galaxy)

    # Create reference model (unchanged)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = model_args.reference_rms_norm()
    reference_model.load_state_dict(partial_state_dict)

    input = torch.rand(1, 1, 32, model_args.dim)
    reference_output = reference_model(input)

    # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=model_args.cluster_shape),
        memory_config=(
            model_args.get_model_config()["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        ),
    )

    tt_output = tt_model(tt_input, mode=mode)

    # DistributedNorm outputs are replicated across devices
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=(0, 3) if model_args.is_galaxy else (3, 0), mesh_shape=model_args.cluster_shape
        ),
    )[:1, :, :, :]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc=0.9999)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("rms_norm Passed!")
    else:
        logger.warning("rms_norm Failed!")

    assert passing, f"rms_norm output does not meet PCC requirement {0.9999}."
