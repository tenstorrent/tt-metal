# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm as TtRMSNorm
from models.demos.qwen.reference.model import RMSNorm as RefRMSNorm
from models.demos.qwen.tt.distributed_norm import DistributedNorm
from models.demos.qwen.tt.model_config import TtModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


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
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_qwen_rms_norm_inference(mesh_device, use_program_cache, reset_seeds, ensure_gc, mode):
    if mesh_device.shape != (1, 1):
        pytest.skip("Only N150 is supported")
    dtype = ttnn.bfloat16

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", 0)
    first_layer_prefix = state_dict_prefix + "input_layernorm."

    # Create the inner RMSNormxw
    tt_inner_norm = TtRMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key="input_layernorm",
        weight_dtype=dtype,
        is_distributed=model_args.is_distributed_norm,
        sharded_program_config=model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
        sharded_output_config=model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
    )

    # Wrap it in DistributedNorm
    tt_model = DistributedNorm(tt_inner_norm, model_args)

    # Create reference model (unchanged)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = RefRMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
    reference_model.load_state_dict(partial_state_dict)

    input = torch.rand(1, 1, 32, model_args.dim)
    reference_output = reference_model(input)

    # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_model(tt_input, mode=mode)

    # DistributedNorm outputs are replicated across devices
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        :1, :, :
    ].squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Qwen_rms_norm Passed!")
    else:
        logger.warning("Qwen_rms_norm Failed!")

    assert passing, f"Qwen_rms_norm output does not meet PCC requirement {0.99}."
