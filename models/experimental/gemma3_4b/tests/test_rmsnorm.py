"""Gemma-3-4b-it Test for Text RMSNorm"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import os

import ttnn
from models.experimental.gemma3_4b.tt.rmsnorm import RMSNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.ccl import TT_CCL
from models.experimental.gemma3_4b.tests.references import reference_transformer

from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull

from models.tt_transformers.tt.model_config import ModelArgs


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("device"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "tt_layer_name, torch_layer_name, dim",
    (
        ("norm", "norm", 2560),
        ("layers.0.attention_norm", "layers.0.input_layernorm", 2560),
        ("layers.0.ffn_norm", "layers.0.post_attention_layernorm", 2560),
        ("layers.0.pre_feedforward_layernorm", "layers.0.pre_feedforward_layernorm", 2560),
        ("layers.0.post_feedforward_layernorm", "layers.0.post_feedforward_layernorm", 2560),
        ("layers.0.attention.q_norm", "layers.0.self_attn.q_norm", 256),
        ("layers.0.attention.k_norm", "layers.0.self_attn.k_norm", 256),
    ),
)
@pytest.mark.parametrize(
    "seq_len",
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_rmsnorm_inference(seq_len, batch_size, reset_seeds, device, tt_layer_name, torch_layer_name, dim):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_ccl = TT_CCL(device)

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()
    reference_model = reference_transformer(tt_model_args, wrap=False)  # Gemma3 Entire Model
    reference_model = reference_model.model.get_submodule(torch_layer_name)

    state_dict_prefix = ""
    first_layer_prefix = state_dict_prefix + tt_layer_name + "."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model.load_state_dict(partial_state_dict)

    tt_inner_norm = RMSNorm(
        device=device,
        tt_ccl=tt_ccl,
        dim=dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key=tt_layer_name,
        weight_dtype=dtype,
        is_distributed=tt_model_args.is_distributed_norm,
        sharded_program_config=tt_model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
        sharded_output_config=tt_model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
    )

    # Wrap it in DistributedNorm
    tt_model = DistributedNorm(tt_inner_norm, tt_model_args, tt_ccl=tt_ccl, TG=tt_model_args.is_galaxy)

    input = torch.rand(1, 1, dim)

    reference_output = reference_model(input)

    # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
    tt_input = ttnn.from_torch(
        input,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(None, -1), mesh_shape=tt_model_args.cluster_shape),
        memory_config=(
            tt_model_args.get_model_config()["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        ),
    )

    tt_output = tt_model(tt_input, mode=mode)

    # DistributedNorm outputs are replicated across devices
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            device, dims=(0, 2) if tt_model_args.is_galaxy else (2, 0), mesh_shape=tt_model_args.cluster_shape
        ),
    )[:1, :, :]

    tt_output_torch = tt_output_torch.view(1, 1, dim)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch[0])

    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {torch_layer_name} , {pcc_message}")

    if passing:
        logger.info("rms_norm Passed!")
    else:
        logger.warning("rms_norm Failed!")

    assert passing, f"rms_norm output does not meet PCC requirement {0.99}."
