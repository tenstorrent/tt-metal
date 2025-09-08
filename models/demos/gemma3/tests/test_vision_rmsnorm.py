# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma3.tt.gemma_vision_rmsnorm import RMSNorm
from models.demos.gemma3.tt.model_config import ModelArgs
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
    "seq_len",
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_rmsnorm_inference(mesh_device, seq_len, batch_size, reset_seeds):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    reference_model = tt_model_args.reference_vision_rms_norm()  # Gemma3 RMSNorm
    first_layer_prefix = "model.multi_modal_projector.mm_soft_emb_norm."

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    # reference_model.load_state_dict(partial_state_dict)

    tt_inner_norm = RMSNorm(
        device=mesh_device,
        dim=1152,
        state_dict=state_dict,
        state_dict_prefix="",
        weight_key="model.multi_modal_projector.mm_soft_emb_norm",
        weight_dtype=dtype,
        is_distributed=False,
        sharded_program_config=tt_model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
        sharded_output_config=tt_model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
    )

    # Wrap it in DistributedNorm
    # tt_model = DistributedNorm(tt_inner_norm, tt_model_args, tt_ccl=TT_CCL(mesh_device), TG=tt_model_args.is_galaxy)
    tt_model = tt_inner_norm

    input = torch.rand(1, 1, 1152)

    reference_output = reference_model(input)

    # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=tt_model_args.cluster_shape),
        memory_config=(
            tt_model_args.get_model_config()["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        ),
    )

    tt_output = tt_model(tt_input, mode=mode)

    # DistributedNorm outputs are replicated across devices
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=(0, 2) if tt_model_args.is_galaxy else (2, 0), mesh_shape=tt_model_args.cluster_shape
        ),
    )[:1, :, :].squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("rms_norm Passed!")
    else:
        logger.warning("rms_norm Failed!")

    assert passing, f"rms_norm output does not meet PCC requirement {0.99}."


@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_rms_norm(mesh_device):
    from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm

    mode = "prefill"
    tt_model_args = ModelArgs(
        mesh_device,
    )
    dtype = ttnn.bfloat16
    state_dict = tt_model_args.load_state_dict()

    batch_size = 1
    seq_len = 4096
    model_dim = 1152
    x = torch.rand(batch_size, seq_len, model_dim)
    tt_input = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ln_post = TtLayerNorm(
        device=mesh_device,
        dim=tt_model_args.vision_dim,
        state_dict=state_dict,
        state_dict_prefix="model.vision_tower.vision_model.ln_post.",
        # weight_cache_path=tt_model_args.weight_cache_path(dtype),
        weight_cache_path=None,
        weight_dtype=dtype,
        eps=tt_model_args.norm_eps,
    )

    test_output = ln_post(tt_input)
    test_torch_output = ttnn.to_torch(test_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :]
    ref_model = tt_model_args.reference_vision_model().post_layernorm

    ref_output = ref_model(x)

    passing, pcc_message = comp_pcc(ref_output, test_torch_output)

    logger.info(comp_allclose(ref_output, test_torch_output))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"rms_norm output does not meet PCC requirement {0.99}."
