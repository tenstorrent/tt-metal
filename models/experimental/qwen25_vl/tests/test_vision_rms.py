"""Test for Qwen 2.5 VL RMSNorm Layer Inference"""

from loguru import logger

import torch
import pytest
import os

import ttnn
from models.experimental.qwen25_vl.tt.rmsnorm import RMSNorm

from models.tt_transformers.tt.distributed_norm import DistributedNorm


from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
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
    "seq_len",
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_rmsnorm_inference(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    dim = tt_model_args.vision_dim

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    reference_model = tt_model_args.reference_vision_rms_norm()  # Qwen2_5 RMSNorm
    first_layer_prefix = "visual.blocks.0.norm1."

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model.load_state_dict(partial_state_dict)

    tt_inner_norm = RMSNorm(
        device=device,
        dim=dim,
        state_dict=state_dict,
        state_dict_prefix="",
        weight_key=first_layer_prefix[:-1],  # Remove trailing dot
        weight_dtype=dtype,
        is_distributed=False,
        sharded_program_config=tt_model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
        sharded_output_config=tt_model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
    )

    # Wrap it in DistributedNorm
    tt_model = DistributedNorm(tt_inner_norm, tt_model_args, TG=tt_model_args.is_galaxy)

    input = torch.rand(1, 1, 1280)

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

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("rms_norm Passed!")
    else:
        logger.warning("rms_norm Failed!")

    assert passing, f"rms_norm output does not meet PCC requirement {0.99}."
