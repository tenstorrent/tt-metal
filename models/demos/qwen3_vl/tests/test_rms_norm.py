# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen3_vl.tt.model_config import VisionModelArgs
from models.demos.qwen3_vl.tt.vision_layernorm import LayerNorm


@torch.no_grad()
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
    (
        # 14336,  # TODO: fix padding issues
        14308,  # from 3B test image
    ),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_rms_norm_inference(
    max_seq_len,
    batch_size,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat16

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=max_seq_len)

    reference_model = model_args.reference_rms_norm()

    state_dict = reference_model.state_dict()
    state_dict = {f"norm2.{k}": v for k, v in state_dict.items()}

    # Create the inner RMSNorm
    tt_model = LayerNorm(
        device=mesh_device,
        dim=model_args.dim,
        eps=1e-6,  # Qwen3_VLVisionBlock hard-codes this
        state_dict=state_dict,
        state_dict_prefix="norm2",
        weight_cache_path=model_args.weight_cache_path(dtype),
        weight_dtype=dtype,
    )

    # # Not sure if distributed norm is supported for layer norm
    # tt_model = DistributedNorm(tt_inner_norm, model_args, tt_ccl=tt_ccl, TG=model_args.is_galaxy)

    input = torch.rand(1, 1, max_seq_len, model_args.dim)
    reference_output = reference_model(input)

    # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input)

    # DistributedNorm outputs are replicated across devices
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            dims=(0, 3) if model_args.is_galaxy else (3, 0),
            mesh_shape=model_args.cluster_shape,
        ),
    )[:1, :, :, :]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("rms_norm Passed!")
    else:
        logger.warning("rms_norm Failed!")

    assert passing, f"rms_norm output does not meet PCC requirement {0.99}."
