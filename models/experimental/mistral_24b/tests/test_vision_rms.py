# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import os

import ttnn
from models.experimental.mistral_24b.tt.rmsnorm import RMSNorm

from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.load_checkpoints import convert_vision_meta_to_hf


def reference_vision_rms(model_args):
    """Mistral-specific reference method for vision RMS norm."""
    model = model_args.reference_vision_transformer(wrap=False)
    layer = model.vision_tower.transformer.layers[0].ffn_norm
    layer._load_state_dict = layer.load_state_dict
    layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, model_args.head_dim))
    return layer


@torch.no_grad()
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device",
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
def test_rmsnorm_inference(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    reference_model = reference_vision_rms(tt_model_args)

    first_layer_prefix = "vision_tower.transformer.layers.0.ffn_norm."

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model.load_state_dict(partial_state_dict)

    tt_model = RMSNorm(
        device=device,
        dim=1024,
        state_dict=state_dict,
        state_dict_prefix="vision_tower.transformer.layers.0.",
        weight_key="ffn_norm",
        weight_dtype=dtype,
        is_distributed=False,
        simplified_rms=True,
    )
    input = torch.rand(batch_size, seq_len, 1024)

    reference_output = reference_model(input)

    tt_input = ttnn.from_torch(
        input,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=device),
    )

    tt_output = tt_model(tt_input, mode=mode)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))[
        :, : tt_output.shape[-1]
    ]

    logger.info(f"tt_output_torch: {tt_output_torch.shape}")
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("rms_norm Passed!")
    else:
        logger.warning("rms_norm Failed!")

    assert passing, f"rms_norm output does not meet PCC requirement {0.99}."
