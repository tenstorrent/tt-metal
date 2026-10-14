# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger
from transformers.models.mixtral.modeling_mixtral import MixtralRMSNorm as RefRMSNorm

import ttnn
from models.common.rmsnorm import RMSNorm as RMSNorm
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.load_checkpoints import HF_LAYER_NORM_KEYS, load_hf_state_dict_filtered
from models.tt_transformers.tt.model_config import ModelArgs
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh


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
    (4096,),  # Align with the sequence length used in the test input tensor
)
@pytest.mark.parametrize("mode", [Mode.PREFILL, Mode.DECODE])
def test_rms_norm_inference(
    max_seq_len,
    batch_size,
    mode,
    mesh_device,
    reset_seeds,
    ensure_gc,
    monkeypatch,
):
    dtype = ttnn.bfloat16
    # norm_type = "attention"
    norm_type = "ffn"
    config_type = "MLP"

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    # Read only the one norm weight this test uses. model_args.load_state_dict() would materialise the whole
    # 93 GB Mixtral-8x7B checkpoint from the network mount (4-5 minutes cold on the T3000 perf hosts) to
    # hand over an 8 KB gamma, which is what kept tripping the 300 s pytest timeout. The guard turns a
    # reintroduced full load into an immediate failure instead of a 5-minute timeout.
    monkeypatch.setattr(
        ModelArgs,
        "load_state_dict",
        lambda self: pytest.fail(
            "test_rms_norm_inference must not load the full checkpoint; read the norm weight only"
        ),
    )
    hf_norm_prefix = f"model.layers.0.{HF_LAYER_NORM_KEYS[norm_type]}."
    norm_weight = load_hf_state_dict_filtered(model_args.CKPT_DIR, [hf_norm_prefix])[f"{hf_norm_prefix}weight"]
    state_dict_prefix = model_args.get_state_dict_prefix("", 0)
    state_dict = {f"{state_dict_prefix}{norm_type}_norm.weight": norm_weight}
    reference_model = RefRMSNorm(hidden_size=model_args.dim)
    reference_model.load_state_dict({"weight": norm_weight})

    # Create the inner RMSNormxw
    tt_inner_norm = RMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key=f"{norm_type}_norm",
        weight_dtype=dtype,
        is_distributed=model_args.is_distributed_norm,
    )

    input = torch.rand(1, 1, 4096, 4096)
    reference_output = reference_model(input)[0]

    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    norm_config = model_args.get_norm_config("ff", mode, None)
    tt_output = tt_inner_norm(tt_input, mode=mode, norm_config=norm_config)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))[0]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mixtral_rms_norm Passed!")
    else:
        logger.warning("Mixtral_rms_norm Failed!")

    assert passing, f"Mixtral_rms_norm output does not meet PCC requirement {0.99}."
