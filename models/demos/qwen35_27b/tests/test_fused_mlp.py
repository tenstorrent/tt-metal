# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen35_27b.tt.fused_mlp import Qwen35FusedMLP
from models.demos.qwen35_27b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode


def _tt_state_dict(model_args, hf_mlp):
    layer_prefix = model_args.get_state_dict_prefix(model_args.mlp_cls.__name__, 0)

    rename_keys = {
        "gate_proj.weight": f"{layer_prefix}.w1.weight",
        "down_proj.weight": f"{layer_prefix}.w2.weight",
        "up_proj.weight": f"{layer_prefix}.w3.weight",
    }
    state_dict = {rename_keys.get(k): v.clone() for k, v in hf_mlp.state_dict().items()}
    return state_dict


@pytest.fixture(scope="session")
def hf_model():
    os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-27B")


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
    "seq_len",
    (64 * 1024, 32 * 1024, 512, 32),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_inference(seq_len, batch_size, mesh_device, hf_model, tmp_path, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b
    mode = Mode.DECODE if seq_len <= 32 else Mode.PREFILL

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
    )

    hf_mlp = model_args.reference_mlp()

    state_dict = _tt_state_dict(model_args, hf_mlp)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = Qwen35FusedMLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=tmp_path,
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )

    torch_input = torch.randn(1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(hf_mlp, model_args.model_name))
    reference_output = hf_mlp(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=dtype,
        memory_config=model_args.get_mlp_input_mem_config(mode),
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info("Run Fused MLP")

    tt_output = tt_model(tt_input, mode)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info(f"Fused MLP Passed For Input Seqence Length: {seq_len}!")
    else:
        logger.warning(f"Fused MLP Failed For Input Seqence Length: {seq_len}!")

    assert passing, f"Fused MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
