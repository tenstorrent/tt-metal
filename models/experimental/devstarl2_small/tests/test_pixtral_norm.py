# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: Hugging Face PixtralRMSNorm vs TtPixtralRMSNorm using real gamma weights from
mistralai/Devstral-Small-2-24B-Instruct-2512 (vision tower layer 0 ``attention_norm``).

Weights are loaded via safetensors partial read so the full checkpoint is not loaded into RAM.
"""

import os

import pytest
import torch
from loguru import logger
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# HF ``PixtralRMSNorm`` is the same implementation as ``LlamaRMSNorm`` (see modeling_pixtral).
# Some transformers builds ship a corrupted pixtral module (norm class accidentally named ``m``),
# so we use ``LlamaRMSNorm`` here as the reference for PCC.

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_pixtralnorm import TtPixtralRMSNorm
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
VISION_ATTN_NORM_PREFIX = "vision_tower.transformer.layers.0.attention_norm."
WEIGHT_KEY = f"{VISION_ATTN_NORM_PREFIX}weight"
PIXTRAL_BLOCK_RMS_EPS = 1e-5


def _load_attention_norm_tensors(repo_id: str) -> dict:
    return load_hf_state_dict_filtered(repo_id, (VISION_ATTN_NORM_PREFIX,), local_files_only=os.getenv("CI") == "true")


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_pixtral_rmsnorm_pcc_devstral_weights(mesh_device, seq_len, batch_size):
    dtype = ttnn.bfloat16
    hf_sd = _load_attention_norm_tensors(DEVSTRAL_REPO_ID)
    gamma = hf_sd[WEIGHT_KEY]
    hidden_size = gamma.shape[0]

    ref_norm = LlamaRMSNorm(hidden_size, eps=PIXTRAL_BLOCK_RMS_EPS)
    ref_norm.load_state_dict({"weight": gamma.clone()}, strict=True)
    ref_norm.to(torch.bfloat16)
    ref_norm.eval()

    tt_norm = TtPixtralRMSNorm(
        mesh_device,
        hf_sd,
        eps=PIXTRAL_BLOCK_RMS_EPS,
        weight_key=WEIGHT_KEY,
        dtype=dtype,
    )

    torch_input = torch.randn(batch_size, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    ref_out = ref_norm(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_out = tt_norm(tt_input)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[:, :, :, :hidden_size]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
