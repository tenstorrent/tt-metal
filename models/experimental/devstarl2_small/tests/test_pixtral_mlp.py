# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: Hugging Face PixtralMLP vs MistralTTVisionMLP using real weights from
mistralai/Devstral-Small-2-24B-Instruct-2512 (vision tower layer 0 feed-forward).

Weights are loaded via safetensors partial read so the full 24B checkpoint is not loaded into RAM.
The test sets ``HF_MODEL`` to the Devstral repo id (via ``monkeypatch``) so ``ModelArgs`` matches
text/vision hyperparameters used elsewhere for this family of models.
"""

import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.pixtral.modeling_pixtral import PixtralMLP

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_pixtralmlp import MistralTTVisionMLP
from models.tt_transformers.tt.load_checkpoints import (
    convert_vision_hf_to_meta,
    load_hf_state_dict_filtered,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
VISION_MLP_KEY_PREFIX = "vision_tower.transformer.layers.0.feed_forward."


def _load_layer0_vision_mlp_tensors(repo_id: str) -> dict:
    return load_hf_state_dict_filtered(repo_id, (VISION_MLP_KEY_PREFIX,), local_files_only=os.getenv("CI") == "true")


def _vision_mlp_hf_to_meta(raw: dict, text_head_dim: int) -> dict:
    return convert_vision_hf_to_meta(standardize_hf_keys_multimodal(raw), text_head_dim)


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
def test_pixtral_mlp_pcc_devstral_weights(mesh_device, seq_len, batch_size, monkeypatch):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=256)

    hf_cfg = AutoConfig.from_pretrained(
        DEVSTRAL_REPO_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    text_head_dim = hf_cfg.text_config.hidden_size // hf_cfg.text_config.num_attention_heads

    pref = VISION_MLP_KEY_PREFIX
    hf_mlp_sd = _load_layer0_vision_mlp_tensors(DEVSTRAL_REPO_ID)
    meta_state = _vision_mlp_hf_to_meta(hf_mlp_sd, text_head_dim)

    pixtral_mlp = PixtralMLP(hf_cfg.vision_config)
    pix_sd = {k[len(pref) :]: v for k, v in hf_mlp_sd.items() if k.startswith(pref) and k.endswith(".weight")}
    pixtral_mlp.load_state_dict(pix_sd, strict=True)
    pixtral_mlp.to(torch.bfloat16)
    pixtral_mlp.eval()

    tt_mlp = MistralTTVisionMLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=meta_state,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict_prefix=pref,
        dtype=dtype,
    )

    hidden_size = model_args.vision_dim
    torch_input = torch.randn(batch_size, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    ref_out = pixtral_mlp(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_out = tt_mlp(tt_input)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[:, :, :, :hidden_size]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
