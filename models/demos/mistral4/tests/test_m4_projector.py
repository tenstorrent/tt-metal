# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC for the Mistral-Small-4 multi-modal projector (vision features -> text-hidden image embeds).

vs HF Mistral3MultiModalProjector: norm -> 2x2 patch-merge -> merging_layer -> linear_1 -> gelu ->
linear_2. Recipe torch-verified at 0.9999996 (incl. the channel-major unfold). bf16 weights.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tt.mistral4_text import TtMistral4Projector
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_projector(mesh_device, reset_seeds):
    pcc_required = 0.99
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt)
    vh, patch, sm = cfg.vision_config.hidden_size, cfg.vision_config.patch_size, cfg.spatial_merge_size

    # reference projector (bf16) with real weights
    ref = Mistral3MultiModalProjector(cfg).to(torch.bfloat16).eval()
    sd = load_hf_state_dict_filtered(ckpt, ["multi_modal_projector."])
    proj_sd = {k[len("multi_modal_projector.") :]: v for k, v in sd.items()}
    ref.load_state_dict(proj_sd)

    H = W = patch * 32  # 448 -> 32x32 = 1024 patches -> 16x16 = 256 merged tokens
    n = (H // patch) * (W // patch)
    torch.manual_seed(0)
    feats = torch.rand(n, vh, dtype=torch.bfloat16)
    image_sizes = torch.tensor([[H, W]])
    with torch.no_grad():
        golden = ref(feats, image_sizes)

    tt_proj = TtMistral4Projector(mesh_device, proj_sd, cfg)
    feats_tt = ttnn.from_torch(
        feats, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    out = tt_proj(feats_tt, image_sizes)
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[: golden.shape[0]]
    passing, msg = comp_pcc(golden, out_t, pcc_required)
    logger.info(f"Mistral-Small-4 projector PCC: {msg}")
    assert passing, f"projector PCC below {pcc_required}: {msg}"
