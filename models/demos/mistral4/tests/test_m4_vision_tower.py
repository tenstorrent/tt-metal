# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""fp8-free vision-tower PCC for Mistral-Small-4-119B (Mistral3/Pixtral VLM).

Loads ONLY the bf16 vision_tower.* weights (no 226 GB / no fp8 MoE), builds a standalone
HF PixtralVisionModel reference + the TT MistralVisionTower, and compares.
Run: HF_MODEL=<Mistral-Small-4 snapshot> MESH_DEVICE=P150x8 pytest -s this_file.py
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.mistral_24b.mistral_vision_tower import MistralVisionTower


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_vision_tower(mesh_device, reset_seeds):
    pcc_required = 0.99
    args = ModelArgs(mesh_device)
    prefix = "vision_tower."
    # vision-only weights — all bf16; never touches the fp8 MoE/MLA text shards
    sd = load_hf_state_dict_filtered(args.CKPT_DIR, [prefix])
    vis_sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    logger.info(f"loaded {len(vis_sd)} vision_tower tensors")

    # Small image (multiple of patch_size) — the full 1540x1540 (~12k patches) makes the CPU
    # reference forward intractably slow; a small image exercises the same vision-tower math.
    patch = args.vision_patch_size
    B, C, H, W = 1, 3, patch * 32, patch * 32  # 448x448 -> 32x32 = 1024 patches
    inp = torch.rand((B, C, H, W), dtype=torch.bfloat16)

    # --- standalone HF reference (fp8-free) ---
    from transformers import AutoConfig
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

    vcfg = AutoConfig.from_pretrained(args.CKPT_DIR).vision_config
    ref = PixtralVisionModel(vcfg).to(torch.bfloat16).eval()
    missing, unexpected = ref.load_state_dict(vis_sd, strict=False)
    logger.info(f"reference load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    with torch.no_grad():
        ref_out = ref(inp, image_sizes=[(H, W)]).last_hidden_state

    # --- TT MistralVisionTower ---
    # The HF reference consumes raw HF vision keys; the TT tower expects them mapped to the
    # meta layout (patch_conv -> patch_conv._linear, etc.) — the same conversion the full
    # load_state_dict() applies for the multimodal path. Apply it to the vision-only subset
    # so we never materialize the 226 GB fp8 text core.
    from models.tt_transformers.tt.load_checkpoints import (
        convert_vision_hf_to_meta,
        convert_vision_hf_to_meta_no_qkv_permute,
    )

    convert = convert_vision_hf_to_meta_no_qkv_permute if args.use_hf_rope else convert_vision_hf_to_meta
    tt_sd = convert(dict(sd), args.head_dim)

    tt_ccl = TT_CCL(mesh_device)
    vm = MistralVisionTower(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=tt_sd,
        state_dict_prefix=prefix,
        dtype=ttnn.bfloat16,
        configuration=args,
    )
    tt_out = vm(inp, image_sizes=[(H, W)])
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[
        :, :, :, : tt_out.shape[-1]
    ].squeeze(0)

    passing, msg = comp_pcc(ref_out, tt_out, pcc_required)
    logger.info(f"Mistral-Small-4 vision-tower PCC: {msg}")
    assert passing, f"PCC below {pcc_required}: {msg}"
