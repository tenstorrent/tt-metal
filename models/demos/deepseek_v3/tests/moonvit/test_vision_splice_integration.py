# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Integration test: MoonViT vision tower output spliced into a text-embedded
stream. Bridges Phase 1 (vision tower PCC) and Phase 2 (splice) without
needing the DeepSeek LLM half.

Flow:
  1. Build the MoonViT vision tower (with projector) from HF reference.
  2. Run a real image-patches batch through MoonViT -> vision tokens
     on host (shape (L_new, text_hidden)).
  3. Synthesize a fake "text embedding" tensor of shape (1, 1, seq_len, text_hidden)
     and a token stream with image_token_id at the splice positions.
  4. Splice on host. Then also splice via the device-roundtrip wrapper.
  5. Verify:
       - shape preserved,
       - non-image positions identical to the fake text embedding,
       - image positions hold the MoonViT vision-token rows.

This catches integration bugs that the unit-level splice and unit-level
vision tower tests would miss:
  - vision_hidden vs text_hidden mismatch at the splice boundary,
  - row-order misalignment between patch_merger output and splice consumer,
  - dtype/layout conversions between MoonViT output and Embedding2D-style
    text tensor.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_vision_splice_integration.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.model import MoonViT
from models.demos.deepseek_v3.tt.moonvit.prefill_splice import (
    splice_vision_into_text_embeddings,
    splice_vision_via_host,
)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize(
    "grid_hws, prefix_tokens, suffix_tokens",
    [
        # Standard: image followed by a short text suffix.
        ([[16, 16]], 10, 30),  # L_new = 64, total seq = 10 + 64 + 30 = 104
    ],
)
def test_vision_tower_into_splice(mesh_device, model_args, grid_hws, prefix_tokens, suffix_tokens):
    """End-to-end: MoonViT -> vision tokens -> splice into fake text embeddings."""

    text_hidden = model_args.text_hidden_size
    image_token_id = 9999  # synthetic; real path uses model_args.media_placeholder_token_id

    # ---- 1. Build MoonViT (with projector) -----------------------------
    moonvit = MoonViT.from_torch(
        model_args=model_args,
        mesh_device=mesh_device,
        with_projector=True,
        dtype=ttnn.bfloat16,
    )

    # ---- 2. Run MoonViT on dummy patches to get vision tokens ----------
    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    L_patches = int(grid_tensor.prod(dim=1).sum().item())  # 256 for [[16,16]]
    L_new = sum((h // 2) * (w // 2) for h, w in grid_hws)  # 64 for [[16,16]]

    torch.manual_seed(0)
    pixel_patches = torch.randn(L_patches, 3, 14, 14, dtype=torch.float32)

    vision_tt = moonvit(pixel_patches, grid_tensor)
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    vision_pt = ttnn.to_torch(
        vision_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and vision_pt.shape[0] != 1:
        vision_pt = vision_pt[:1]
    vision_tokens = vision_pt.view(L_new, text_hidden)
    assert vision_tokens.shape == (
        L_new,
        text_hidden,
    ), f"vision_tokens has shape {tuple(vision_tokens.shape)}, expected ({L_new}, {text_hidden})"

    # ---- 3. Synthesize fake text embedding + token stream --------------
    seq_len = prefix_tokens + L_new + suffix_tokens
    image_positions = list(range(prefix_tokens, prefix_tokens + L_new))

    tokens = torch.full((1, 1, seq_len), fill_value=1, dtype=torch.int64)
    for p in image_positions:
        tokens[0, 0, p] = image_token_id

    # Unique-per-position text embedding so we can detect any clobbering of non-image rows.
    text_emb = (
        torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1).expand(1, 1, seq_len, text_hidden).clone()
    )

    # ---- 4a. Splice on host (reference) --------------------------------
    fused_host = splice_vision_into_text_embeddings(text_emb, tokens, vision_tokens.to(torch.float32), image_token_id)
    assert fused_host.shape == text_emb.shape

    # Image positions should hold the MoonViT vision tokens, in order.
    for vis_i, pos in enumerate(image_positions):
        diff = (fused_host[0, 0, pos] - vision_tokens[vis_i].to(torch.float32)).abs().max().item()
        assert diff < 1e-3, f"position {pos}: host splice did not place vision token {vis_i} " f"(max abs diff {diff})"
    # Non-image positions: untouched.
    for pos in range(seq_len):
        if pos in image_positions:
            continue
        assert torch.all(fused_host[0, 0, pos] == float(pos)), f"position {pos}: host splice clobbered a text position"

    # ---- 4b. Splice via device roundtrip -------------------------------
    text_tt = ttnn.from_torch(
        text_emb.to(torch.bfloat16).contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    fused_tt = splice_vision_via_host(
        mesh_device=mesh_device,
        text_embedded_tt=text_tt,
        tokens=tokens,
        vision_tokens=vision_tokens,  # already host bf16 from MoonViT output
        image_token_id=image_token_id,
        dtype=ttnn.bfloat16,
    )
    fused_device = ttnn.to_torch(
        fused_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and fused_device.shape[0] != 1:
        fused_device = fused_device[:1]
    fused_device = fused_device.view(1, 1, seq_len, text_hidden)

    # ---- 5. Compare host vs device splice ------------------------------
    pcc_threshold = 0.999
    passing, pcc_msg = comp_pcc(fused_host, fused_device, pcc_threshold)
    logger.info(
        f"[grid_hws={grid_hws} seq={seq_len} L_new={L_new}] " f"{comp_allclose(fused_host, fused_device)} {pcc_msg}"
    )
    assert passing, f"device-roundtrip integration splice diverges from host: {pcc_msg}"
