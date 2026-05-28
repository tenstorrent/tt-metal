# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Per-submodule PCC test for MoonVisionBlock (full encoder layer).

Compares against HF MoonVitEncoderLayer.forward at PCC >= 0.99. Tests
both single-image and multi-image packed inputs. The HF reference is
pinned to `sdpa` attention.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_block.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.block import MoonVisionBlock
from models.demos.deepseek_v3.tt.moonvit.rope import Rope2DSetup


def _build_cu_seqlens(grid_hws: torch.Tensor) -> torch.Tensor:
    """Mirror MoonVitEncoder.forward's cu_seqlens construction."""
    lengths = torch.cat([torch.zeros(1, dtype=grid_hws.dtype), grid_hws[:, 0] * grid_hws[:, 1]])
    return lengths.cumsum(dim=0, dtype=torch.int32)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize(
    "grid_hws",
    [
        [[16, 16]],  # single image.
        [[16, 16], [16, 16]],  # two equal images packed.
    ],
)
def test_moonvit_block(mesh_device, model_args, grid_hws):
    """Encoder layer forward matches HF at PCC >= 0.99."""
    pcc_threshold = 0.99

    # 1. HF reference layer (force sdpa for correct multi-image masking).
    ref_layer = model_args.reference_block(layer_num=0)
    ref_layer.attn_implementation = "sdpa"
    ref_layer_fp32 = ref_layer.float()

    # 2. RoPE freqs (computed at the encoder level in HF).
    ref_rope = model_args.reference_rope_2d()
    rope = Rope2DSetup.from_torch(ref_rope)

    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    L = int(grid_tensor.prod(dim=1).sum().item())
    freqs_cis = ref_rope.get_freqs_cis(grid_tensor)
    cu_seqlens_pt = _build_cu_seqlens(grid_tensor)

    # 3. Run HF forward in fp32.
    torch.manual_seed(0)
    x_pt = torch.randn(L, model_args.hidden_size, dtype=torch.float32)
    ref_out = ref_layer_fp32(x_pt, cu_seqlens_pt, rope_freqs_cis=freqs_cis)
    assert ref_out.shape == (L, model_args.hidden_size)

    # 4. Build ttnn block.
    tt_block = MoonVisionBlock.from_torch(
        mesh_device=mesh_device,
        ref_layer=ref_layer,
        hidden_size=model_args.hidden_size,
        num_heads=model_args.num_attention_heads,
        head_dim=model_args.head_dim,
        dtype=ttnn.bfloat16,
    )

    # 5. Stage inputs.
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_4d = x_pt.to(torch.bfloat16).view(1, 1, L, model_args.hidden_size).contiguous()
    x_tt = ttnn.from_torch(
        x_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    cos_pt, sin_pt = rope.get_cos_sin(grid_tensor, dtype=torch.float32)
    cos_tt, sin_tt = tt_block.attention.stage_cos_sin(cos_pt, sin_pt)
    cu_tt = tt_block.attention.stage_cu_seqlens(cu_seqlens_pt)

    # 6. Forward.
    out_tt = tt_block(x_tt, cu_tt, cos_tt, sin_tt)
    out_pt = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_pt.shape[0] != 1:
        out_pt = out_pt[:1]
    out_pt = out_pt.view(L, model_args.hidden_size)

    # 7. PCC.
    passing, pcc_msg = comp_pcc(ref_out, out_pt, pcc_threshold)
    logger.info(f"[grid_hws={grid_hws} L={L}] {comp_allclose(ref_out, out_pt)} {pcc_msg}")
    assert passing, f"PCC below threshold for grid_hws={grid_hws}: {pcc_msg}"
