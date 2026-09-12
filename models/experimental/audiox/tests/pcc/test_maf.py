# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.audiox.reference.maf import MAF_Block
from models.experimental.audiox.tt.maf import TtMAFBlock
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "batch, seq, dim, num_heads, num_experts_per_modality, num_fusion_layers",
    [
        (1, 32, 768, 8, 4, 2),
    ],
)
def test_maf_pcc(device, batch, seq, dim, num_heads, num_experts_per_modality, num_fusion_layers):
    torch.manual_seed(0)

    reference = MAF_Block(
        dim=dim,
        num_experts_per_modality=num_experts_per_modality,
        num_heads=num_heads,
        num_fusion_layers=num_fusion_layers,
    ).eval()

    video = torch.randn(batch, seq, dim)
    text = torch.randn(batch, seq, dim)
    audio = torch.randn(batch, seq, dim)

    with torch.no_grad():
        ref_out = reference(video, text, audio)

    tt_model = TtMAFBlock(
        mesh_device=device,
        state_dict=reference.state_dict(),
        num_heads=num_heads,
        num_fusion_layers=num_fusion_layers,
    )

    tt_video = ttnn.from_torch(video, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_text = ttnn.from_torch(text, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_audio = ttnn.from_torch(audio, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = tt_model(tt_video, tt_text, tt_audio)

    for modality in ("video", "text", "audio"):
        actual = ttnn.to_torch(tt_out[modality])
        assert_with_pcc(ref_out[modality], actual, pcc=PCC_THRESHOLD)
