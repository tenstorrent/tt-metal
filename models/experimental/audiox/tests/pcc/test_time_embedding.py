# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.audiox.reference.time_embedding import TimestepEmbedding
from models.experimental.audiox.tt.time_embedding import TtTimestepEmbedding
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "batch, embed_dim, fourier_dim",
    [
        (4, 1536, 256),  # AudioX DiT spec: embed_dim=1536, timestep_features_dim=256
    ],
)
def test_time_embedding_pcc(device, batch, embed_dim, fourier_dim):
    torch.manual_seed(0)

    reference = TimestepEmbedding(embed_dim=embed_dim, fourier_dim=fourier_dim).eval()

    t = torch.rand(batch)

    with torch.no_grad():
        ref_out = reference(t)

    tt_model = TtTimestepEmbedding(mesh_device=device, state_dict=reference.state_dict())
    tt_t = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_model(tt_t)

    assert_with_pcc(ref_out, ttnn.to_torch(tt_out), pcc=PCC_THRESHOLD)
