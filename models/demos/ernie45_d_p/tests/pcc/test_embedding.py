# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.4: token embedding (replicated table) vs golden, both chunks of 2k->2k."""

import pytest
import torch

from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import replicated_to_torch
from models.demos.ernie45_d_p.tt.embedding import TtEmbedding

TASK = "P2.4"


@mesh_1x4
@pytest.mark.parametrize("chunk", [0, 1])
def test_embedding(mesh_device, cfg, loader, golden_2k, record, chunk):
    emb = TtEmbedding(mesh_device, loader.get("model.embed_tokens.weight"))
    m = golden_2k.model(chunk)
    y = replicated_to_torch(emb(m["tokens"].long()))[0, 0]
    record(f"pcc_embed_c{chunk}", y, m["embed"], 0.9999)
    assert torch.equal(y.to(torch.bfloat16), m["embed"]), "embedding lookup should be exact in bf16"
    record.check()
