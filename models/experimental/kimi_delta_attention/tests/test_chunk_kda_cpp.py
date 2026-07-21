# SPDX-License-Identifier: Apache-2.0
# PCC: fused C++ ttnn.transformer.chunk_kda vs torch naive_chunk_kda (diagonal gate).
import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.experimental.kimi_delta_attention.torch_functional import kda_ops as ref
from models.common.utility_functions import comp_pcc

torch.manual_seed(5)


@pytest.mark.parametrize("T", [32, 64, 128])
def test_chunk_kda_cpp(device, T):
    B, HV, K, V, C = 1, 4, 128, 128, 32
    q = ref.l2norm(torch.randn(B, T, HV, K))
    k = ref.l2norm(torch.randn(B, T, HV, K))
    v = torch.randn(B, T, HV, V)
    g = -F.softplus(torch.randn(B, T, HV, K))
    beta = torch.sigmoid(torch.randn(B, T, HV))

    o_ref, _ = ref.naive_chunk_kda(q, k, v, g, beta, chunk_size=C)

    def up(x):
        return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    o = ttnn.transformer.chunk_kda(up(q), up(k), up(v), up(g), up(beta), scale=K ** -0.5, chunk_size=C)
    if isinstance(o, (tuple, list)):
        o = o[0]
    ok, pcc = comp_pcc(o_ref, ttnn.to_torch(o), pcc=0.98)
    logger.info(f"[chunk_kda_cpp] T={T} PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"
