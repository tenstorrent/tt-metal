# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Prototype for perf #2: does chunk_kda's FLAT (OPT-A) input path — rank-3 q/k/v with in-kernel L2-norm —
# match the current head-major + host-L2-norm path? If yes, #2 (kill the head reshapes) is a call-change.
# Compares both against the torch reference. NOT the layer; the op in isolation, KDA dims.

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.experimental.kimi_delta_attention.torch_functional import kda_ops as ref
from models.common.utility_functions import comp_pcc

torch.manual_seed(5)


@pytest.mark.parametrize("T", [64, 128])
def test_flat_vs_headmajor(device, T):
    B, HV, K, V, C = 1, 4, 128, 128, 32  # KDA: H==HV, K==V
    H = HV
    q_raw = torch.randn(B, T, HV, K)  # NOT normalized (kernel normalizes on the flat path)
    k_raw = torch.randn(B, T, HV, K)
    v = torch.randn(B, T, HV, V)
    g = -F.softplus(torch.randn(B, T, HV, K))
    beta = torch.sigmoid(torch.randn(B, T, HV))

    # reference uses host-L2-normed q/k (what both device paths should reproduce)
    o_ref, _ = ref.naive_chunk_kda(ref.l2norm(q_raw), ref.l2norm(k_raw), v, g, beta, chunk_size=C)

    def up(x):
        return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def run(o):
        return ttnn.to_torch(o[0] if isinstance(o, (tuple, list)) else o)

    # --- control: current head-major path (rank-4, host L2-norm) ---
    o_hm = run(ttnn.transformer.chunk_kda(
        up(ref.l2norm(q_raw)), up(ref.l2norm(k_raw)), up(v), up(g), up(beta), scale=K ** -0.5, chunk_size=C))
    ok_hm, pcc_hm = comp_pcc(o_ref, o_hm, pcc=0.98)
    logger.info(f"[flat_optA] T={T} head-major (control) PCC={pcc_hm}")

    # --- flat OPT-A: rank-3 q/k/v (RAW, no host norm) → kernel L2-norms internally ---
    q_flat = up(q_raw.reshape(B, T, H * K))
    k_flat = up(k_raw.reshape(B, T, H * K))
    v_flat = up(v.reshape(B, T, HV * V))
    # rank-3 q/k auto-enables in-kernel L2-norm (qk_norm = flat_qk && C==32); use_qk_l2norm stays False.
    o_flat = run(ttnn.transformer.chunk_kda(
        q_flat, k_flat, v_flat, up(g), up(beta), scale=K ** -0.5, chunk_size=C))
    ok_flat, pcc_flat = comp_pcc(o_ref, o_flat, pcc=0.98)
    logger.info(f"[flat_optA] T={T} FLAT (OPT-A) PCC={pcc_flat}")

    assert ok_hm, f"head-major control PCC too low: {pcc_hm}"
    assert ok_flat, f"flat OPT-A PCC too low: {pcc_flat}"
