# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the multi-core combine planner on joint routing rows at top-K 4 / 6 / 8 (MiniMax M3, DeepSeek V4.1, the
K3 / GLM family) and 1 / 32 tokens a page, on the 2x4 against its host model, bit for bit."""

import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, plan_reference, port_luts
from ttnn.bringup.combine_rf.routing_a2a import worst_rows

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

R, C, L, A = 2, 4, 2, 15


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((R, C), fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0), id="2x4")],
    indirect=True,
)
@pytest.mark.parametrize("routing", ["uniform", "packed"])  # (packed: every token on chip 0's experts - the worst case)
@pytest.mark.parametrize("pt", [32, 1])
@pytest.mark.parametrize("K, EPC, S", [(6, 48, 128), (4, 16, 128), (8, 32, 128), (6, 48, 320)])
def test_plan_k(mesh_device, device_params, K, EPC, S, pt, routing):
    N = R * C
    g = torch.Generator().manual_seed(K)
    span = EPC if routing == "packed" else N * EPC
    ids = torch.stack([torch.randperm(span, generator=g)[:K] for _ in range(N * S)]).view(N, S, K)
    gates = torch.softmax(torch.randn(N, S, K, generator=g), -1).to(torch.bfloat16)
    wbits = gates.view(torch.int16).to(torch.int64) & 0xFFFF
    luts, nps = port_luts(mesh_device, L)
    rows_cap = worst_rows(N, S, K, EPC)
    run = combine_planner_prepare(
        mesh_device,
        S=S,
        K=K,
        EPC=EPC,
        A=A,
        L=L,
        port_luts=luts,
        n_ports=nps,
        rows_cap=rows_cap,
        cores=[(i % 8, i // 8) for i in range(8)],
        row_pages=True,
        joint=True,
        tokens_per_page=pt,
    )
    both = torch.cat([ids.reshape(N * S, K), wbits.reshape(N * S, K)], 1).to(torch.int32)
    t = ttnn.from_torch(
        both.reshape(N * S // pt, pt * 2 * K),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    outs = run(t, t)
    ttnn.synchronize_device(mesh_device)
    got = {
        k: [ttnn.to_torch(x).to(torch.int64).reshape(-1) & 0xFFFFFFFF for x in ttnn.get_device_tensors(v)]
        for k, v in outs.items()
    }
    for ci, ch in enumerate([(r, c) for r in range(R) for c in range(C)]):
        ref = plan_reference(
            ids, wbits, ci, R, C, S, K, EPC, N * EPC, A, L, nps[ch], luts[ch], rows_cap, token_major=run.token_major
        )
        for k, v in ref.items():
            exp = torch.tensor(v, dtype=torch.int64)
            bad = (got[k][ci][: exp.numel()] != exp).nonzero().flatten()
            assert (
                bad.numel() == 0
            ), f"K {K} pt {pt} chip {ci} {k}: {bad.numel()} words differ, first {bad[:8].tolist()}"
    logger.info(f"PLAN_K K {K} EPC {EPC} S {S} pt {pt} {routing}: every chip's plan matches the host model")
