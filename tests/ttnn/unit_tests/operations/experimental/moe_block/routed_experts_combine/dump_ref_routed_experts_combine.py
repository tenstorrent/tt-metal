# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference cases for ttnn.experimental.moe_block.routed_experts_combine, dumped from the Python exchange of the trimmed
tree (mstaletovic/moe-trim: models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py, the experts + combine program
flat_routed_expert_with_program + sparse_combine_rf). Run IN THE REFERENCE TREE's environment (its ttnn), e.g. on a
Galaxy 8x4:

    cd /data/mstaletovic/tt-metal-trimb
    MOE_MESH=8x4 pytest -q /data/mstaletovic/tt-metal-port-routed_experts_combine/tests/ttnn/unit_tests/operations/\
experimental/moe_block/routed_experts_combine/dump_ref_routed_experts_combine.py

Per case and chip it saves the op's inputs (the dispatched rows xg, the planner's counts / regions / token_index /
table / plan; the experts' weights regenerate from the seed: torch.randn x 0.02 per chip, per local expert Wg, Wu, Wd)
and its output y_own (the combine's row-major [1, 1, S,
Hr]), with combine_snap 2^-14 (bit-identical in any landing order). Built as MoEBlock builds its exchange: per-expert
capacity = the chunk, the flat layout RoutedExperts.layout_of (rows_hint chunk x K / E, one x helper). Every chip holds
the same local experts' weights (one set of EPC experts, packed once and replicated: K3's 896 distinct experts would be
~118 GB of fp32 host weights - the first dump's host OOM). The routing comes from the device planner of the
standalone exchange (routed()), the same tables the fused head's planner writes.
"""

import os
import random
import sys
from pathlib import Path

import pytest
import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ref_harness import from_mesh, save_case  # noqa: E402

OP = "routed_experts_combine"
R, C = (int(v) for v in os.environ.get("MOE_MESH", "8x4").split("x"))
SNAP = 2.0**-14
CHUNK = 5120

# (name, model key in models/demos/moe_block/moe_block.MODELS, routing, routed width Hr)
CASES = [
    ("glm53_8x4_uniform", "glm53", "uniform", 6144),
    ("k3_8x4_uniform", "kimi_k3", "uniform", 3584),
]


def fabric_payload(W):
    """The MoE block's fabric payload for rows of W bf16 (closeout_models.fabric_payload): the combine's L1 hand-off
    sends a row as one piece plus its 32 B trailer - a smaller payload never delivers the trailers (a hang)."""
    return max(7232, (2 * W + 32 + 63) // 64 * 64)


def _device_params(W):
    router = ttnn._ttnn.fabric.FabricRouterConfig()
    router.max_packet_payload_size_bytes = fabric_payload(W)
    return dict(
        fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY if (R, C) == (8, 4) else ttnn.FabricConfig.FABRIC_2D,
        fabric_router_config=router,
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
        l1_small_size=0,  # (as the block tests)
    )


def _model(key, chips):
    from models.demos.moe_block.tests.test_moe_block_matrix import mesh_model

    name = {"glm53": "glm53", "kimi_k3": "k3"}[key]
    return mesh_model(name, chips)


def _joint_rows(N, S, K, E, seed):
    """Uniform routing as the router's joint rows: per chip [S / 32, 32 x 2K] uint16 (a token's K ids, then its K gate
    weights' bf16 bits)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(N):
        tok = []
        for _ in range(S):
            ids = sorted(rng.sample(range(E), K))
            w = torch.tensor([rng.uniform(0.05, 1.0) for _ in range(K)]).to(torch.bfloat16).view(torch.int16)
            tok.append(torch.cat([torch.tensor(ids, dtype=torch.int16), w]))
        rows.append(torch.stack(tok).reshape(S // 32, 32 * 2 * K))
    return torch.stack(rows)  # [N, S / 32, 32 x 2K]


@pytest.mark.parametrize(
    "mesh_device, device_params, case, key, routing",
    [pytest.param((R, C), _device_params(W), case, key, routing, id=case) for case, key, routing, W in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_dump(mesh_device, device_params, case, key, routing):
    from ttnn.bringup.combine_rf.routing_a2a import worst_rows
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import K3Dims, TtK3TokenSplitMoE

    N = R * C
    m = _model(key, N)
    assert m.routed_width == dict((k, w) for _, k, _, w in CASES)[key], "CASES' routed width"
    Hr, I, E, K = m.routed_width, m.routed_expert_hidden_dim, m.num_experts, m.top_k
    S, EPC = CHUNK // N, E // N
    seed = 1234
    torus = ((R, C) == (8, 4), (R, C) == (8, 4))
    from models.demos.moe_block.moe_block import _FLAT_ACT

    act = _FLAT_ACT[m.routed_expert_activation.kind]
    d = K3Dims(H=Hr, latent=Hr, I=I, E=E, K=K, act=act)
    cap = max(CHUNK, 256)  # (RoutedExperts.cap_of)
    flat_layout = dict(rows_hint=round(CHUNK * K / E), x_helpers=1)  # (RoutedExperts.layout_of)
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.02  # noqa: E731
    gids = [list(range(c * EPC, (c + 1) * EPC)) for c in range(N)]
    local = [(r(Hr, I), r(Hr, I), r(I, Hr)) for _ in range(EPC)]  # (one set: every chip's local experts)
    per_dev = [local] * N
    # (the experts first, as MoEBlock builds them: their pinned L1 is there when the block plans its arena)
    op = FlatRoutedExpert(
        mesh_device, per_dev, m=cap, H=Hr, I=I, gids=gids, n_global=E, wdtype="bf4", act=d.act, pin=1,
        layout=flat_layout,
    )  # fmt: skip
    block = TtK3TokenSplitMoE(
        mesh_device,
        None,
        d,
        S,
        cap=cap,
        rows_cap=worst_rows(N, S, K, EPC),
        flat_layout=flat_layout,
        torus=torus,
        combine_snap=SNAP,
    )
    x = ttnn.from_torch(
        r(N, S, Hr).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    rows = ttnn.from_torch(
        _joint_rows(N, S, K, E, seed).reshape(N * S // 32, 32 * 2 * K),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    out = block.routed(x, rows, None, op, tilize=False)
    ttnn.synchronize_device(mesh_device)
    pl = block.planner.outs
    xg = ttnn.reshape(block._dprep_bufs[0], [N * S, Hr])
    named = dict(
        xg=xg,
        counts=pl["counts"],
        regions=pl["regions"],
        token_index=pl["token_index"],
        table=pl["table"],
        plan=pl["plan"],
        y_own=out,
    )
    per_chip = {(i // C, i % C): {} for i in range(N)}
    for name, t in named.items():
        for ch, v in from_mesh(t, (R, C)).items():
            per_chip[ch][name] = v
    inputs = [k for k in named if k != "y_own"]
    save_case(
        OP,
        case,
        (R, C),
        per_chip,
        inputs=inputs,
        outputs=["y_own"],
        seed=seed,
        model=key,
        config=dict(
            S=S,
            Hr=Hr,
            I=I,
            E=E,
            K=K,
            chunk_tokens=CHUNK,
            m=int(block.op.m),
            combine_snap=SNAP,
            torus=list(torus),
            n_acc=block.A,
            acc_on_ports=bool(block.acc_on_ports),
            arena_bytes=int(block.arena_bytes),
            cells=[list(c) for c in block.cells],
            writers=[list(c) for c in block.op.plan["down"]]
            + ([list(c) for c in block.op.plan["rdn"]] if block.op.plan["rdown"] else []),
            flat_static_bytes=int(block.op.plan["static_l1_bytes"]),
            routing=routing,
            activation=d.act,
            flat_layout=dict(block.flat_layout),
            fabric_config="FABRIC_2D_TORUS_XY" if (R, C) == (8, 4) else "FABRIC_2D",
            fabric_payload_bytes=fabric_payload(Hr),
            weights="the same EPC local experts on every chip: (Wg [Hr, I], Wu [Hr, I], Wd [I, Hr]) x EPC, randn x 0.02",
        ),
        extra=dict(
            dtypes={k: str(v.dtype) for k, v in named.items()},
            combine=dict(depth=block.combine.depth, pool=block.combine.pool, l1=block.combine.l1),
        ),
    )
    del per_chip, per_dev, local
