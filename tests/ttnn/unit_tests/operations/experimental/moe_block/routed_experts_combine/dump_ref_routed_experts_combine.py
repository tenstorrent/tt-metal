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
table / plan, the flat expert's weights and words gidx / done) and its output y_own (the combine's row-major [1, 1, S,
Hr]), with combine_snap 2^-14 (bit-identical in any landing order). The routing comes from the device planner of the
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

# (name, model key in models/demos/moe_block/moe_block.MODELS, routing)
CASES = [
    ("glm53_8x4_uniform", "glm53", "uniform"),
    ("k3_8x4_uniform", "kimi_k3", "uniform"),
]


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


@pytest.mark.parametrize("case, key, routing", CASES)
@pytest.mark.parametrize(
    "device_params",
    [dict(fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY if (R, C) == (8, 4) else ttnn.FabricConfig.FABRIC_2D)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(R, C)], indirect=True)
def test_dump(mesh_device, case, key, routing):
    from ttnn.bringup.combine_rf.routing_a2a import worst_rows
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import K3Dims, TtK3TokenSplitMoE

    N = R * C
    m = _model(key, N)
    Hr, I, E, K = m.routed_width, m.routed_expert_hidden_dim, m.num_experts, m.top_k
    S, EPC = CHUNK // N, E // N
    seed = 1234
    torus = ((R, C) == (8, 4), (R, C) == (8, 4))
    from models.demos.moe_block.moe_block import _FLAT_ACT

    act = _FLAT_ACT[m.routed_expert_activation.kind]
    d = K3Dims(H=Hr, latent=Hr, I=I, E=E, K=K, act=act)
    block = TtK3TokenSplitMoE(
        mesh_device,
        None,
        d,
        S,
        rows_cap=worst_rows(N, S, K, EPC),
        torus=torus,
        combine_snap=SNAP,
    )
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.02  # noqa: E731
    gids = [list(range(c * EPC, (c + 1) * EPC)) for c in range(N)]
    per_dev = [[(r(Hr, I), r(Hr, I), r(I, Hr)) for _ in gl] for gl in gids]
    op = FlatRoutedExpert(
        mesh_device, per_dev, m=block.op.m, H=Hr, I=I, gids=gids, n_global=E, wdtype="bf4", act=d.act, pin=1,
        layout=block.flat_layout,
    )  # fmt: skip
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
        w_gu=op.w_gu,
        w_d=op.w_d,
        gidx=op.gidx,
        y_own=out,
    )
    if op.w_rd is not None:
        named["w_rd"] = op.w_rd
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
        ),
        extra=dict(
            dtypes={k: str(v.dtype) for k, v in named.items()},
            combine=dict(depth=block.combine.depth, pool=block.combine.pool, l1=block.combine.l1),
        ),
    )
