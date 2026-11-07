# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load-based decode buckets, layer level: the drafter of the FIRST Ub USERS of every mesh row (``DSparkDrafter.view_users(Ub)``, T_d = 5 Ub rows, shared ring caches / weights) against the full
drafter of U = 4 users per mesh row (B = 16) on the same inputs (the CPU-reference data of tests/test_spec_mtp.py, repeated to 16 users): draft tokens (all 5 positions), markov-biased logits PCC and
confidence PCC of the users u < Ub. The full drafter runs first (it writes the rings of all users); each view then re-writes the same main_kv rows of its users and drafts.
Env: DSV41_NV (verify rows per user n, default 6), VIEW_UBS (default "2,1"). Prints 'VIEWTEST Ub ... tokens_equal ... logits_pcc ... conf_pcc ...'.
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.mtp import BLOCK, NOISE_ID, DSparkDrafter, load_mtp_stage
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

REF = os.environ.get("DSV41_MTP_REF", "/mnt/tt-data/ssinghal/dsv4-spec-accept/mtp_ref.pt")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_drafter_view(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    n = int(os.environ.get("DSV41_NV", "6"))
    U = 4
    B = rows * U
    ref = torch.load(REF)
    S = ref["S"]
    rep = lambda t: t.repeat(2, *([1] * (t.dim() - 1)))[:B]
    log = lambda m: print(m, flush=True)
    sh = _Shards()
    stage_w = [load_mtp_stage(i, sh) for i in range(3)]
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    emb = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=BLOCK * U)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=1e-20)
    dr = DSparkDrafter(md, mc, ccl, stage_w, emb.weight, head, users_per_row=U, n=n)
    ttnn.synchronize_device(md)
    for s, a in enumerate(dr.attn):
        a.seed_ring(rep(ref["ring"][s]), S)
    sm = lambda dim: ttnn.ShardTensor2dMesh(md, dims=(dim, None), mesh_shape=(rows, cols))
    step = ref["steps"][(len(ref["steps"]) - 1) % len(ref["steps"])]
    steps = [ref["steps"][(0 + j) % len(ref["steps"])] for j in range(n)]
    hid_all = torch.stack([rep(s_["main_hidden"]) for s_ in steps], 1)  # [B, n, 15360] (user-major)
    pos_all = torch.full((B, 1), S) + torch.arange(n).reshape(1, n)  # [B, n]
    tok_all = rep(steps[-1]["tok"])  # [B]
    f_all = torch.full((B,), S + n - 1)

    def inputs(dd, Ub):
        """device tensors of drafter ``dd`` for the users u < Ub of every mesh row"""
        sel = [r * U + u for r in range(rows) for u in range(Ub)]
        T_d, T_v = BLOCK * Ub, Ub * n
        hid = hid_all[sel].reshape(len(sel) * n, 15360)
        pos = pos_all[sel].reshape(-1)
        tok = torch.full((rows, BLOCK, Ub), NOISE_ID, dtype=torch.int64)
        tok[:, 0, :] = tok_all[sel].reshape(rows, Ub)
        f = f_all[sel]
        fr = f.reshape(rows, 1, Ub).repeat(1, BLOCK, 1).reshape(rows * T_d)
        h = lambda t, dt, lay, dim: ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=sm(dim), device=md)
        hid_d = h(hid.reshape(1, 1, len(sel) * n, 15360).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, 2)
        posv_d = h(pos.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, 0)
        tok_d = h(tok.reshape(rows * T_d, 1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, 0)
        f_d = h(fr.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, 0)
        return hid_d, posv_d, tok_d, f_d

    def run(dd, Ub):
        hid_d, posv_d, tok_d, f_d = inputs(dd, Ub)
        st_v, st_d = dd.state.build_both(posv_d, f_d)
        dd.write_main(hid_d, st_v)
        out = dd.draft(tok_d, f_d, st=st_d)
        ttnn.synchronize_device(md)
        host_tok = lambda t: torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(-1) for r in range(rows)]
        ).long()
        toks = torch.stack([host_tok(t) for t in out["tokens"]], 1)  # [rows*Ub, 5]
        lgs = [
            ttnn.to_torch(
                out["logits"][i], mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))
            )
            .reshape(rows * Ub, -1)
            .float()
            for i in range(BLOCK)
        ]
        conf = torch.cat(
            [
                ttnn.to_torch(ttnn.get_device_tensors(out["conf"])[r * cols])[0, 0, : BLOCK * Ub, 0]
                .reshape(BLOCK, Ub)
                .T
                for r in range(rows)
            ]
        ).float()  # [rows*Ub, 5]
        return toks, lgs, conf

    ftoks, flgs, fconf = run(dr, U)
    for Ub in [int(x) for x in os.environ.get("VIEW_UBS", "2,1").split(",")]:
        view = dr.view_users(Ub)
        vtoks, vlgs, vconf = run(view, Ub)
        sel = torch.tensor([r * U + u for r in range(rows) for u in range(Ub)])
        eq = (vtoks == ftoks[sel]).float().mean(0).tolist()
        pl = [round(R.pcc(vlgs[i], flgs[i][sel]), 6) for i in range(BLOCK)]
        pc = round(R.pcc(vconf, fconf[sel]), 6)
        log(
            f"VIEWTEST Ub {Ub} T_d {BLOCK * Ub} tokens_equal_per_position {[round(e, 3) for e in eq]} logits_pcc {pl} conf_pcc {pc}"
        )
        assert min(eq) >= 0.99 and min(pl) >= 0.999 and pc >= 0.999, (Ub, eq, pl, pc)
