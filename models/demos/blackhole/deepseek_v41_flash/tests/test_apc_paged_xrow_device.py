# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test of the cross-row hit of the paged APC (DSV41_APC_PAGED_XROW=1): the production B=8 geometry (4 rows x 2 users, single chunk trace, fp8 pool, bounded replay), layers 0-23.

A donor prompt D runs cold on a user of mesh row 0 and its blocks are cached there. Both users of row 0 are then occupied (index reservations), and a prompt P that shares the first ``s`` tokens
of D is planned: the best hit is on row 0, which has no free user, so the index plans a CROSS-ROW hit on another row (fresh pages, donor blocks pinned); the model copies the blocks (host-staged)
and continues P from ``s``. Compared with a cold run of P (and cold-vs-cold in another row as the noise baseline): last-token logits and 8 greedy decode tokens. A third request that finds the same
prefix must now hit LOCALLY on the receiving row.

    DSV41_APC_PAGED=1 DSV41_APC_PAGED_XROW=1 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_paged_xrow_device.py -s
"""

import json
import os
import time

import pytest
import torch

import ttnn

U = 2
CHUNK = 512
LAYERS = os.environ.get("DSV41_PH_LAYERS", "0-23")
DECODE = int(os.environ.get("DSV41_PH_DECODE", "8"))
MAXCTX = int(os.environ.get("DSV41_PH_MAXCTX", "16384"))
CASES = [
    tuple(int(x) for x in c.split(":"))
    for c in os.environ.get("DSV41_XR_CASES", "6000:3072:6000,8000:6144:8400").split(",")
]


def set_env():
    os.environ.update(
        {
            "DSV41_APC_PAGED": "1",
            "DSV41_APC_PAGED_XROW": "1",
            "DSV41_APC_PAGED_XROW_MIN": "1024",
            "DSV41_PF_PAGED": "read",
            "DSV41_PF_SINGLE": "1",
            "DSV41_PF_UMASK": "1",
            "DSV41_PREFILL_UP": str(U),
            "DSV41_PREFILL_UP_BATCH": "1",
            "DSV41_POOL_DTYPE": "fp8",
            "DSV41_FP8_SERVING": "1",
            "DSV41_PF_PAGED_FP8": "1",
            "DSV41_MO_OVERLAP": "0",
        }
    )


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    return dict(
        max=float(d.max()),
        mean=float(d.mean()),
        argmax_equal=int(a.argmax()) == int(b.argmax()),
        bits=bool(torch.equal(a.float(), b.float())),
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_IL_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(9000)
@torch.no_grad()
def test_cross_row_hit_equals_cold(mesh_device):
    set_env()
    from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_model as APM
    from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
    from models.tt_transformers.tt.common import PagedAttentionConfig

    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    pp = default_page_params(MAXCTX, U)
    pp["page_max_num_blocks_per_dp"] += APM.extra_pool_pages(U)
    args, m, pool, _ = create_tt_model(
        md,
        B,
        MAXCTX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    pa = m.enable_apc_paged()
    assert pa is not None and pa.idx.xrow, "paged APC / cross-row switch off (see the log)"
    S_pad = m.single_spad(CHUNK)
    g = torch.Generator().manual_seed(55)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)

    def run(u, P, st, rid):
        e = P.numel()
        t1 = time.perf_counter()
        res = m.prefill_interleaved(
            [(u, P, st, e)], CHUNK, s_pad_max=S_pad, want_logits=True, totals={u: e}, final={u: True}, rids={u: rid}
        )
        m.flush_key_exports(why="test")
        return int(res[u][0]), res[u][1].float().clone(), time.perf_counter() - t1

    def decode(u, first, e, steps=DECODE):
        out, cur = [first], first
        for k in range(steps):
            tok = torch.zeros(B, dtype=torch.long)
            pos = torch.zeros(B, dtype=torch.long)
            tok[u], pos[u] = cur, e + k
            cur = int(m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True).clone()[u])
            out.append(cur)
        return out

    def clean():
        for x in range(B):
            m.release_user(x)
        pa.reset()

    results = []
    for e, s, de in CASES:
        P = rnd(e)
        D = torch.cat([P[:s], rnd(de - s)])
        tag = f"e={e} s={s} donor_e={de}"
        clean()
        t1_, l1, _ = run(0, P, 0, f"c1-{e}")
        clean()
        t1_, l1, dt1 = run(0, P, 0, f"c1b-{e}")  # (clean timing)
        d1 = decode(0, t1_, e)
        clean()
        t2_, l2, dt2 = run(6, P, 0, f"c2-{e}")  # cold in another row: noise baseline
        clean()
        run(0, D, 0, f"donor-{e}")  # donor on row 0, user 0
        m.release_user(0)
        pa.idx.bind("occ0", 0, user=0)
        pa.idx.bind("occ1", 0, user=1)  # row 0 is full
        plan = pa.idx.plan_hit(f"x-{e}", P.numpy(), e)
        log(f"[{tag}] plan: hit {plan.hit_tokens} row {plan.row} user {plan.user} xrow_src {plan.xrow_src}")
        assert plan.hit_tokens >= s - CHUNK and plan.xrow_src == 0 and plan.row != 0, plan
        t3_, l3, dt3 = run(plan.user, P, plan.hit_tokens, f"x-{e}")
        d3 = decode(plan.user, t3_, e)
        log(pa.line())
        assert pa.stats["xrow"] >= 1 and pa.stats["xrow_failed"] == 0, pa.stats
        # a later request with the prefix finds it on the receiving row (user 0 / 1 of row 0 still occupied, the other user of the receiving row is free)
        m.release_user(plan.user)
        pa.idx.release_user(plan.user)
        plan2 = pa.idx.plan_hit(f"l-{e}", P.numpy(), e)
        local = plan2.hit_tokens >= plan.hit_tokens and plan2.xrow_src < 0 and plan2.row == plan.row
        pa.idx.cancel(f"l-{e}")
        pa.idx.release("occ0")
        pa.idx.release("occ1")
        r = dict(
            tag=tag,
            hit=plan.hit_tokens,
            xrow_ms=round(pa.stats["xrow_s"] * 1e3),
            cold_s=round(dt1, 3),
            xrow_hit_s=round(dt3, 3),
            hit_vs_cold=cmp(l3, l1),
            cold_vs_cold=cmp(l1, l2),
            decode_equal=d1 == d3,
            decode_cold=d1,
            decode_xrow=d3,
            second_is_local=bool(local),
        )
        tol = float(os.environ.get("DSV41_PH_TOL_X", "3"))
        r["ok"] = bool(
            r["hit_vs_cold"]["max"] <= tol * max(r["cold_vs_cold"]["max"], 1e-3)
            and r["hit_vs_cold"]["argmax_equal"]
            and local
        )
        log(json.dumps(r))
        results.append(r)
        pa.idx.check()
    clean()
    out = os.environ.get("DSV41_XR_OUT")
    if out:
        json.dump(results, open(out, "w"), indent=1)
    bad = [r for r in results if not r["ok"]]
    assert not bad, [r["tag"] for r in bad]
    log(f"ALL {len(results)} cases: cross-row hit == cold within noise; stats {pa.stats}")
