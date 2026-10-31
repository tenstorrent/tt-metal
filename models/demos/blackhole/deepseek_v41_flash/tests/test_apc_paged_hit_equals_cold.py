# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1b device test of the block-paged APC (tt/apc_paged_model.py): hit == cold at the model level (``prefill_interleaved``), the production B=8 geometry (4 rows x 2 users, Up = U = 2, single chunk
trace, fp8 pool, bounded replay R = 2560).

For every case a prompt P (length e) is run COLD twice (user a, then another user in another mesh row: the cold-vs-cold noise baseline of the Up=2 stack) and once as a HIT: a donor prompt that shares
the first ``s`` tokens with P (then random tokens, so no deeper block matches) is prefilled first (the cache is reset before each case), the index plans the hit of P (must be exactly ``s``), P runs on the
reserved user from ``s``. Compared: the last-token logits (max |diff|, argmax) and 8 greedy decode steps (rings, compressor state, decode key slab all come from the replayed tail).
Cases cover the 512 grid, hit lengths below / at / above the zone start p0(e) = (e - 2560) // 512 * 512 (halo A only / halo B), and ``b_zone < p0_new`` (a donor shorter than P by 512..2560 tokens, hit past p0(P)).

Env: DSV41_PH_LAYERS (default 0-23: both halo parts exist, 3 B layers), DSV41_PH_CASES ("e:s,e:s,..."), DSV41_PH_DECODE (8), DSV41_PH_TOL_X (hit-vs-cold diff allowed as a multiple of cold-vs-cold, default 3),
DSV41_APC_PAGED_ZONE_EQ (strict zone rule: a b_zone < p0 hit becomes a miss -> the case then asserts hit == 0).
    DSV41_APC_PAGED=1 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_paged_hit_equals_cold.py -s
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
# (e, s, donor_e): P has e tokens; the donor shares P[:s] and has donor_e tokens (donor_e - s >= 128 so that a donor block ends at s)
DEFAULT_CASES = "2688:512:2688,4096:1024:4096,4096:3584:4096,6000:2048:6000,6000:3072:6000,6000:3584:6000,8000:6144:6444,8000:7424:7800,9000:8192:8500"
CASES = [tuple(int(x) for x in c.split(":")) for c in os.environ.get("DSV41_PH_CASES", DEFAULT_CASES).split(",")]


def zone_start(e, R=2560):
    return max(0, (e - R) // CHUNK * CHUNK)


def set_env():
    os.environ.update(
        {
            "DSV41_APC_PAGED": "1",
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
def test_hit_equals_cold(mesh_device):
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
    t0 = time.perf_counter()
    args, m, pool, _ = create_tt_model(
        md,
        B,
        MAXCTX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    log(f"model built in {time.perf_counter() - t0:.0f} s (layers {LAYERS})")
    pa = m.enable_apc_paged()
    assert pa is not None, "paged APC refused (see the log)"
    S_pad = m.single_spad(CHUNK)
    g = torch.Generator().manual_seed(77)
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
            nxt = m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True).clone()
            cur = int(nxt[u])
            out.append(cur)
        return out

    def clean():
        for x in range(B):
            m.release_user(x)
        pa.reset()

    results = []
    first_case = True
    for e, s, de in CASES:
        P = rnd(e)
        D = torch.cat([P[:s], rnd(de - s)])
        p0 = zone_start(e)
        tag = f"e={e} s={s} (p0={p0}, {'halo B' if s > p0 else 'halo A only'}) donor_e={de} (p0_donor={zone_start(de)})"
        # cold 1 (user 0) + decode, cold 2 (user 4: other mesh row)
        clean()
        t1_, l1, dt1 = run(0, P, 0, f"cold1-{e}-{s}")
        if first_case:  # the very first call compiles and captures: run again for a clean timing
            clean()
            t1_, l1, dt1 = run(0, P, 0, f"cold1b-{e}-{s}")
        d1 = decode(0, t1_, e)
        clean()
        t2_, l2, dt2 = run(4, P, 0, f"cold2-{e}-{s}")
        # hit: donor on user 2 then P on the reserved user
        clean()
        run(2, D, 0, f"donor-{e}-{s}")
        m.release_user(2)  # (the donor request ends; its blocks stay cached)
        plan = pa.idx.plan_hit(f"hit-{e}-{s}", P.numpy(), e)
        strict = os.environ.get("DSV41_APC_PAGED_ZONE_EQ", "0") == "1"
        expect = s
        if strict and de != e and s > p0 and zone_start(de) != p0:
            expect = 0
        log(
            f"[{tag}] plan: hit {plan.hit_tokens} (expect {expect}) row {plan.row} user {plan.user} halo_a {plan.halo_a} halo_b {plan.halo_b}"
        )
        assert plan.hit_tokens == expect, (tag, plan)
        if plan.hit_tokens:
            t3_, l3, dt3 = run(plan.user, P, plan.hit_tokens, f"hit-{e}-{s}")
            d3 = decode(plan.user, t3_, e)
        else:
            pa.idx.cancel(f"hit-{e}-{s}")
            t3_, l3, dt3, d3 = None, None, 0.0, None
        r = dict(
            tag=tag, e=e, s=s, hit=plan.hit_tokens, cold1_s=round(dt1, 3), cold2_s=round(dt2, 3), hit_s=round(dt3, 3)
        )
        r["cold_vs_cold"] = cmp(l1, l2)
        r["tok_cold"] = (t1_, t2_)
        if l3 is not None:
            r["hit_vs_cold"] = cmp(l3, l1)
            r["tok_hit"] = t3_
            r["decode_equal"] = d3 == d1
            r["decode_cold"], r["decode_hit"] = d1, d3
            tol = float(os.environ.get("DSV41_PH_TOL_X", "3"))
            base = max(r["cold_vs_cold"]["max"], 1e-3)
            r["ok"] = bool(r["hit_vs_cold"]["max"] <= tol * base and r["hit_vs_cold"]["argmax_equal"])
        log(json.dumps(r))
        log(pa.line())
        results.append(r)
        first_case = False
        pa.idx.check()
    clean()
    pa.idx.check()
    out = os.environ.get("DSV41_PH_OUT")
    if out:
        json.dump(results, open(out, "w"), indent=1)
    bad = [r for r in results if r.get("hit") and not r.get("ok")]
    assert not bad, f"hit != cold beyond the cold-vs-cold noise in {len(bad)} case(s): " + "; ".join(
        r["tag"] for r in bad
    )
    log(f"ALL {len(results)} cases: hit == cold within noise; stats {pa.stats}")
