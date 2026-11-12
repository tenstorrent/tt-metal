# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""(T24 copy-hiding experiments E-A / E-B; derived from test_apc_paged_xrow_device) Device test of the cross-row hit of the paged APC (DSV41_APC_PAGED_XROW=1): the production B=8 geometry (4 rows x 2 users, single chunk trace, fp8 pool, bounded replay), layers 0-23.

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
HIT = int(os.environ.get("DSV41_OV_HIT", "65536"))
CQS = int(os.environ.get("DSV41_OV_CQS", "2"))
REPS = int(
    os.environ.get("DSV41_OV_REPS", "0")
)  # > 0 enables the threaded copy-while-decoding probe (risky: shared CQ0, model not thread-safe; a hang costs a board reset)
DECODE_STEPS = int(os.environ.get("DSV41_OV_STEPS", "60"))
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
                "num_command_queues": CQS,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(9000)
@torch.no_grad()
def test_xrow_copy_phases_and_overlap(mesh_device):
    """E-A: per-phase host-enqueue vs device-wait of xrow_copy (DSV41_XROW_PROFILE=1) with sync every 1 / 8 pieces, bit-identity to cold.
    E-B: (0) which ttnn APIs accept a command queue id, (1) decode step time alone, (2) decode step time while the copy runs in a host thread (shared CQ0), (3) the same copy repeated REPS times for hangs
    / correctness. A CQ1 path for the copy kernels exists only if probe (0) shows cq_id support for every op the copy uses (paged_scatter_rows, key_save, halo_copy are custom ops).
    """
    import threading

    set_env()
    os.environ["DSV41_XROW_PROFILE"] = "1"
    from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_model as APM
    from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
    from models.tt_transformers.tt.common import PagedAttentionConfig

    md = mesh_device
    log = lambda m: print(m, flush=True)
    res = {"cqs": CQS, "hit": HIT}
    # ---- (0) capability probe -------------------------------------------------------------------------------------------------------------------------
    import inspect

    probe = {}
    for name in (
        "record_event",
        "wait_for_event",
        "synchronize_device",
        "from_device",
        "copy_host_to_device_tensor",
        "to_torch",
        "from_torch",
        "slice",
        "embedding",
        "execute_trace",
    ):
        fn = getattr(ttnn, name, None)
        doc = (getattr(fn, "__doc__", "") or "") if fn is not None else ""
        try:
            sig = str(inspect.signature(fn)) if fn is not None else ""
        except (TypeError, ValueError):
            sig = ""
        probe[name] = dict(
            present=fn is not None,
            cq_in_sig=("cq_id" in sig or "queue_id" in sig),
            cq_in_doc=("cq_id" in doc or "queue_id" in doc),
        )
    res["probe"] = probe
    log("PROBE " + json.dumps(probe))
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    ctx = int(os.environ.get("DSV41_PH_MAXCTX", str(HIT + 8192)))
    pp = default_page_params(ctx, U)
    pp["page_max_num_blocks_per_dp"] += APM.extra_pool_pages(U)
    args, m, pool, _ = create_tt_model(
        md,
        B,
        ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    pa = m.enable_apc_paged()
    assert pa is not None and pa.idx.xrow, "paged APC / cross-row switch off"
    S_pad = m.single_spad(CHUNK)
    g = torch.Generator().manual_seed(77)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)
    L = HIT + 4096
    P = rnd(L)
    D = torch.cat([P[:HIT], rnd(L - HIT)])

    def run(u, Pp, st, rid):
        t1 = time.perf_counter()
        r = m.prefill_interleaved(
            [(u, Pp, st, Pp.numel())],
            CHUNK,
            s_pad_max=S_pad,
            want_logits=True,
            totals={u: Pp.numel()},
            final={u: True},
            rids={u: rid},
        )
        m.flush_key_exports(why="test")
        return int(r[u][0]), r[u][1].float().clone(), time.perf_counter() - t1

    def clean():
        for x in range(B):
            m.release_user(x)
        pa.reset()

    def dec_step(u, tok_, pos_):
        tok = torch.zeros(B, dtype=torch.long)
        pos = torch.zeros(B, dtype=torch.long)
        tok[u], pos[u] = tok_, pos_
        return int(m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True).clone()[u])

    # cold reference
    clean()
    t_cold, l_cold, dt_cold = run(0, P, 0, "cold")
    res["cold_prefill_s"] = round(dt_cold, 2)
    logs = []
    for sync_every, batch in (
        (1, 0),
        (8, 0),
        (8, 1),
        (1, 0),
        (8, 0),
        (8, 1),
        (8, 1),
        (8, 0),
    ):  # (the first pass of each program compiles: compare the later ones)
        os.environ["DSV41_XROW_SYNC_EVERY"] = str(sync_every)
        os.environ["DSV41_XROW_BATCH"] = str(batch)
        clean()
        run(0, D, 0, f"donor{sync_every}")
        m.release_user(0)
        pa.idx.bind("occ0", 0, user=0)
        pa.idx.bind("occ1", 0, user=1)
        plan = pa.idx.plan_hit(f"x{sync_every}", P.numpy(), L)
        assert plan.xrow_src == 0 and plan.row != 0 and plan.hit_tokens > 0, plan
        log(f"[E-A sync_every={sync_every}] plan hit {plan.hit_tokens} row {plan.row} user {plan.user}")
        t_x, l_x, dt_x = run(plan.user, P, plan.hit_tokens, f"x{sync_every}")
        log(pa.line())
        ok = bool(
            torch.equal(l_x, l_cold)
            or (float((l_x - l_cold).abs().max()) <= 3e-3 and int(l_x.argmax()) == int(l_cold.argmax()))
        )
        logs.append(
            dict(sync_every=sync_every, xrow_s=round(pa.stats["xrow_s"], 3), call_s=round(dt_x, 2), bit_or_noise_ok=ok)
        )
        log("E-A " + json.dumps(logs[-1]))
        pa.idx.release("occ0")
        pa.idx.release("occ1")
        res.setdefault("E_A", []).append(logs[-1])
        pa.stats["xrow_s"] = 0.0
    # ---- E-B: decode step time alone vs while a copy runs in a host thread ---------------------------------------------------------------------------
    clean()
    d_user = 4  # a decoder on row 2; the copy goes into row 1 / 3 (users 2/3 or 6/7)
    t0_, _, _ = run(d_user, rnd(2048), 0, "dec")
    cur, pos0 = t0_, 2048
    for k in range(5):  # warm the decode trace
        cur = dec_step(d_user, cur, pos0 + k)
    pos0 += 5
    ts = []
    for k in range(DECODE_STEPS):
        t1 = time.perf_counter()
        cur = dec_step(d_user, cur, pos0 + k)
        ts.append(time.perf_counter() - t1)
    pos0 += DECODE_STEPS
    alone = sorted(ts)[len(ts) // 2]
    res["E_B"] = dict(decode_alone_ms_median=round(alone * 1e3, 2))
    log("E-B decode alone median %.2f ms" % (alone * 1e3))
    # prepare a copy: donor on row 0 (users 0/1 occupied), the receiving user in another row
    reps = []
    for rep in range(REPS):
        os.environ["DSV41_XROW_SYNC_EVERY"] = "8"
        # keep the decoder; clean only the copy users
        for x in (0, 1, 2, 3, 6, 7):
            m.release_user(x)
        pa.reset()
        run(0, D, 0, f"bdonor{rep}")
        m.release_user(0)
        pa.idx.bind(f"bocc0{rep}", 0, user=0)
        pa.idx.bind(f"bocc1{rep}", 0, user=1)
        plan = pa.idx.plan_hit(f"bx{rep}", P.numpy(), L)
        if not (plan.xrow_src == 0 and plan.row != 0):
            log(f"E-B rep {rep}: no cross-row plan ({plan}); skipped")
            continue
        # the copy itself runs inside prefill_interleaved (prepare_hits -> xrow_copy); do it in a thread and decode meanwhile
        out = {}

        def copy_job():
            try:
                t1 = time.perf_counter()
                tt_, ll_, _ = run(plan.user, P, plan.hit_tokens, f"bx{rep}")
                out["t"] = time.perf_counter() - t1
                out["ok"] = bool(
                    float((ll_ - l_cold).abs().max()) <= 3e-3 and int(ll_.argmax()) == int(l_cold.argmax())
                )
            except Exception as ex:  # noqa: BLE001
                out["err"] = repr(ex)

        th = threading.Thread(target=copy_job)
        steps = []
        th.start()
        t_start = time.perf_counter()
        while th.is_alive() and time.perf_counter() - t_start < 600:
            t1 = time.perf_counter()
            cur = dec_step(d_user, cur, pos0)
            pos0 += 1
            steps.append(time.perf_counter() - t1)
        th.join(timeout=60)
        r = dict(
            rep=rep,
            copy_call_s=round(out.get("t", -1), 2),
            decode_steps_during=len(steps),
            decode_median_ms=round(sorted(steps)[len(steps) // 2] * 1e3, 2) if steps else None,
            decode_max_ms=round(max(steps) * 1e3, 1) if steps else None,
            copy_ok=out.get("ok"),
            err=out.get("err"),
        )
        log("E-B " + json.dumps(r))
        reps.append(r)
        pa.idx.release(f"bocc0{rep}")
        pa.idx.release(f"bocc1{rep}")
        pa.idx.check()
    res["E_B"]["threaded_copy"] = reps
    out_f = os.environ.get("DSV41_OV_OUT")
    if out_f:
        json.dump(res, open(out_f, "w"), indent=1)
    log("RESULT " + json.dumps(res))
