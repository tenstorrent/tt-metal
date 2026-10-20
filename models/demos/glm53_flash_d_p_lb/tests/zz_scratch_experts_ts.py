# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: GLM-5.3-Flash routed experts, the K3 token-split block (GLM_EXPERTS_MODE=ts, tt/experts_ts.py) vs the
all-gather path (tt/experts_ag.py) vs a CPU fp32 reference, without the checkpoint: random experts (a pool of P
distinct ones, expert g = pool[g % P]), random tokens and top-8 routing (uniform, or one hot expert every token
picks), the split layout's full-mesh input (each chip its own S/n rows). Per-token rel L2 against the reference on
a token subset, and ts against ag on all tokens.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_FABRIC_PAYLOAD=8704 \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_experts_ts.py -s

GLM_TS_CHUNK (default 5120), GLM_TS_POOL (default 4), GLM_TS_ROUTINGS (default uniform,hot). GLM_TS_SHARED=1: the
shared expert (random, I_s 2048) folded into the ts block (TP over each mesh row, partials through the combine); the
reference then includes it, and ag (routed only) is not compared."""

import os

import torch
from loguru import logger

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S_ = spec()
pytestmark = device_timeout(S_)
H, I, E, K = 4096, 2048, 288, 8
T = int(os.environ.get("GLM_TS_CHUNK", "5120"))
SHARED = os.environ.get("GLM_TS_SHARED") == "1"
IS = 2048
P = int(os.environ.get("GLM_TS_POOL", "4"))


class _Pool:
    def __init__(self, pool):
        self.pool = pool

    def __len__(self):
        return E

    def __getitem__(self, g):
        gw, uw, dw = self.pool[g % P]
        return {"gate_proj": gw, "up_proj": uw, "down_proj": dw}


def _routing(name, g):
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    if name == "hot":  # (expert 0 in every token's top-8)
        has = (ids == 0).any(1)
        ids[~has, 0] = 0
    w = torch.rand(T, K, generator=g) + 0.1
    w = w / w.sum(1, keepdim=True) * 2.5
    return ids, w


def _reference(x, ids, w, pool, rows):
    out = torch.zeros(len(rows), H)
    for j, t in enumerate(rows):
        for k in range(K):
            gw, uw, dw = pool[int(ids[t, k]) % P]
            g_, u_ = x[t] @ gw.T, x[t] @ uw.T
            a = torch.nn.functional.silu(g_.clamp(max=10.0)) * u_.clamp(-10.0, 10.0)
            out[j] += float(w[t, k]) * (a @ dw.T)
    return out


def _shared_ref(x, se, rows):
    gw, uw, dw = se
    xs = x[rows]
    a = torch.nn.functional.silu((xs @ gw.T).clamp(max=10.0)) * (xs @ uw.T).clamp(-10.0, 10.0)
    return a @ dw.T


def _rel(got, want):
    d = (got.double() - want.double()).norm(dim=-1) / want.double().norm(dim=-1).clamp(min=1e-30)
    return d.mean().item(), d.max().item()


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


@mesh_parametrize
def test_experts_ts(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.tt.common import split_from_host, split_to_host
    from models.demos.glm53_flash_d_p.tt.experts_ag import TtExpertsAg
    from models.demos.glm53_flash_d_p.tt.experts_ts import TtExpertsTs

    hooks.apply_device_settings(S_)
    g = torch.Generator().manual_seed(0)
    pool = [
        (
            torch.randn(I, H, generator=g) * 0.02,
            torch.randn(I, H, generator=g) * 0.02,
            torch.randn(H, I, generator=g) * 0.02,
        )
        for _ in range(P)
    ]
    tw = _Pool(pool)
    kw = dict(num_experts=E, emb_dim=H, hidden_dim=I, top_k=K, max_seq_len=T, weights_dtype=ttnn.bfloat4_b, cache=False)
    se = (
        (
            torch.randn(IS, H, generator=g) * 0.02,
            torch.randn(IS, H, generator=g) * 0.02,
            torch.randn(H, IS, generator=g) * 0.02,
        )
        if SHARED
        else None
    )
    ts = TtExpertsTs(mesh_device, 4, tw, num_links=2, shared_weights=se, **kw)  # (first: plan options process-wide)
    ag = TtExpertsAg(mesh_device, 4, tw, num_links=2, **kw) if not SHARED else None
    # (shared: the same routed experts without it, to isolate the shared expert's part: folded - routed only)
    tr = TtExpertsTs(mesh_device, 4, tw, num_links=2, **kw) if SHARED else None
    for routing in os.environ.get("GLM_TS_ROUTINGS", "uniform,hot").split(","):
        x = (torch.randn(T, H, generator=g)).to(torch.bfloat16)
        ids, w = _routing(routing, g)
        outs = {}
        for name, mod in (("ts", ts), ("ag", ag)) if not SHARED else (("ts", ts), ("tr", tr)):
            xd = split_from_host(mesh_device, x.view(1, 1, T, H))
            idd = split_from_host(mesh_device, ids.view(1, 1, T, K).to(torch.int32), dtype=ttnn.uint16)
            wd = split_from_host(mesh_device, w.view(1, 1, T, K).to(torch.bfloat16))
            y = mod(xd, idx=idd, wts=wd, split=True, full_mesh=True)
            y = mod(xd, idx=idd, wts=wd, split=True, full_mesh=True)  # (second call: the ts block's arena check)
            outs[name] = split_to_host(y).reshape(T, H).float()
            for t_ in (xd, idd, wd):
                ttnn.deallocate(t_)
        rows = list(range(0, T, T // 256))
        ref = _reference(x.float(), ids, w, pool, rows)
        if SHARED:
            sref = _shared_ref(x.float(), se, rows)
            ref = ref + sref
            m, mx = _rel(outs["ts"][rows] - outs["tr"][rows], sref)
            logger.info(
                f"TS {routing}: shared part (folded - routed only) vs CPU fp32 on {len(rows)} tokens: rel L2 mean "
                f"{m:.5f} max {mx:.5f}, PCC {_pcc(outs['ts'][rows] - outs['tr'][rows], sref):.6f}"
            )
            own = [t for t in rows if (t // (T // 8)) % 4 == 0]  # (tokens of column-0 chips: their own partial)
            m2, _ = _rel(outs["ts"][own] - outs["tr"][own], _shared_ref(x.float(), se, own))
            logger.info(f"TS {routing}: shared part on column-0 chips' tokens: rel L2 mean {m2:.5f}")
        for name in ("ts", "ag") if not SHARED else ("ts",):
            m, mx = _rel(outs[name][rows], ref)
            logger.info(
                f"TS {routing}: {name} vs CPU fp32 on {len(rows)} tokens: rel L2 mean {m:.4f} max {mx:.4f}, "
                f"PCC {_pcc(outs[name][rows], ref):.6f}, finite {bool(torch.isfinite(outs[name]).all())}"
            )
        if not SHARED:
            m, mx = _rel(outs["ts"], outs["ag"])
            logger.info(
                f"TS {routing}: ts vs ag on {T} tokens: rel L2 mean {m:.4f} max {mx:.4f}, "
                f"PCC {_pcc(outs['ts'], outs['ag']):.6f}"
            )
        if os.environ.get("GLM_TS_PROF") == "1":  # (host time of the ts call, no syncs: where it goes)
            import cProfile
            import pstats
            import time as _t

            xd = split_from_host(mesh_device, x.view(1, 1, T, H))
            idd = split_from_host(mesh_device, ids.view(1, 1, T, K).to(torch.int32), dtype=ttnn.uint16)
            wd = split_from_host(mesh_device, w.view(1, 1, T, K).to(torch.bfloat16))
            ttnn.deallocate(ts(xd, idx=idd, wts=wd, split=True, full_mesh=True))  # (warm: a rebuild after ag's L1)
            ttnn.synchronize_device(mesh_device)
            pr = cProfile.Profile()
            t0 = _t.perf_counter()
            pr.enable()
            for _ in range(5):
                ttnn.deallocate(ts(xd, idx=idd, wts=wd, split=True, full_mesh=True))
            pr.disable()
            t1 = _t.perf_counter()
            ttnn.synchronize_device(mesh_device)
            logger.info(
                f"TS host {(t1 - t0) / 5 * 1e3:.1f} ms per call (dispatch only), {(_t.perf_counter() - t0) / 5 * 1e3:.1f} with the sync"
            )
            pstats.Stats(pr).sort_stats("cumulative").print_stats(30)
            if os.environ.get("GLM_TS_HOST") == "1":  # host enqueue vs untraced device vs traced, per call
                from models.demos.glm53_flash_d_p.tt import experts_ts as _ets

                call = lambda: ttnn.deallocate(ts(xd, idx=idd, wts=wd, split=True, full_mesh=True))  # noqa: E731
                for b in _ets._BLOCKS.values():  # (K3_HOST_LOG: per-stage means over the isolated calls only)
                    b.__dict__.pop("_host", None)
                _ets._HOST.clear()
                iso = []
                for _ in range(84):  # host time with an idle device (no queue back-pressure)
                    ttnn.synchronize_device(mesh_device)
                    s0 = _t.perf_counter()
                    call()
                    iso.append(_t.perf_counter() - s0)
                ttnn.synchronize_device(mesh_device)
                iso.sort()
                t0 = _t.perf_counter()
                for _ in range(200):
                    call()
                t1 = _t.perf_counter()
                ttnn.synchronize_device(mesh_device)
                t2 = _t.perf_counter()
                tr = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                yt = ts(xd, idx=idd, wts=wd, split=True, full_mesh=True)
                ttnn.end_trace_capture(mesh_device, tr, cq_id=0)
                ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=True)
                t3 = _t.perf_counter()
                for _ in range(200):
                    ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                t4 = _t.perf_counter()
                ttnn.release_trace(mesh_device, tr)
                ttnn.deallocate(yt)
                logger.info(
                    f"TS host: enqueue with an idle device {1e3 * iso[len(iso) // 2]:.2f} ms median "
                    f"({1e3 * iso[0]:.2f} min); back-to-back untraced {1e3 * (t2 - t0) / 200:.2f} ms per call "
                    f"(host loop {1e3 * (t1 - t0) / 200:.2f}, final sync {1e3 * (t2 - t1):.1f} ms); "
                    f"traced replay {1e3 * (t4 - t3) / 200:.2f} ms per call"
                )
                spy = os.environ.get("GLM_TS_SPY")  # (py-spy --native over isolated calls)
                if spy:
                    import signal
                    import subprocess

                    p = subprocess.Popen(
                        [
                            "py-spy",
                            "record",
                            "--native",
                            "--format",
                            "raw",
                            "-r",
                            "200",
                            "-o",
                            spy,
                            "-p",
                            str(os.getpid()),
                        ]
                    )
                    _t.sleep(2)
                    for _ in range(300):
                        ttnn.synchronize_device(mesh_device)
                        call()
                    ttnn.synchronize_device(mesh_device)
                    p.send_signal(signal.SIGINT)
                    p.wait(timeout=120)
        assert torch.isfinite(outs["ts"]).all()
        assert _pcc(outs["ts"][rows], ref) > 0.98
