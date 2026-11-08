# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the stage/enqueue vs sync/read split of the verify (Qwen36Model) and propose (MTPSpecEngine) flows.

The reference is the pre-split code, loaded from ``git show HEAD:<file>`` (skipped when git or HEAD is missing): the
split pieces, composed in order, must make the identical sequence of device calls (copy_host_to_device_tensor,
execute_trace, synchronize_device, from_device, to_torch) and return identical outputs. Also checked: the
``after_external_sync`` / ``sync=False`` variants (the caller syncs) differ from the composed flow only by who calls
synchronize_device, and the GDN parity flips exactly once per verify replay.
No device.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_stage_split_host.py -q
"""

import ast
import os
import subprocess

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_hostset_host import _drafter_engine, _hostset_vfy_model
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _Buf, _fake_ttnn, _vfy_model
from models.demos.blackhole.qwen36.tt import model as model_mod
from models.demos.blackhole.qwen36.tt import spec_engine as se_mod
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

_QWEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _old_funcs(rel, cls_name, names, module):
    """Pre-split methods of ``cls_name`` from git HEAD, compiled in ``module``'s namespace."""
    try:
        top = subprocess.check_output(["git", "-C", _QWEN, "rev-parse", "--show-toplevel"], text=True).strip()
        full = os.path.relpath(os.path.join(_QWEN, rel), top)
        src = subprocess.check_output(["git", "-C", top, "show", f"HEAD:{full}"], text=True, stderr=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"git HEAD source unavailable: {e}")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    out = {}
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            mod = ast.Module(body=[n], type_ignores=[])
            ns = dict(module.__dict__)
            exec(compile(mod, f"HEAD:{full}", "exec"), ns)
            out[n.name] = ns[n.name]
    assert set(out) == set(names), f"missing {set(names) - set(out)}"
    return out


class _Rec:
    """Records the device-call sequence with content fingerprints."""

    def __init__(self):
        self.log = []


def _install(monkeypatch, log, model):
    """_fake_ttnn plus from_device (non-blocking host copy) and a to_torch that logs; content-aware H2D log."""
    _fake_ttnn(monkeypatch, log, model)

    def copy(h, buf):
        buf.data, buf.dtype, buf.layout = h.t, h.dtype, h.layout
        buf.writes += 1
        log.append(("h2d", buf.name, buf.data.clone().reshape(-1).tolist(), str(h.dtype)))

    def from_device(t, blocking=True):
        log.append(("from_device", t.name, blocking))
        return t

    def to_torch(t, *a, **k):
        log.append(("to_torch", t.name))
        return t.data

    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", copy)
    monkeypatch.setattr(ttnn, "from_device", from_device)
    monkeypatch.setattr(ttnn, "to_torch", to_torch)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def _verify_model(B, T, fused, hostset, sharded):
    if hostset:
        m, bufs = _hostset_vfy_model(B, T, fused=fused)
        m._vfy_sharded, m._vfy_logits_trace_id = sharded, (8 if sharded else None)
    else:
        m, bufs = _vfy_model(B, T, fused=fused, sharded=sharded)
    m._vfy_topk_trace_id = 9
    m._vfy_narrow_ready = False
    m.num_devices, m._vfy_topk_m = 2, 4
    m._vfy_topk_vals_out, m._vfy_topk_ids_out = _Buf("tv"), _Buf("ti")
    m._vfy_topk_vals_out.data = torch.arange(32 * 8, dtype=torch.float32).reshape(1, 1, 32, 8)
    m._vfy_topk_ids_out.data = torch.arange(32 * 8, dtype=torch.int32).reshape(1, 1, 32, 8)
    return m


KW = [{}, {"read_logits": True}, {"read_topk": True}]


@pytest.mark.parametrize("hostset", [True, False])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("sharded", [True, False])
@pytest.mark.parametrize("kw", KW, ids=["ids", "logits", "topk"])
def test_verify_split_matches_old(monkeypatch, kw, sharded, fused, hostset):
    old = _old_funcs("tt/model.py", "Qwen36Model", ["verify_traced"], model_mod)["verify_traced"]
    B, T = 2, 4
    tokens = [[u * 10 + j for j in range(T)] for u in range(B)]
    pt = torch.arange(1, B * 8 + 1, dtype=torch.int32).reshape(B, 8)
    steps = [([0, 1], [100, 200], pt), ([3, 0], [104, 203], pt), ([1, 2], [90, 80], pt + 1)]
    res = {}
    for mode in ("old", "composed", "staged", "external"):
        m = _verify_model(B, T, fused, hostset, sharded)
        log = []
        _install(monkeypatch, log, m)
        outs = []
        for mi, pos, tbl in steps:
            if mode == "old":
                r = old(m, tokens, pos, mi, page_tables=tbl, **kw)
            elif mode == "composed":
                r = m.verify_traced(tokens, pos, mi, page_tables=tbl, **kw)
            elif mode == "staged":
                mark = len(log)
                h = m.verify_stage_and_enqueue(tokens, pos, mi, page_tables=tbl, **kw)
                assert not any(x[0] in ("sync", "to_torch") for x in log[mark:]), "stage must not sync or read"
                r = m.verify_sync_and_read(h)
            else:  # the caller syncs itself, then reads
                h = m.verify_stage_and_enqueue(tokens, pos, mi, page_tables=tbl, **kw)
                ttnn.synchronize_device(m.device)
                r = m.verify_after_external_sync(h)
            lt = r[2]
            lt = tuple(x.clone() for x in lt) if isinstance(lt, tuple) else (None if lt is None else lt.clone())
            outs.append((r[0], r[1], lt, m._vfy_last_col0, [dn._spec_win_par for dn in m._vfy_gdn]))
        res[mode] = (log, outs)
    base_log, base_out = res["old"]
    for mode in ("composed", "staged", "external"):
        assert res[mode][0] == base_log, mode
        assert len(res[mode][1]) == len(base_out)
        for a, b in zip(res[mode][1], base_out):
            assert a[0] == b[0] and a[1] == b[1] and a[3] == b[3] and a[4] == b[4]
            if isinstance(a[2], tuple):
                assert all(torch.equal(x, y) for x, y in zip(a[2], b[2]))
            elif a[2] is None:
                assert b[2] is None
            else:
                assert torch.equal(a[2], b[2])
    if fused:  # parity: one flip per replay
        assert base_out[-1][4] == [len(steps) % 2] * 3
    # the trace calls are enqueued before the (single) sync, and the sync is the last device call before reads
    kinds = [x[0] for x in base_log]
    assert kinds.count("sync") == len(steps)


def test_verify_stage_validation_runs_before_any_device_call(monkeypatch, expect_error):
    m = _verify_model(2, 4, True, True, False)
    log = []
    _install(monkeypatch, log, m)
    with expect_error(ValueError, "exclusive reads"):
        m.verify_stage_and_enqueue([[1] * 4] * 2, [1, 2], [0, 0], read_logits=True, read_topk=True)
    with expect_error(AssertionError, "expected 2 users, got 3"):
        m.verify_stage_and_enqueue([[1] * 4] * 3, [1, 2], [0, 0])
    assert log == []


# --------------------------------------------------------------------------- #
# propose
# --------------------------------------------------------------------------- #
def _engine(B, K, packed, monkeypatch, log, active=None):
    nb = 8
    e = _drafter_engine(B, K, packed=packed, nb=nb)
    e.model._vfy_gdn = []
    e.model._fit_spec_page_table = lambda pt, n: torch.as_tensor(pt, dtype=torch.int32)
    e.model.prestage_verify_aux = lambda pos, mi: log.append(("prestage", list(pos), list(mi)))
    e.captured, e._vfeed, e.recorded_drafter = True, "vfeed", True
    e.eager_reseed_len = 10**9
    e._phys_perm = None
    e._last_pt, e._last_cap_blocks = None, nb
    e._row_active = active
    e._scratch_block = 5000
    del e.__dict__["propose"]  # _propose_engine installs a fake
    del e.__dict__["_ids_to_host"]  # the real reader: from_device x K, one synchronize, to_torch x K
    e._dr_outs = [_Buf(f"out{k}") for k in range(K)]
    for k, o in enumerate(e._dr_outs):
        o.data = torch.arange(32, dtype=torch.int32) * (k + 3) % 1000
    e._no_draft = set()
    e._stage_cache = {}
    e._mtp_pt_rows = [_Buf(f"ptrow{u}") for u in range(B)]
    e._hp_buf = "hp"
    # eager-path device pieces: logged stand-ins (their own internals are not under test)
    e._set_anchor = lambda feed, m: log.append(("set_anchor", list(m)))
    e._reseed_mtp = lambda ct, cp, mkv: log.append(("reseed", [int(x) for x in mkv]))
    e._reseed_mtp_batched = lambda ct, cp, mkv, pt: log.append(
        ("reseed_b", [int(x) for x in mkv], pt.reshape(-1).tolist())
    )
    e._draft = lambda pend, hp, p: (
        log.append(("draft", [int(x) for x in pend], [int(x) for x in p]))
        or torch.arange(B * K, dtype=torch.int32).reshape(B, K) + 1
    )
    _install(monkeypatch, log, e.model)
    monkeypatch.setattr(ttnn, "execute_trace", lambda *a, **k: log.append(("trace", a[1])))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda d: log.append(("sync",)))
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t])
    return e


def _runner_args(B, K, g, acc=None, bad_fit=()):
    T = K + 1
    acc = torch.randint(1, T + 1, (B,), generator=g) if acc is None else torch.tensor(acc)
    cpos = torch.randint(1, 300, (B, 1), generator=g) + torch.arange(T)
    for u in bad_fit:
        cpos[u] += 8 * 64  # window past the table
    ctok = torch.randint(0, 900, (B, T), generator=g)
    for u in range(B):
        ctok[u, int(acc[u]) :] = -1
        cpos[u, int(acc[u]) :] = -1
    pt = torch.randint(1, 4000, (B, 8), generator=g, dtype=torch.int32)
    return ctok, cpos, acc, pt


def _eq(a, b):
    return (
        torch.equal(a.draft_token_ids, b.draft_token_ids)
        and torch.equal(a.num_valid, b.num_valid)
        and a.draft_token_ids.dtype == b.draft_token_ids.dtype
    )


class _OldEngine(MTPSpecEngine):
    pass


def _old_engine_cls():
    names = ["propose", "propose_draft_tokens", "_propose_recorded", "_propose_recorded_packed"]
    funcs = _old_funcs("tt/spec_engine.py", "MTPSpecEngine", names, se_mod)
    ns = dict(funcs)
    return type("OldEngine", (MTPSpecEngine,), ns)


@pytest.mark.parametrize("B", [1, 3, 8])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("sc", ["plain", "declined", "padding", "nofit", "all_declined", "skip_keep"])
def test_propose_split_matches_old(monkeypatch, B, packed, sc):
    Old = _old_engine_cls()
    K = 3
    seqs = {}
    for mode in ("old", "composed", "staged", "external"):
        g = torch.Generator().manual_seed(B * 7 + 1)
        log = []
        active = [u % 3 != 2 for u in range(B)] if sc == "padding" else None
        e = _engine(B, K, packed, monkeypatch, log, active)
        e.__class__ = Old if mode == "old" else MTPSpecEngine
        outs = []
        for step in range(6):
            bad = [0] if sc == "nofit" and step % 2 else []
            acc = [1] * B if sc == "all_declined" and B == 1 else None
            ctok, cpos, accv, pt = _runner_args(B, K, g, acc=acc, bad_fit=bad)
            if sc == "all_declined":
                bad = list(range(B))
                cpos = cpos + 8 * 64
            if sc == "skip_keep":
                e._no_draft = {u for u in range(B) if (u + step) % 2 == 0}
            if mode in ("old", "composed"):
                r = e.propose_draft_tokens(K, ctok, cpos, accv, pt)
            else:
                plan = e._propose_plan(K, ctok, cpos, accv, pt)
                e._propose_stage_and_enqueue(plan)
                if mode == "staged":
                    drafts = e._propose_read(plan)
                else:
                    handles = e._propose_enqueue_reads(plan)
                    if not plan.skip_all and plan.core.recorded:
                        ttnn.synchronize_device(e.mesh)
                    drafts = e._propose_read_from_handles(plan, handles)
                r = e._propose_finish(plan, drafts)
            outs.append((r, sorted(e._offered), e.skipped_propose_calls, e.mtp_extra_steps))
        seqs[mode] = (log, outs, [b.data for b in e._bufs.values() if b.data is not None])
    base = seqs["old"]
    for mode in ("composed", "staged"):
        assert seqs[mode][0] == base[0], mode
        for (r, *rest), (r0, *rest0) in zip(seqs[mode][1], base[1]):
            assert _eq(r, r0) and rest == rest0
    # external sync: identical except the synchronize moves from inside the read to just before it (the caller's)
    nosync = lambda lg: [x for x in lg if x[0] != "sync"]  # noqa: E731
    assert nosync(seqs["external"][0]) == nosync(base[0])
    assert len([x for x in seqs["external"][0] if x[0] == "sync"]) == len([x for x in base[0] if x[0] == "sync"])
    for (r, *rest), (r0, *rest0) in zip(seqs["external"][1], base[1]):
        assert _eq(r, r0) and rest == rest0
    assert any(x[0] == "trace" for x in base[0]) or sc in ("all_declined", "skip_keep")


def test_propose_external_sync_read_has_no_sync_of_its_own(monkeypatch):
    log = []
    e = _engine(2, 3, True, monkeypatch, log)
    g = torch.Generator().manual_seed(5)
    ctok, cpos, acc, pt = _runner_args(2, 3, g)
    plan = e._propose_plan(3, ctok, cpos, acc, pt)
    n = len(log)
    assert n == 0, "planning makes no device call"
    e._propose_stage_and_enqueue(plan)
    assert [x[0] for x in log if x[0] in ("sync", "to_torch")] == [], "stage must not sync or read"
    assert log[-1][0] == "prestage" or any(x[0] == "trace" for x in log)
    mark = len(log)
    handles = e._propose_enqueue_reads(plan)
    assert [x[0] for x in log[mark:]] == ["from_device"] * 3
    e._propose_read_from_handles(plan, handles)
    assert [x[0] for x in log[mark:]] == ["from_device"] * 3 + ["to_torch"] * 3
    mark = len(log)
    e._propose_read(plan)
    assert [x[0] for x in log[mark:]] == ["from_device"] * 3 + ["sync"] + ["to_torch"] * 3
