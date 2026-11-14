# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the verify device cuts (note 42): QWEN36_VERIFY_DAR, QWEN36_VERIFY_ATTN_LEAN, QWEN36_VERIFY_LAYOUT_CUTS.

Host-checkable parts only, over a recording fake of the few ttnn calls involved:
* the three flags (default on, "0" off) and the QWEN36_VERIFY_SDPA_MAX_CORES knob;
* the fused K+V group write issues T paged_fused_update_cache calls (not 2T paged_update_cache), pads nothing, and
  takes group j's positions from the shared per-trace slices when given;
* the attention dispatch (_verify_lean_eligible) and the core-disjointness rule of the fused update;
* the model dispatch: _verify_dar gating and the replicated tail.
The device numerics are checked by the device tests listed in note 42. No device. Run:
pytest models/demos/blackhole/qwen36/tests/test_spec_verifycut_host.py -q
"""

import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


class FT:
    """Fake tensor: just a shape and a name."""

    def __init__(self, shape, name="t"):
        self.shape = tuple(shape)
        self.name = name

    def buffer_address(self):
        return id(self)


@pytest.fixture
def rec(monkeypatch):
    calls = []

    def op(name, ret=None):
        def f(*a, **k):
            calls.append((name, a, k))
            return ret(*a, **k) if ret is not None else FT((1,), name)

        return f

    monkeypatch.setattr(ttnn, "reshape", op("reshape", lambda t, s, p=None, **k: FT(tuple(s), "view")))
    monkeypatch.setattr(ttnn, "slice", op("slice"))
    monkeypatch.setattr(ttnn, "to_memory_config", op("to_memory_config"))
    monkeypatch.setattr(ttnn, "deallocate", op("deallocate", lambda *a, **k: None))
    monkeypatch.setattr(ttnn, "pad", op("pad"))
    monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", op("fused", lambda *a, **k: None), raising=False)
    monkeypatch.setattr(ttnn.experimental, "paged_update_cache", op("update", lambda *a, **k: None), raising=False)
    monkeypatch.setattr(ttnn, "Shape", lambda s: tuple(s), raising=False)
    return calls


def _attn(n_users_ok=True):
    a = TPAttention.__new__(TPAttention)
    a.HD, a.NH, a.NKV = 256, 3, 1
    a.mesh = types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=11, y=10))
    a.paged_k, a.paged_v = FT((1,), "pk"), FT((1,), "pv")
    a.B = 8
    a._kv_key_shard_cfg = lambda n: ("kcfg", n)
    a._kv_shard_cfg = lambda n: ("vcfg", n)
    return a


def test_flags_default_on_and_off(monkeypatch):
    for name, fn in (
        ("QWEN36_VERIFY_DAR", tpc.verify_dar_enabled),
        ("QWEN36_VERIFY_ATTN_LEAN", tpc.verify_attn_lean_enabled),
        ("QWEN36_VERIFY_LAYOUT_CUTS", tpc.verify_layout_cuts_enabled),
    ):
        monkeypatch.delenv(name, raising=False)
        assert fn() is True
        monkeypatch.setenv(name, "0")
        assert fn() is False
        monkeypatch.setenv(name, "1")
        assert fn() is True


@pytest.mark.parametrize("n,ok", [(1, True), (2, True), (4, True), (8, True), (16, False)])
def test_fused_update_needs_disjoint_k_and_v_cores(n, ok):
    assert _attn()._kv_fused_cores_ok(n) is ok


@pytest.mark.parametrize("n,T", [(8, 4), (1, 4), (2, 4)])
@pytest.mark.parametrize("shared_pos", [True, False])
def test_group_write_is_T_fused_calls_without_pad(rec, n, T, shared_pos):
    a, B = _attn(), n * T
    pos = [FT((n,), f"pos{j}") for j in range(T)] if shared_pos else None
    pt = FT((n, 4), "pt")
    a._write_kv_fused_groups(FT((1, B, 1, 256), "k"), FT((1, B, 1, 256), "v"), FT((B,), "cur"), n, pt, pos, B)
    names = [c[0] for c in rec]
    assert names.count("fused") == T and "update" not in names and "pad" not in names
    fused = [c for c in rec if c[0] == "fused"]
    for j, (_, args, kw) in enumerate(fused):
        assert kw["page_table"] is pt
        assert kw["update_idxs_tensor"] is (pos[j] if shared_pos else kw["update_idxs_tensor"])
        assert args[0] is a.paged_k and args[2] is a.paged_v
    # shared positions: no position slice per group; otherwise one strided slice per group
    pos_slices = [c for c in rec if c[0] == "slice" and len(c[1][1]) == 1]
    assert len(pos_slices) == (0 if shared_pos else T)
    # K goes to the key cores, V to the value cores
    cfgs = [c[1][1] for c in rec if c[0] == "to_memory_config"]
    assert cfgs == [("kcfg", n), ("vcfg", n)] * T


def test_lean_dispatch_rules(monkeypatch):
    monkeypatch.delenv("QWEN36_VERIFY_ATTN_LEAN", raising=False)  # the tests set their own flag values
    a = _attn()
    a.use_paged, a._fused_qkv, a._use_nlp_decode_heads = True, True, True
    a._spec_sdpa_plan = lambda T, n=1: ("cfg", n * (2 if T == 4 else 1), 4) if T in (4, 8) else None
    import models.demos.blackhole.qwen36.tt.attention.tp as tp

    monkeypatch.setattr(tp, "KV_GROUP_WRITE_OK", True)
    upt, spt = FT((8, 4)), FT((16, 4))
    assert a._verify_lean_eligible(32, True, 8, upt, spt) is True
    monkeypatch.setenv("QWEN36_VERIFY_ATTN_LEAN", "0")
    assert a._verify_lean_eligible(32, True, 8, upt, spt) is False
    monkeypatch.delenv("QWEN36_VERIFY_ATTN_LEAN")
    assert a._verify_lean_eligible(32, False, 8, upt, spt) is False  # no page table
    assert a._verify_lean_eligible(32, True, 8, None, spt) is False  # no per-user table
    assert a._verify_lean_eligible(32, True, 8, upt, FT((3, 4))) is False  # wrong group-table rows
    assert a._verify_lean_eligible(21, True, 7, FT((7, 4)), spt) is False  # T=3: no SDPA plan
    assert a._verify_lean_eligible(8, True, 8, upt, spt) is False  # T=1
    monkeypatch.setattr(tp, "KV_GROUP_WRITE_OK", False)
    assert a._verify_lean_eligible(32, True, 8, upt, spt) is False  # no strided slice


def _model(fused=(True, True), decode_ar=object()):
    m = Qwen36Model.__new__(Qwen36Model)
    m._decode_ar = decode_ar
    gdn = [
        types.SimpleNamespace(is_full_attention=False, attention=types.SimpleNamespace(_spec_fused=f)) for f in fused
    ]
    m.layers = gdn + [types.SimpleNamespace(is_full_attention=True, attention=types.SimpleNamespace())]
    return m


def test_verify_dar_gating(monkeypatch):
    monkeypatch.delenv("QWEN36_VERIFY_DAR", raising=False)
    m = _model()
    assert m._verify_dar(32) is m._decode_ar
    assert m._verify_dar(33) is None  # more than one RES40 tile of rows
    assert _model(decode_ar=None)._verify_dar(32) is None  # QWEN36_DECODE_ALLREDUCE=0 / unsupported mesh
    assert _model(fused=(True, False))._verify_dar(32) is None  # an unfused GDN layer keeps the fractured verify
    monkeypatch.setenv("QWEN36_VERIFY_DAR", "0")
    assert m._verify_dar(32) is None


def test_spec_tail_dispatches_on_the_residual_width(monkeypatch):
    m = _model()
    m.args = types.SimpleNamespace(dim=5120)
    seen = []
    m._spec_tail_replicated = lambda x, sharded: seen.append((x, sharded)) or "replicated"
    assert m._spec_tail(FT((1, 1, 32, 5120)), sharded=True) == "replicated" and seen[0][1] is True
    # a fractured residual must NOT take the replicated tail (it reaches the clone of the old tail instead)
    monkeypatch.setattr(ttnn, "clone", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("old tail")))
    try:
        m._spec_tail(FT((1, 1, 32, 640)))
    except RuntimeError as e:
        assert "old tail" in str(e)
    else:
        raise AssertionError("fractured residual skipped the old tail")
