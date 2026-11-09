# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of QWEN36_GLUE (r12r decode-glue cuts; no device). Flag parsing, the eligibility guards, and that
each cut routes to the layout-free op only when on (flag off = today's calls). Device byte identity of the cuts:
ttsim glue_sim (kvwo: RES40-out wo_pair == 84-core matmul + reshard; guin: dsmc gate|up on the RES40 shard ==
resharded act_shard_hidden in0; both bit for bit)."""

import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import mlp as mlp_mod
from models.demos.blackhole.qwen36.tt import tp_common as tpc


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(tpc.GLUE_ENV, raising=False)
    monkeypatch.setattr(ttnn, "TILE_SIZE", 32, raising=False)
    tpc._RES40_PC_CACHE.clear()
    yield
    tpc._RES40_PC_CACHE.clear()


@pytest.mark.parametrize("raw", ["", "0", "off", "none"])
def test_off_by_default(monkeypatch, raw):
    monkeypatch.setenv(tpc.GLUE_ENV, raw)
    assert tpc.glue_cuts() == frozenset()
    assert not any(tpc.glue_cut(c) for c in tpc.GLUE_CUTS)


def test_all_and_lists(monkeypatch):
    monkeypatch.setenv(tpc.GLUE_ENV, "all")
    assert tpc.glue_cuts() == frozenset(tpc.GLUE_CUTS)
    monkeypatch.setenv(tpc.GLUE_ENV, "kvwo, guin")
    assert tpc.glue_cuts() == {"kvwo", "guin"}
    monkeypatch.setenv(tpc.GLUE_ENV, "kvwo")
    assert tpc.glue_cut("kvwo") and not tpc.glue_cut("guin")


def test_unknown_cut_raises(monkeypatch):
    monkeypatch.setenv(tpc.GLUE_ENV, "kvwo,bogus")
    with pytest.raises(ValueError, match="bogus"):
        tpc.glue_cuts()
    monkeypatch.delenv(tpc.GLUE_ENV)
    with pytest.raises(ValueError):
        tpc.glue_cut("bogus")


# ---- guin --------------------------------------------------------------------------------------------------------
class _MC:
    def __init__(self, width):
        self.shard_spec = None if width is None else types.SimpleNamespace(shape=[32, width])


class _X:
    def __init__(self, width):
        self._mc = _MC(width)

    def memory_config(self):
        return self._mc


def _gu_args(ibw):
    return types.SimpleNamespace(
        mlp_gu_dsmc_progcfg_bfp4=types.SimpleNamespace(in0_block_w=ibw), mlp_gu_dsmc_progcfg=None
    )


@pytest.mark.parametrize("ibw,ok", [(1, True), (2, True), (4, True), (3, False), (8, False)])
def test_guin_shard_guard(monkeypatch, ibw, ok):
    monkeypatch.setattr(ttnn, "bfloat4_b", "bf4", raising=False)
    w = types.SimpleNamespace(dtype="bf4")
    # RES40: 5120 / 40 = 128 columns = 4 K tiles per in0 core
    assert mlp_mod._dsmc_in0_shard_ok(_X(128), w, _gu_args(ibw)) is ok


def test_guin_guard_needs_a_shard_and_a_config(monkeypatch):
    monkeypatch.setattr(ttnn, "bfloat4_b", "bf4", raising=False)
    w = types.SimpleNamespace(dtype="bf4")
    assert not mlp_mod._dsmc_in0_shard_ok(_X(None), w, _gu_args(2))
    assert not mlp_mod._dsmc_in0_shard_ok(
        _X(128), w, types.SimpleNamespace(mlp_gu_dsmc_progcfg_bfp4=None, mlp_gu_dsmc_progcfg=None)
    )


# ---- kvwo --------------------------------------------------------------------------------------------------------
class _T:
    def __init__(self, shape, sharded=False):
        self.shape = shape
        self._sharded = sharded

    def is_sharded(self):
        return self._sharded


def _attn(ibw=8):
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention

    a = TPAttention.__new__(TPAttention)
    a.args = types.SimpleNamespace(dim=5120)
    a._wo_pair_pc = types.SimpleNamespace(in0_block_w=ibw)
    return a


def _fake_res40_pc(monkeypatch):
    # the real builder needs ttnn's program-config class; its K blocking is covered by test_verify_out_res40_host
    monkeypatch.setattr(
        tpc, "res40_out_progcfg", lambda k, n: types.SimpleNamespace(in0_block_w=8) if n == 5120 else None
    )


def test_kvwo_guard(monkeypatch):
    _fake_res40_pc(monkeypatch)
    a = _attn()
    gated = _T([1, 1, 32, 1536])
    monkeypatch.delenv(tpc.VERIFY_OUT_RES40_ENV, raising=False)
    assert not a._kvwo_res40_ok(gated)  # RES40 all-reduce input layout is off
    monkeypatch.setenv(tpc.VERIFY_OUT_RES40_ENV, "1")
    assert a._kvwo_res40_ok(gated)  # K = 48 tiles: in0_block_w 8 in both configs
    assert not _attn(ibw=4)._kvwo_res40_ok(gated)  # different K blocking -> keep today's matmul
    assert not a._kvwo_res40_ok(_T([1, 1, 64, 1536]))
    assert not a._kvwo_res40_ok(_T([1, 1, 32, 1536], sharded=True))


def _run_kv_split_out(monkeypatch, glue):
    from models.demos.blackhole.qwen36.tt.attention import tp as attn_tp

    if glue:
        monkeypatch.setenv(tpc.GLUE_ENV, glue)
    monkeypatch.setenv(tpc.VERIFY_OUT_RES40_ENV, "1")
    _fake_res40_pc(monkeypatch)
    calls = []
    gated = _T([1, 1, 32, 1536])
    monkeypatch.setattr(ttnn, "transpose", lambda *a, **k: "heads", raising=False)
    monkeypatch.setattr(ttnn.experimental, "nlp_concat_heads", lambda *a, **k: "flat", raising=False)
    monkeypatch.setattr(ttnn, "matmul", lambda *a, **k: "full", raising=False)
    monkeypatch.setattr(ttnn, "multiply", lambda *a, **k: gated, raising=False)
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ttnn, "reshape", lambda t, s, *a, **k: ("reshaped", t), raising=False)

    def _res40(x, w, cfg, dar, site="out"):
        calls.append(("res40", site))
        return "res40_partial"

    def _mm1d(x, w, pc, cfg, out_memory_config=None):
        calls.append(("mm1d", out_memory_config))
        return _T([1, 1, 32, 5120])

    monkeypatch.setattr(tpc, "res40_out_matmul", _res40)
    monkeypatch.setattr(tpc, "matmul_1d_decode", _mm1d)
    dar = types.SimpleNamespace(all_reduce=lambda p: ("ar", p))
    a = _attn()
    a.tw = {"wo_pair": "wo_pair"}
    a.wo_cfg = "wo_cfg"
    out = attn_tp.TPAttention._kv_split_out(a, "attn", "gate", 32, {"scatter": "sc"}, dar)
    return calls, out


def test_kv_split_out_flag_off_is_todays_path(monkeypatch):
    calls, out = _run_kv_split_out(monkeypatch, None)
    assert [c[0] for c in calls] == ["mm1d"]
    assert out[0] == "ar" and out[1][0] == "reshaped"


def test_kv_split_out_kvwo_takes_res40(monkeypatch):
    calls, out = _run_kv_split_out(monkeypatch, "kvwo")
    assert calls == [("res40", "attn_wo_pair")]
    assert out == ("ar", "res40_partial")


# ---- kvt ---------------------------------------------------------------------------------------------------------
def test_nkv_one_guard():
    from models.demos.blackhole.qwen36.tt.attention import tp as attn_tp

    assert attn_tp.NKV_ONE(types.SimpleNamespace(NKV=1))
    assert not attn_tp.NKV_ONE(types.SimpleNamespace(NKV=2))
    assert not attn_tp.NKV_ONE(types.SimpleNamespace())


def _stub_write(monkeypatch):
    log = []
    monkeypatch.setattr(ttnn, "Shape", lambda s: tuple(s), raising=False)
    monkeypatch.setattr(ttnn, "reshape", lambda t, *a, **k: ("view", t), raising=False)
    monkeypatch.setattr(
        ttnn, "to_memory_config", lambda t, cfg: log.append(("to_mem", cfg)) or ("sh", t), raising=False
    )
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("free", t)), raising=False)
    monkeypatch.setattr(
        ttnn.experimental,
        "paged_fused_update_cache",
        lambda kc, k, vc, v, **kw: log.append(("update", k, v, kw.get("rows_per_user"))),
        raising=False,
    )
    monkeypatch.setattr(tpc, "aliases", lambda a, b: True)
    return log


def _attn_kv():
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention

    a = TPAttention.__new__(TPAttention)
    a.HD, a.NKV = 256, 1
    a.paged_k, a.paged_v = "PK", "PV"
    a._kv_rows_shard_cfg = lambda first, n, rows: ("rows", first, n, rows)
    return a


def test_multirow_default_reshards(monkeypatch):
    log = _stub_write(monkeypatch)
    _attn_kv()._write_kv_fused_multirow("k", "v", types.SimpleNamespace(shape=[32]), 8, 4, "upt")
    assert [e[0] for e in log if e[0] == "to_mem"] == ["to_mem", "to_mem"]
    assert ("update", ("sh", ("view", "k")), ("sh", ("view", "v")), 4) in log


def test_multirow_presharded_skips_reshards(monkeypatch):
    log = _stub_write(monkeypatch)
    _attn_kv()._write_kv_fused_multirow("k", "v", types.SimpleNamespace(shape=[32]), 8, 4, "upt", presharded=True)
    assert not [e for e in log if e[0] == "to_mem"]
    assert ("update", ("view", "k"), ("view", "v"), 4) in log
    # the views are freed exactly once, after the update (sources alias them)
    frees = [e[1] for e in log if e[0] == "free"]
    assert frees.count(("view", "k")) == 1 and frees.count(("view", "v")) == 1


def test_kv_split_heads_transposes_into_given_layouts(monkeypatch):
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention

    a = TPAttention.__new__(TPAttention)
    a.NH, a.NKV, a.HD, a.rope_dim = 3, 1, 256, 64
    a.tw = {"qk_norm_dense_pair": "w"}
    seen = []
    monkeypatch.setattr(ttnn.experimental, "nlp_create_qkv_heads", lambda *a, **k: ("q", "k", "v"), raising=False)
    monkeypatch.setattr(ttnn.experimental, "rotary_embedding_hf", lambda *a, **k: "roped", raising=False)
    for fn in ("concat", "rms_norm", "multiply", "slice", "reshape", "deallocate"):
        monkeypatch.setattr(ttnn, fn, lambda *a, **k: "t", raising=False)
    monkeypatch.setattr(tpc, "aliases", lambda a, b: True)
    monkeypatch.setattr(
        ttnn, "transpose", lambda t, d0, d1, memory_config=None: seen.append(memory_config) or t, raising=False
    )
    TPAttention._kv_split_heads(a, "qkv3", "cos", "sin", 32, k_mem="KMEM", v_mem="VMEM")
    assert seen == ["KMEM", "VMEM"]
    seen.clear()
    TPAttention._kv_split_heads(a, "qkv3", "cos", "sin", 32)
    assert seen == [ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]
