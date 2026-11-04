# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_MTP_KV_ONLY=1: an MTP KV-maintenance forward (need_logits=False) stops after the drafter layer's
paged KV write.

No device. The ttnn ops on the attention path are replaced by a symbolic tracer: every op returns a ``_Sym`` whose name
is the expression that produced it. The K and V that reach the paged cache write are then compared, as expressions,
between the full step and the KV-only step. Equal expressions mean the same ops on the same inputs, so the device writes
the same bytes; test_mtp_kv_only.py checks that on device.

Run: pytest models/demos/blackhole/qwen36/tests/test_mtp_kv_only_host.py -q
"""
import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common
from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP


class _Sym:
    """A fake tensor named by the expression that produced it."""

    def __init__(self, name, shape=(1, 1, 8, 64)):
        self.name, self.shape = name, tuple(shape)

    def __repr__(self):
        return self.name


def _shape_of(args):
    return next((a.shape for a in args if isinstance(a, _Sym)), (1, 1, 8, 64))


def _op(log, name):
    def f(*args, **kw):
        kws = ", ".join(f"{k}={v!r}" for k, v in sorted(kw.items()) if k not in ("memory_config", "program_config"))
        expr = f"{name}({', '.join(repr(a) for a in args)}{', ' if args and kws else ''}{kws})"
        log.append(name)
        return _Sym(expr, _shape_of(args))

    return f


class _Grid:
    x, y = 11, 10


class _Mesh:
    def compute_with_storage_grid_size(self):
        return _Grid()


def _patch_ttnn(monkeypatch, log, writes):
    """Symbolic stand-ins for every ttnn op the decode attention runs. Cache writes are recorded in ``writes``."""
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    for name in (
        "rms_norm",
        "multiply",
        "pad",
        "to_memory_config",
        "sharded_to_interleaved",
        "sigmoid",
        "reshape",
        "slice",
    ):
        monkeypatch.setattr(ttnn, name, _op(log, name))
    monkeypatch.setattr(ttnn, "SDPAProgramConfig", lambda **k: "sdpa_cfg")
    monkeypatch.setattr(ttnn, "UnaryWithParam", lambda *a, **k: "act", raising=False)

    def nlp_heads(qkv, **k):
        log.append("nlp_create_qkv_heads_decode")
        return _Sym(f"q[{qkv!r}]"), _Sym(f"k[{qkv!r}]"), _Sym(f"v[{qkv!r}]")

    def puc(cache, x, **k):
        log.append("paged_update_cache")
        writes.append(("puc", cache, repr(x), repr(k.get("update_idxs_tensor")), repr(k.get("page_table"))))

    def pfuc(kc, k_in, vc, v_in, **k):
        log.append("paged_fused_update_cache")
        writes.append(("pfuc", kc, repr(k_in), vc, repr(v_in), repr(k.get("update_idxs_tensor"))))

    monkeypatch.setattr(ttnn.experimental, "nlp_create_qkv_heads_decode", nlp_heads)
    monkeypatch.setattr(ttnn.experimental, "paged_update_cache", puc)
    monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", pfuc)
    monkeypatch.setattr(ttnn.transformer, "paged_scaled_dot_product_attention_decode", _op(log, "sdpa"))
    monkeypatch.setattr(tp_mod, "apply_partial_rope_decode", _op(log, "rope"))
    monkeypatch.setattr(tp_mod, "apply_partial_rope_decode_single", _op(log, "rope1"))
    monkeypatch.setattr(tp_mod, "tt_all_reduce", _op(log, "all_reduce"))


def _attention(B):
    a = TPAttention.__new__(TPAttention)
    a.mesh, a.args, a.tt_ccl, a.B = _Mesh(), types.SimpleNamespace(ccl_topology=lambda: "ring"), None, B
    a.tw = {"q_norm": "qn", "k_norm": "kn", "wo": "wo"}
    a.NH, a.NKV, a.HD, a.rope_dim = 3, 1, 64, 16
    a.scale = 0.125
    a.use_paged, a.paged_k, a.paged_v = True, "K$", "V$"
    a._use_nlp_decode_heads, a._fused_qkv, a.lean_decode = True, True, True
    a.decode_sdpa_max_cores = 64
    a._kv_key_cfg_cache, a._kv_shard_cfg_cache = {}, {}
    a._qkv = lambda x, decode_qkv3_l1=False: (_Sym(f"qkv3({x!r})"), _Sym(f"gate({x!r})"), None)
    a._make_heads_decode = lambda qg, kp, vp, b: (_Sym(f"q[{qg!r}]"), kp, _Sym(f"k[{qg!r}]"), _Sym(f"v[{qg!r}]"))
    a._kv_shard_cfg = lambda b: f"kvcfg{b}"
    a._kv_key_shard_cfg = lambda b: f"kkeycfg{b}"
    a._concat_heads_decode = _op([], "concat_heads")
    a._wo_proj = _op([], "wo")
    return a


def _run(monkeypatch, B, kv_only, **kw):
    log, writes = [], []
    _patch_ttnn(monkeypatch, log, writes)
    a = _attention(B)
    if kw.get("alias_kv_write"):
        a._write_kv_aliased = lambda keys, values, k_p, v_p, pos, pt, b, hd, **k: writes.append(
            ("aliased", keys, values, repr(k_p), repr(v_p), repr(pos), repr(pt), b, k.get("spec_n_users"))
        )
    x = _Sym("x", (1, 1, B, 640))
    pos, pt = _Sym("pos", (B,)), _Sym("pt", (B, 4))
    out = a.forward_decode(x, pos, _Sym("cos"), _Sym("sin"), page_table=pt, kv_only=kv_only, **kw)
    return out, log, writes


@pytest.mark.parametrize(
    "B, kw, lean",
    [
        (32, dict(alias_kv_write=True, spec_n_users=8), False),  # batched reseed: legacy path, grouped aliased write
        (4, dict(alias_kv_write=True, spec_n_users=1), False),  # B == 1 engine reseed shape: per-row aliased write
        (1, {}, True),  # eager 1-row reseed / warm: lean path, fused K+V update
        (8, {}, True),  # join warm (independent rows): lean path, B > 1 rope
    ],
)
def test_kv_only_writes_the_same_kv_and_skips_the_rest(monkeypatch, B, kw, lean):
    monkeypatch.setenv("QWEN36_ATTN_DECODE_LEAN", "1")
    full_out, full_log, full_writes = _run(monkeypatch, B, False, **kw)
    kv_out, kv_log, kv_writes = _run(monkeypatch, B, True, **kw)
    assert full_out is not None and kv_out is None
    assert full_writes and kv_writes == full_writes, f"KV write differs:\n{full_writes}\n{kv_writes}"
    assert ("paged_fused_update_cache" in full_log) == lean
    for skipped in ("sdpa", "all_reduce"):
        assert skipped in full_log and skipped not in kv_log, skipped
    # Q is never normed or roped in KV-only mode: one rms_norm (k) and one rope (k) only.
    assert kv_log.count("rms_norm") == 1 and kv_log.count("rope") + kv_log.count("rope1") == 1
    assert full_log.count("rms_norm") == 2 and full_log.count("rope") + full_log.count("rope1") == 2


def test_kv_only_refuses_the_spec_verify(monkeypatch):
    with pytest.raises(AssertionError):
        _run(monkeypatch, 32, True, alias_kv_write=True, spec_verify_mode=True, spec_n_users=8)


class _Buf(_Sym):
    """A fake device tensor with a buffer: a view shares its source's buffer, as ttnn.pad's in-place tile fill does."""

    def __init__(self, name, shape=(1, 1, 8, 64), buf=None):
        super().__init__(name, shape)
        self.buf = buf if buf is not None else [name]

    def buffer_address(self):
        return id(self.buf)


@pytest.mark.parametrize("pad_is_view", [True, False], ids=["pad_view", "pad_copy"])
@pytest.mark.parametrize("kv_only", [False, True], ids=["full", "kv_only"])
@pytest.mark.parametrize(
    "B, kw",
    [
        (32, dict(alias_kv_write=True, spec_n_users=8)),  # batched reseed: grouped aliased write
        (4, dict(alias_kv_write=True, spec_n_users=1)),  # per-row aliased write
        (8, {}),  # legacy non-aliased write (QWEN36_ATTN_DECODE_LEAN=0)
    ],
)
def test_legacy_kv_write_never_reads_a_freed_pad_view(monkeypatch, B, kw, kv_only, pad_is_view):
    """The legacy paged branch pads k / v ([1, B, NKV, HD]) to 32 rows. For NKV < 32 the rows already sit in the tiles,
    so ttnn.pad fills the tile padding in place and returns a VIEW of the same buffer. ttnn.deallocate forces, so
    freeing k / v right after the pad freed the buffer under k_p / v_p, and the KV write's own slices and shards could
    land on it before it was read (device: V rows differing between full and kv_only, test_mtp_kv_only.py). The
    source must stay allocated until the KV write has read k_p / v_p; the branch frees it only when pad made a copy."""
    monkeypatch.setenv("QWEN36_ATTN_DECODE_LEAN", "0")
    log, writes = [], []
    _patch_ttnn(monkeypatch, log, writes)
    freed = []
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: freed.append(getattr(t, "buf", None)))

    def pad(t, *a, **k):
        log.append("pad")
        return _Buf(f"pad({t!r})", t.shape, t.buf if pad_is_view else None)

    monkeypatch.setattr(ttnn, "pad", pad)
    # rope and the head split hand back buffer-backed tensors, so the padded k / v have a source buffer to share.
    monkeypatch.setattr(tp_mod, "apply_partial_rope_decode", lambda k, *a, **kw: _Buf(f"rope({k!r})", k.shape))
    reads = []  # (what, buffer still allocated?) when the KV write consumes k_p / v_p

    def consume(what, t):
        reads.append((what, all(t.buf is not f for f in freed)))

    def to_mc(t, cfg, *a, **k):
        if isinstance(t, _Buf) and repr(t).startswith("pad("):
            consume("to_memory_config", t)
        return _Sym(f"to_memory_config({t!r})", t.shape)

    monkeypatch.setattr(ttnn, "to_memory_config", to_mc)
    a = _attention(B)
    a._make_heads_decode = lambda qg, kp, vp, b: (
        _Buf(f"q[{qg!r}]"),
        kp,
        _Buf(f"k[{qg!r}]"),
        _Buf(f"v[{qg!r}]", (1, b, 1, 64)),
    )
    if kw.get("alias_kv_write"):

        def aliased(keys, values, k_p, v_p, *args, **k):
            consume("_write_kv_aliased k", k_p)
            consume("_write_kv_aliased v", v_p)

        a._write_kv_aliased = aliased
    x, pos, pt = _Sym("x", (1, 1, B, 640)), _Sym("pos", (B,)), _Sym("pt", (B, 4))
    a.forward_decode(x, pos, _Sym("cos"), _Sym("sin"), page_table=pt, kv_only=kv_only, **kw)
    assert len(reads) == 2, reads
    assert all(alive for _, alive in reads), f"the KV write read a freed pad view: {reads}"
    if not pad_is_view:  # pad made copies: the branch frees the sources (roped k, head-split v) itself: no leak
        freed_names = [b[0] for b in freed if isinstance(b, list)]
        assert any(n.startswith("rope(") and "k[" in n for n in freed_names), f"k never freed: {freed_names}"
        assert any(n.startswith("v[") for n in freed_names), f"v never freed: {freed_names}"


# --------------------------------------------------------------------------- #
# Decoder layer: kv_only returns after the attention, never touching ffn_norm / MLP
# --------------------------------------------------------------------------- #
class _FakeAttn:
    def __init__(self):
        self.kw = None

    def forward_decode(self, x, pos, cos, sin, **kw):
        self.kw = kw
        return None if kw.get("kv_only") else _Sym("attn_out")


def _layer(full_attention=True, nd=8):
    lay = Qwen36DecoderLayer.__new__(Qwen36DecoderLayer)
    lay.num_devices, lay.is_full_attention, lay._decode_ar = nd, full_attention, None
    lay.args = types.SimpleNamespace(dim=5120, get_norm_config=lambda key, mode: {})
    lay.attention_norm = lambda x, mode=None, norm_config=None: _Sym("attn_in")
    lay.attention = _FakeAttn()
    lay.ffn_norm = lambda *a, **k: (_ for _ in ()).throw(AssertionError("ffn_norm ran"))
    lay.feed_forward = types.SimpleNamespace(forward=lambda *a, **k: (_ for _ in ()).throw(AssertionError("mlp ran")))
    return lay


def test_layer_kv_only_stops_after_attention(monkeypatch):
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    lay = _layer()
    x = _Sym("x", (1, 1, 32, 640))
    assert (
        lay.forward(x, mode="decode", position_tensor="p", page_table="pt", alias_kv_write=True, kv_only=True) is None
    )
    assert lay.attention.kw["kv_only"] is True


def test_layer_default_call_is_unchanged(monkeypatch):
    """kv_only off: the attention call carries no new keyword (the base model's call is the same call as before)."""
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "add", _op([], "add"))
    lay = _layer()
    lay.ffn_norm = lambda *a, **k: _Sym("ff_in")
    lay.feed_forward = types.SimpleNamespace(forward=lambda x, mode=None, **k: _Sym("ff_out"))
    out = lay.forward(_Sym("x", (1, 1, 32, 640)), mode="decode", position_tensor="p", page_table="pt")
    assert "kv_only" not in lay.attention.kw and out is not None


def test_layer_kv_only_never_leaves_a_residual_armed(monkeypatch):
    """QWEN36_MTP_KV_ONLY x QWEN36_DECODE_AR_FUSE_RESIDUAL: the kv_only return skips the residual add, so the layer
    must not arm the decode all-reduce's residual (and disarms on the way out)."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    dar = tpc.DecodeResidualAllReduce.__new__(tpc.DecodeResidualAllReduce)
    dar.fuse_residual, dar._fused_out = True, None
    dar._armed = _Sym("stale")  # left over from an earlier forward: the kv_only exit must clear it
    armed = []
    dar.rms_norm = lambda x, norm: _Sym("attn_in")
    dar.arm_residual = lambda x: armed.append(x)
    lay = _layer()
    lay._decode_ar = dar
    lay.attention_norm = types.SimpleNamespace(norm=None)
    x = _Sym("x", (1, 1, 32, 5120))  # last dim == args.dim: the replicated-residual (_dar) path
    assert lay.forward(x, mode="decode", position_tensor="p", page_table="pt", kv_only=True) is None
    assert armed == [] and dar._armed is None and dar._fused_out is None
    assert lay.attention.kw["decode_ar"] is dar and lay.attention.kw["kv_only"] is True


@pytest.mark.parametrize("full_attention, nd, mode", [(False, 8, "decode"), (True, 1, "decode"), (True, 8, "prefill")])
def test_layer_kv_only_refuses_unsupported(monkeypatch, full_attention, nd, mode):
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    with pytest.raises(AssertionError):
        _layer(full_attention, nd).forward(_Sym("x"), mode=mode, kv_only=True)


# --------------------------------------------------------------------------- #
# MTP head: only need_logits=False with the flag resolves to a KV-only forward
# --------------------------------------------------------------------------- #
class _Stop(Exception):
    pass


def _mtp(kv_only_maint):
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices = 8
    m.kv_only_maint = kv_only_maint
    m.embd = lambda ids: _Sym("emb", (8, 1, 640))
    m._fuse = lambda e, h, mode: _Sym("fused")
    m.calls = []

    def dec(x, **kw):
        m.calls.append(kw)
        return None if kw.get("kv_only") else _Sym("block_out")

    m.decoder = types.SimpleNamespace(forward=dec)

    def head_norm(*a, **k):
        raise _Stop

    m.head_norm = head_norm
    m.args = types.SimpleNamespace(get_norm_config=lambda key, mode: {})
    return m


def test_mtp_forward_decode_kv_only(monkeypatch):
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "reshape", _op([], "reshape"))
    args = (_Sym("h"), _Sym("tok"), _Sym("pos"), _Sym("cos"), _Sym("sin"), _Sym("pt"))

    on = _mtp(True)
    assert on.forward_decode(*args, need_logits=False, alias_kv_write=True, spec_n_users=8) == (None, None)
    assert on.calls[-1]["kv_only"] is True
    with pytest.raises(_Stop):  # a draft step (need_logits=True) is never KV-only: it reaches the head norm
        on.forward_decode(*args)
    assert "kv_only" not in on.calls[-1]

    off = _mtp(False)
    logits, hn = off.forward_decode(*args, need_logits=False)
    assert logits is None and repr(hn) == "block_out" and "kv_only" not in off.calls[-1]

    bare = Qwen36MTP.__new__(Qwen36MTP)  # a head built without __init__ (other host tests) behaves as flag off
    bare.__dict__.update({k: v for k, v in off.__dict__.items() if k != "kv_only_maint"})
    assert repr(bare.forward_decode(*args, need_logits=False)[1]) == "block_out"


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("QWEN36_MTP_KV_ONLY", raising=False)
    assert tp_common.mtp_kv_only_enabled() is False
    monkeypatch.setenv("QWEN36_MTP_KV_ONLY", "1")
    assert tp_common.mtp_kv_only_enabled() is True
    monkeypatch.setenv("QWEN36_MTP_KV_ONLY", "0")
    assert tp_common.mtp_kv_only_enabled() is False


def test_free_maint_skips_none(monkeypatch):
    from models.demos.blackhole.qwen36.tt.spec_engine import _free_maint

    freed = []
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    _free_maint("a", None, "b")
    assert freed == ["a", "b"]
