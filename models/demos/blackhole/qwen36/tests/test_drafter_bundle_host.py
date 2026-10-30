# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the r6 drafter bundle flags (tp_common, "r6 drafter bundle"): flag parsing, and that each flag
off leaves the drafter's calls exactly as they were while each flag on changes only the intended call.

  QWEN36_MTP_FC_PROGCFG (+ _IBW, _CORES)  explicit 1D decode progcfg for the MTP fc, L1 concat input
  QWEN36_MTP_SDPA_WINDOW                  sliding window on the drafter's decode SDPA
  QWEN36_MTP_LAYER_BFP4                   bfloat4_b + LoFi drafter layer weights
  QWEN36_MTP_EMBED_ROW                    one-tile-row drafter token embedding

Needs ttnn importable (no device: every ttnn op used here is monkeypatched).

  pytest -q models/demos/blackhole/qwen36/tests/test_drafter_bundle_host.py
"""
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import layer as layer_mod
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP
from models.tt_transformers.tt.common import Mode

_FLAGS = (
    "QWEN36_MTP_FC_PROGCFG",
    "QWEN36_MTP_FC_IBW",
    "QWEN36_MTP_FC_CORES",
    "QWEN36_MTP_SDPA_WINDOW",
    "QWEN36_MTP_LAYER_BFP4",
    "QWEN36_MTP_EMBED_ROW",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for f in _FLAGS:
        monkeypatch.delenv(f, raising=False)
    yield


# ------------------------------------------------------------------------------------------------------------------ #
# flag parsing
# ------------------------------------------------------------------------------------------------------------------ #
def test_all_flags_default_off():
    assert tpc.mtp_fc_decode_progcfg(10240, 640, 11) is None
    assert tpc.mtp_sdpa_window() == 0
    assert tpc.mtp_layer_bfp4_parts() == frozenset()
    assert tpc.mtp_embed_row_enabled() is False


def test_fc_progcfg_defaults_and_overrides(monkeypatch):
    seen = []
    monkeypatch.setattr(tpc, "create_matmul_1d_decode_progcfg", lambda *a, **k: seen.append((a, k)) or "PC")
    monkeypatch.setenv("QWEN36_MTP_FC_PROGCFG", "1")
    assert tpc.mtp_fc_decode_progcfg(10240, 640, 11) == "PC"
    assert seen[-1] == ((32, 10240, 640), {"num_cores": 20, "grid_w": 11, "max_in0_block_w": 32})
    monkeypatch.setenv("QWEN36_MTP_FC_IBW", "16")
    monkeypatch.setenv("QWEN36_MTP_FC_CORES", "10")
    tpc.mtp_fc_decode_progcfg(10240, 640, 11)
    assert seen[-1] == ((32, 10240, 640), {"num_cores": 10, "grid_w": 11, "max_in0_block_w": 16})
    monkeypatch.setenv("QWEN36_MTP_FC_IBW", "0")
    with pytest.raises(ValueError):
        tpc.mtp_fc_decode_progcfg(10240, 640, 11)


def test_fc_progcfg_real_config_shape(monkeypatch):
    """The default config on the BH grid: 20 cores (11x2 grid), one output tile column each, 10 K blocks of 32."""
    monkeypatch.setenv("QWEN36_MTP_FC_PROGCFG", "1")
    pc = tpc.mtp_fc_decode_progcfg(10240, 640, 11)
    assert pc.in0_block_w == 32 and pc.per_core_M == 1 and pc.per_core_N == 1
    grid = pc.compute_with_storage_grid_size
    assert (grid.x, grid.y) == (11, 2)
    assert pc.mcast_in0


@pytest.mark.parametrize("raw,want", [(None, 0), ("", 0), ("0", 0), ("-5", 0), ("4096", 4096), (" 1024 ", 1024)])
def test_window_parse(monkeypatch, raw, want):
    if raw is not None:
        monkeypatch.setenv("QWEN36_MTP_SDPA_WINDOW", raw)
    assert tpc.mtp_sdpa_window() == want


def test_window_bad_value(monkeypatch):
    monkeypatch.setenv("QWEN36_MTP_SDPA_WINDOW", "4k")
    with pytest.raises(ValueError):
        tpc.mtp_sdpa_window()


@pytest.mark.parametrize(
    "raw,want",
    [
        ("0", set()),
        ("off", set()),
        ("1", {"attn", "mlp"}),
        ("all", {"attn", "mlp"}),
        ("mlp", {"mlp"}),
        ("attn", {"attn"}),
        ("attn,mlp", {"attn", "mlp"}),
        (" MLP ", {"mlp"}),
    ],
)
def test_layer_bfp4_parts(monkeypatch, raw, want):
    monkeypatch.setenv("QWEN36_MTP_LAYER_BFP4", raw)
    assert tpc.mtp_layer_bfp4_parts() == frozenset(want)


@pytest.mark.parametrize("raw", ["head", "attn,lm", ","])
def test_layer_bfp4_bad(monkeypatch, raw):
    monkeypatch.setenv("QWEN36_MTP_LAYER_BFP4", raw)
    with pytest.raises(ValueError):
        tpc.mtp_layer_bfp4_parts()


# ------------------------------------------------------------------------------------------------------------------ #
# fake device tensor
# ------------------------------------------------------------------------------------------------------------------ #
class _T:
    """Fake device tensor: shape, dtype, a buffer address, sharded flag, deallocate log."""

    _next = [1000]

    def __init__(self, shape, dtype=ttnn.bfloat16, sharded=False):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._sharded = sharded
        self._addr = _T._next[0]
        _T._next[0] += 64
        self.freed = False

    def is_sharded(self):
        return self._sharded

    def buffer_address(self):
        return self._addr

    def deallocate(self, *_):
        self.freed = True


# ------------------------------------------------------------------------------------------------------------------ #
# SDPA window kwarg
# ------------------------------------------------------------------------------------------------------------------ #
def test_window_kwarg():
    a = TPAttention.__new__(TPAttention)  # bare (other host tests build it this way): no window
    assert a._decode_window_kw() == {}
    a.decode_sdpa_window = None
    assert a._decode_window_kw() == {}
    a.decode_sdpa_window = 4096
    assert a._decode_window_kw() == {"sliding_window_size": 4096}


# ------------------------------------------------------------------------------------------------------------------ #
# MTP fc (_fuse) and embedding (forward_decode)
# ------------------------------------------------------------------------------------------------------------------ #
class _Norm:
    def __init__(self, tag):
        self.tag = tag

    def __call__(self, x, mode=None, norm_config=None):
        return _T(x.shape[:-1] + (5120,))


def _fuse_head(pc=None, nd=8):
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices = nd
    m.args = types.SimpleNamespace(get_norm_config=lambda key, mode: {"output_mem_config": None})
    m.pre_fc_norm_embedding = _Norm("e")
    m.pre_fc_norm_hidden = _Norm("h")
    m.fc = "FC"
    m._fc_compute_cfg = "HIFI2"
    if pc is not None:
        m._fc_decode_pc = pc
    return m


def _record(monkeypatch):
    log = []
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    monkeypatch.setattr(
        ttnn, "concat", lambda ts, dim, **kw: log.append(("concat", kw)) or _T(ts[0].shape[:-1] + (10240,))
    )
    monkeypatch.setattr(ttnn, "linear", lambda a, b, **kw: log.append(("linear", kw)) or _T(a.shape[:-1] + (640,)))
    return log


@pytest.mark.parametrize("rows,mode", [(8, Mode.DECODE), (32, Mode.DECODE), (2048, Mode.PREFILL)])
@pytest.mark.parametrize("bare", [True, False])
def test_fuse_flag_off_is_todays_call(monkeypatch, rows, mode, bare):
    """Flag off: a head without the attribute (built via __new__) and one built with it None make today's two calls."""
    log = _record(monkeypatch)
    m = _fuse_head()
    if not bare:
        m._fc_decode_pc = None
    m._fuse(_T((1, 1, rows, 640)), _T((1, 1, rows, 640)), mode)
    assert log == [
        ("concat", {}),
        ("linear", {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "compute_kernel_config": "HIFI2"}),
    ]


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_fuse_progcfg_decode_rows(monkeypatch, rows):
    log = _record(monkeypatch)
    _fuse_head(pc="PC")._fuse(_T((1, 1, rows, 640)), _T((1, 1, rows, 640)), Mode.DECODE)
    assert log == [
        ("concat", {"memory_config": ttnn.L1_MEMORY_CONFIG}),
        (
            "linear",
            {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "compute_kernel_config": "HIFI2", "program_config": "PC"},
        ),
    ]


@pytest.mark.parametrize("rows,mode", [(64, Mode.DECODE), (2048, Mode.PREFILL), (8, Mode.PREFILL)])
def test_fuse_progcfg_not_for_prefill_or_wide(monkeypatch, rows, mode):
    log = _record(monkeypatch)
    _fuse_head(pc="PC")._fuse(_T((1, 1, rows, 640)), _T((1, 1, rows, 640)), mode)
    assert log == [
        ("concat", {}),
        ("linear", {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "compute_kernel_config": "HIFI2"}),
    ]


@pytest.mark.parametrize("ids_shape", [(8, 1), (1, 8), (32, 1), (1, 1), (64, 1)])
def test_embed_rows_ops(monkeypatch, ids_shape):
    """QWEN36_MTP_EMBED_ROW: a row-major lookup, a [1,1,rows,dim/tp] view, zero rows up to whole tiles (row-major pad),
    one tilize, and a view back to the logical rows; the intermediates are freed, the result is not."""
    log, freed = [], []
    rows = ids_shape[0] * ids_shape[1]
    pad = -rows % 32
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a: freed.append(t))
    monkeypatch.setattr(
        ttnn,
        "embedding",
        lambda ids, w, layout=None, memory_config=None: log.append(("embedding", ids.shape, w, layout, memory_config))
        or _T((ids_shape[0], ids_shape[1], 640)),
    )

    def reshape(t, shape, padded=None):
        log.append(("reshape", tuple(shape), None if padded is None else tuple(padded)))
        return _T(tuple(shape))

    monkeypatch.setattr(ttnn, "reshape", reshape)
    monkeypatch.setattr(
        ttnn, "pad", lambda t, p, value=None: log.append(("pad", tuple(map(tuple, p)), value)) or _T((1, 1, rows + pad, 640))
    )
    monkeypatch.setattr(
        ttnn,
        "to_layout",
        lambda t, layout, memory_config=None: log.append(("to_layout", t.shape, layout, memory_config))
        or _T(t.shape),
    )
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.embd = types.SimpleNamespace(weights="W")
    out = m._embed_rows(_T(ids_shape, ttnn.uint32))
    assert out.shape == (1, 1, rows, 640)
    want = [
        ("embedding", ids_shape, "W", ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ("reshape", (1, 1, rows, 640), None),
    ]
    if pad:
        want += [
            ("pad", ((0, 0), (0, 0), (0, pad), (0, 0)), 0.0),
            ("to_layout", (1, 1, rows + pad, 640), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
            ("reshape", (1, 1, rows, 640), (1, 1, rows + pad, 640)),
        ]
    else:
        want += [("to_layout", (1, 1, rows, 640), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)]
    assert log == want
    assert out not in freed
    assert len(freed) == (4 if pad else 2)  # rm + view (+ the padded row-major block + the tilized buffer view source)


def _decode_head(monkeypatch, embed_row):
    """A bare MTP head whose forward_decode runs to the decoder with every device op faked; returns (head, log)."""
    log = []
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices = 8
    if embed_row is not None:
        m._embed_row = embed_row
    m.embd = lambda ids: log.append(("embd", ids.shape)) or _T((ids.shape[0], ids.shape[1], 640))
    m._embed_rows = lambda ids: log.append(("embed_rows", ids.shape)) or _T((1, 1, ids.shape[0] * ids.shape[1], 640))
    m._fuse = lambda e, h, mode: log.append(("fuse", e.shape)) or _T(e.shape)
    m.kv_only_maint = True
    m.decoder = types.SimpleNamespace(forward=lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: log.append(("reshape", tuple(shape))) or _T(tuple(shape)))
    return m, log


@pytest.mark.parametrize("embed_row", [None, False])
def test_forward_decode_embedding_flag_off(monkeypatch, embed_row):
    m, log = _decode_head(monkeypatch, embed_row)
    m.forward_decode(_T((1, 1, 8, 640)), _T((8, 1), ttnn.uint32), "pos", "cos", "sin", "pt", need_logits=False)
    assert log == [("embd", (8, 1)), ("reshape", (1, 1, 8, 640)), ("fuse", (1, 1, 8, 640))]


def test_forward_decode_embedding_flag_on(monkeypatch):
    m, log = _decode_head(monkeypatch, True)
    m.forward_decode(_T((1, 1, 8, 640)), _T((8, 1), ttnn.uint32), "pos", "cos", "sin", "pt", need_logits=False)
    assert log == [("embed_rows", (8, 1)), ("fuse", (1, 1, 8, 640))]


# ------------------------------------------------------------------------------------------------------------------ #
# QWEN36_MTP_LAYER_BFP4: dtype plumbing (layer -> attention weights / MLP) and LoFi decode configs
# ------------------------------------------------------------------------------------------------------------------ #
class _Args:
    num_devices = 8
    is_galaxy = False
    attn_qkv_fused_weight_memcfg = "FUSED_MC"
    gdn_qkvz_weight_memcfg = None
    dim = 64

    def is_full_attention_layer(self, i):
        return True

    def is_moe_layer(self, i):
        return False


def _build_layer(monkeypatch, **kw):
    seen = {}

    class _FakeAttn:
        def __init__(self, mesh, args, tw, ccl):
            self.qkv_cfg = self.wo_cfg = "HIFI2"
            seen["tw"] = tw

    def _load(mesh, sd, args, cache_dir=None, attn_dtype=None):
        seen["attn_dtype"] = attn_dtype
        return {"w": 1}

    monkeypatch.setattr(tp_mod, "TPAttention", _FakeAttn)
    monkeypatch.setattr(tp_mod, "load_attention_weights_tp", _load)
    monkeypatch.setattr(layer_mod, "Qwen36MLP", lambda *a, **k: seen.setdefault("mlp_kw", k) and "MLP")
    monkeypatch.setattr(Qwen36DecoderLayer, "_make_norm", lambda self, *a, **k: "NORM")
    monkeypatch.setattr(tpc, "get_decode_allreduce", lambda args, ccl: None)
    monkeypatch.setattr(layer_mod, "substate", lambda sd, key: sd)
    lay = Qwen36DecoderLayer("mesh", _Args(), {}, layer_num=3, tensor_cache_path=None, tt_ccl="ccl", **kw)
    return lay, seen


def test_layer_default_dtypes_unchanged(monkeypatch):
    lay, seen = _build_layer(monkeypatch)
    assert seen["attn_dtype"] is None
    assert seen["mlp_kw"] == {"args": seen["mlp_kw"]["args"], "tt_ccl": "ccl"}
    assert lay.attention.qkv_cfg == "HIFI2" and lay.attention.wo_cfg == "HIFI2"


def test_layer_bfp4_overrides(monkeypatch):
    lay, seen = _build_layer(monkeypatch, attn_dtype=ttnn.bfloat4_b, mlp_dtype=ttnn.bfloat4_b)
    assert seen["attn_dtype"] == ttnn.bfloat4_b
    assert seen["mlp_kw"]["down_dtype"] == ttnn.bfloat4_b and seen["mlp_kw"]["gateup_dtype"] == ttnn.bfloat4_b
    assert lay.attention.qkv_cfg is tpc.COMPUTE_LOFI and lay.attention.wo_cfg is tpc.COMPUTE_LOFI


def test_load_attention_weights_dtype_override(monkeypatch):
    """attn_dtype=None keeps the QWEN36_BFP4_ATTN choice (BFP8 by default); bfloat4_b loads every projection in BFP4."""
    dtypes = []
    monkeypatch.delenv("QWEN36_BFP4_ATTN", raising=False)
    monkeypatch.setattr(tpc, "shard_w", lambda *a, **k: dtypes.append((k.get("cache_path"), k["dtype"])) or "W")
    monkeypatch.setattr(tpc, "replicate", lambda *a, **k: "R")
    monkeypatch.setattr(tpc, "replicate_kv_weight", lambda w, *a: w)
    monkeypatch.setattr(tpc, "prepare_attn_qkv_deint", lambda *a, **k: torch.zeros(4, 4))
    monkeypatch.setattr(tp_mod.weight_prefetch, "attach_twin", lambda *a, **k: None)
    monkeypatch.setattr(tp_mod.weight_prefetch, "load_prefetched", lambda *a, **k: None)
    monkeypatch.setattr(tpc, "verify_heads_dense_enabled", lambda: False)
    args = types.SimpleNamespace(
        attn_qkv_fused_weight_memcfg="FUSED",
        n_kv_heads=4,
        num_devices=8,
        head_dim=256,
        n_local_heads=3,
        n_local_kv_heads=1,
        proj_1d_decode=True,
        attn_wo_weight_memcfg=None,
    )
    sd = {
        "q_proj.weight": torch.zeros(4, 4),
        "k_proj.weight": torch.zeros(4, 4),
        "v_proj.weight": torch.zeros(4, 4),
        "o_proj.weight": torch.zeros(4, 4),
        "q_norm.weight": torch.zeros(4),
        "k_norm.weight": torch.zeros(4),
    }
    tp_mod.load_attention_weights_tp("mesh", sd, args)
    assert dtypes and all(d == ttnn.bfloat8_b for _, d in dtypes)
    dtypes.clear()
    tp_mod.load_attention_weights_tp("mesh", sd, args, attn_dtype=ttnn.bfloat4_b)
    assert len(dtypes) == 2 and all(d == ttnn.bfloat4_b for _, d in dtypes)
