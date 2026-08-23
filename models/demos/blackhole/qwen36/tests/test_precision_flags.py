# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only gates for the decode precision ladder.

The ladder is four independent env flags, each default OFF:

    QWEN_SDPA_BF8=1        bfloat8_b paged KV cache (+ bf8 Q/KV into SDPA)
    QWEN36_MLP_DOWN_BF4=1  bfloat4_b MLP down-proj (w2) weights
    QWEN36_GDN_BF4=1       bfloat4_b GDN projections (qkvzab in-proj + out-proj)
    QWEN36_LM_HEAD_BF4=1   bfloat4_b LM head weights

Two invariants are pinned here without a device:

1. All flags off => every helper returns the base dtype and an EMPTY cache
   tag, so the flags-off stack is byte-identical to the base (same weights,
   same cache files).
2. A dtype change produces a DISTINCT weight-cache key. Converted weights are
   cached on NFS (TT_CACHE_PATH); a bfp4 arm silently reloading a bfp8 cache
   file would measure bfp8 while claiming bfp4 — the loaders are exercised
   with recording stubs to prove the real cache paths diverge per flag.

No device or weights needed; ttnn python must import (run on any TT host):

    pytest models/demos/blackhole/qwen36/tests/test_precision_flags.py --noconftest -v
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.gdn.tp import load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.gdn.weights import gdn_proj_dtype_and_tag
from models.demos.blackhole.qwen36.tt.mlp import _mlp_down_dtype_and_tag, load_mlp_weights
from models.demos.blackhole.qwen36.tt.model import kv_cache_dtype, lm_head_dtype_and_tag

_ALL_FLAGS = ["QWEN_SDPA_BF8", "QWEN36_MLP_DOWN_BF4", "QWEN36_GDN_BF4", "QWEN36_LM_HEAD_BF4"]


@pytest.fixture(autouse=True)
def _clean_ladder_env(monkeypatch):
    for f in _ALL_FLAGS:
        monkeypatch.delenv(f, raising=False)


def test_defaults_off_are_base_dtypes():
    assert _mlp_down_dtype_and_tag() == (ttnn.bfloat8_b, "")
    assert gdn_proj_dtype_and_tag() == (ttnn.bfloat8_b, "")
    assert lm_head_dtype_and_tag() == (ttnn.bfloat8_b, "")
    assert kv_cache_dtype(ttnn.bfloat16) == ttnn.bfloat16
    assert kv_cache_dtype(ttnn.bfloat8_b) == ttnn.bfloat8_b


@pytest.mark.parametrize(
    "flag, helper",
    [
        ("QWEN36_MLP_DOWN_BF4", _mlp_down_dtype_and_tag),
        ("QWEN36_GDN_BF4", gdn_proj_dtype_and_tag),
        ("QWEN36_LM_HEAD_BF4", lm_head_dtype_and_tag),
    ],
)
def test_bfp4_flags_switch_dtype_and_tag(monkeypatch, flag, helper):
    monkeypatch.setenv(flag, "1")
    assert helper() == (ttnn.bfloat4_b, ".bfp4")
    monkeypatch.setenv(flag, "0")
    assert helper() == (ttnn.bfloat8_b, "")


def test_kv_bfp8_flag(monkeypatch):
    monkeypatch.setenv("QWEN_SDPA_BF8", "1")
    assert kv_cache_dtype(ttnn.bfloat16) == ttnn.bfloat8_b
    monkeypatch.setenv("QWEN_SDPA_BF8", "0")
    assert kv_cache_dtype(ttnn.bfloat16) == ttnn.bfloat16


class _ShardRecorder:
    """Stands in for tp_common.shard_w/shard_small/replicate; records (cache_path, dtype)."""

    def __init__(self):
        self.calls = {}

    def shard_w(self, t, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
        self.calls[Path(cache_path).name] = dtype
        return f"w:{cache_path}"

    def shard_small(self, t, mesh, cache_path, dim=-1, dtype=ttnn.bfloat16):
        self.calls[Path(cache_path).name] = dtype
        return torch.zeros(1)

    def replicate(self, t, mesh, cache_path, dtype=ttnn.bfloat16):
        self.calls[Path(cache_path).name] = dtype
        return torch.zeros(1)


def _run_mlp_loader(monkeypatch, tmp_path, flag_on):
    monkeypatch.setenv("QWEN36_MLP_DOWN_BF4", "1" if flag_on else "0")
    rec = _ShardRecorder()
    monkeypatch.setattr(tpc, "shard_w", rec.shard_w)
    monkeypatch.setattr(tpc, "mlp_gateup_agmm_enabled", lambda tp: False)
    sd = {f"{n}.weight": torch.zeros(8, 4) for n in ("gate_proj", "up_proj", "down_proj")}
    args = SimpleNamespace(num_devices=8, mlp_w1_weight_memcfg=None, mlp_1d_decode=True)
    load_mlp_weights(mesh_device=object(), state_dict=sd, tensor_cache_path=tmp_path, args=args)
    return rec.calls


def test_mlp_w2_cache_key_uniqueness(monkeypatch, tmp_path):
    off = _run_mlp_loader(monkeypatch, tmp_path, flag_on=False)
    on = _run_mlp_loader(monkeypatch, tmp_path, flag_on=True)

    assert off["mlp.down_proj.weight.tp"] == ttnn.bfloat8_b
    assert on["mlp.down_proj.weight.bfp4.tp"] == ttnn.bfloat4_b
    assert "mlp.down_proj.weight.tp" not in on, "bfp4 arm reused the bfp8 cache key"
    # Gate/up untouched by the flag (already bfp4 in the base stack).
    for k in ("mlp.gate_proj.weight.tp", "mlp.up_proj.weight.tp"):
        assert off[k] == on[k] == ttnn.bfloat4_b


def _run_gdn_loader(monkeypatch, tmp_path, flag_on):
    monkeypatch.setenv("QWEN36_GDN_BF4", "1" if flag_on else "0")
    rec = _ShardRecorder()
    monkeypatch.setattr(tpc, "shard_w", rec.shard_w)
    monkeypatch.setattr(tpc, "shard_small", rec.shard_small)
    monkeypatch.setattr(tpc, "replicate", rec.replicate)
    monkeypatch.setattr(ttnn, "exp", lambda t: t)
    monkeypatch.setattr(ttnn, "neg", lambda t: t)
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda mesh, dim: None)

    tp, nk, dk, nv, dv, hidden = 2, 2, 8, 4, 8, 16
    key_dim, value_dim = nk * dk, nv * dv
    qkv_dim = 2 * key_dim + value_dim
    sd = {
        "in_proj_qkv.weight": torch.zeros(qkv_dim, hidden),
        "in_proj_z.weight": torch.zeros(value_dim, hidden),
        "in_proj_a.weight": torch.zeros(nv, hidden),
        "in_proj_b.weight": torch.zeros(nv, hidden),
        "out_proj.weight": torch.zeros(hidden, value_dim),
        "dt_bias": torch.zeros(nv),
        "A_log": torch.zeros(nv),
        "norm.weight": torch.zeros(dv),
        "conv1d.weight": torch.zeros(qkv_dim, 1, 4),
    }
    args = SimpleNamespace(
        num_devices=tp,
        gdn_nk=nk,
        gdn_dk=dk,
        gdn_nv=nv,
        gdn_dv=dv,
        gdn_key_dim=key_dim,
        gdn_value_dim=value_dim,
        gdn_qkv_dim=qkv_dim,
        gdn_qkv_dim_tp=qkv_dim // tp,
        gdn_z_dim_tp=value_dim // tp,
        gdn_nv_tp=nv // tp,
        gdn_conv_kernel_size=4,
        gdn_qkvz_weight_memcfg=object(),  # engage the fused-qkvzab production path
        gdn_qkvzab_weight_memcfg=None,
        gdn_out_weight_memcfg=None,
        proj_1d_decode=True,
    )
    load_gdn_weights_tp(object(), sd, args, cache_dir=tmp_path)
    return rec.calls


def test_gdn_proj_cache_key_uniqueness(monkeypatch, tmp_path):
    off = _run_gdn_loader(monkeypatch, tmp_path, flag_on=False)
    on = _run_gdn_loader(monkeypatch, tmp_path, flag_on=True)

    assert off["qkvzab.il"] == ttnn.bfloat8_b
    assert off["out"] == ttnn.bfloat8_b
    assert on["qkvzab.il.bfp4"] == ttnn.bfloat4_b
    assert on["out.bfp4"] == ttnn.bfloat4_b
    for stale in ("qkvzab.il", "out"):
        assert stale not in on, f"bfp4 arm reused the bfp8 cache key {stale!r}"
    # Non-projection params untouched by the flag.
    for k in ("dt_bias", "A_log", "norm_w", "tap0", "tap1", "tap2", "tap3"):
        assert off[k] == on[k]


def test_lm_head_cache_key_uniqueness(monkeypatch, tmp_path):
    # Qwen36Model.__init__ composes `output.weight.vshard{tag}` / `output.weight{tag}`
    # from this helper; distinct tags per flag state give distinct cache keys.
    names = {}
    for flag_on in (False, True):
        monkeypatch.setenv("QWEN36_LM_HEAD_BF4", "1" if flag_on else "0")
        dtype, tag = lm_head_dtype_and_tag()
        assert dtype == (ttnn.bfloat4_b if flag_on else ttnn.bfloat8_b)
        names[flag_on] = str(tmp_path / f"output.weight.vshard{tag}")
    assert names[False] != names[True], "bfp4 LM head would reuse the bfp8 cache key"
