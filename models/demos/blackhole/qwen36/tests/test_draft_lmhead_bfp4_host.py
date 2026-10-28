# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for QWEN36_DRAFT_LMHEAD_BFP4 (MTP drafter's own bfloat4_b LM-head copy). No device: ttnn
ops are monkeypatched with recorders.

Checks: the flag parses default-off; the base head keeps its exact cache names and dtype; the drafter copy
gets new cache names; every drafter logits branch of Qwen36MTP.forward_decode reads the drafter weight;
Qwen36Model._lm_head without ``weight`` still reads the base weight.

Run: pytest models/demos/blackhole/qwen36/tests/test_draft_lmhead_bfp4_host.py -q
"""
import pathlib
import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

BASE_W = object()
DRAFT_W = object()


@pytest.fixture
def fresh_flag(monkeypatch):
    monkeypatch.setattr(tp_common, "_DRAFT_LMHEAD_BFP4_LOGGED", False)
    monkeypatch.delenv(tp_common.DRAFT_LMHEAD_BFP4_ENV, raising=False)
    return monkeypatch


def test_flag_default_off(fresh_flag):
    assert tp_common.DRAFT_LMHEAD_BFP4_ENV == "QWEN36_DRAFT_LMHEAD_BFP4"
    assert tp_common.draft_lmhead_bfp4_enabled() is False


@pytest.mark.parametrize("val,expect", [("1", True), ("0", False), ("", False), ("true", False)])
def test_flag_values(fresh_flag, val, expect):
    fresh_flag.setenv(tp_common.DRAFT_LMHEAD_BFP4_ENV, val)
    assert tp_common.draft_lmhead_bfp4_enabled() is expect


def _record_as_tensor(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "as_tensor", lambda t, **kw: calls.append(kw) or ("dev", kw["dtype"]))
    monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda dev, dim: ("shard", dim))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda dev: ("repl",))
    return calls


@pytest.mark.parametrize("sharded,nd", [(True, 8), (False, 8), (False, 1)])
def test_build_lm_head_weight_cache_names(monkeypatch, sharded, nd):
    calls = _record_as_tensor(monkeypatch)
    fake = types.SimpleNamespace(_lmhead_vocab_sharded=sharded, mesh_device="mesh", num_devices=nd)
    cache = pathlib.Path("/cache")
    stem = "output.weight.vshard" if sharded else "output.weight"
    Qwen36Model._build_lm_head_weight(fake, "W", cache, ttnn.bfloat8_b)
    Qwen36Model._build_lm_head_weight(fake, "W", cache, ttnn.bfloat4_b, cache_tag=".draft_bfp4")
    base, draft = calls
    # Base head: exactly the pre-existing cache file name and dtype, so existing caches are reused.
    assert base["cache_file_name"] == cache / stem and base["dtype"] == ttnn.bfloat8_b
    # Drafter copy: its own cache name and BFP4; same mapper/layout/memory as the base.
    assert draft["cache_file_name"] == cache / f"{stem}.draft_bfp4" and draft["dtype"] == ttnn.bfloat4_b
    for k in ("layout", "memory_config", "device"):
        assert base[k] == draft[k]
    assert base.get("mesh_mapper") == draft.get("mesh_mapper")
    assert ("mesh_mapper" in base) == (nd > 1)
    # No cache path -> no cache file (unchanged behaviour).
    Qwen36Model._build_lm_head_weight(fake, "W", None, ttnn.bfloat4_b, cache_tag=".draft_bfp4")
    assert calls[-1]["cache_file_name"] is None


def _record_linear(monkeypatch):
    seen = []
    monkeypatch.setattr(ttnn, "linear", lambda x, w, **kw: seen.append((w, kw)) or "logits")
    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda *a, **kw: "cfg")
    return seen


def test_lm_head_weight_kwarg(monkeypatch):
    seen = _record_linear(monkeypatch)
    fake = types.SimpleNamespace(
        lm_head_weight=BASE_W,
        _lmhead_vocab_sharded=False,
        device=types.SimpleNamespace(arch=lambda: "bh"),
    )
    Qwen36Model._lm_head(fake, "x")
    Qwen36Model._lm_head(fake, "x", out_dtype=ttnn.float32)
    Qwen36Model._lm_head(fake, "x", out_dtype=ttnn.float32, weight=DRAFT_W)
    assert [w for w, _ in seen] == [BASE_W, BASE_W, DRAFT_W]
    assert seen[2][1]["dtype"] == ttnn.float32


def _fake_mtp(monkeypatch, sharded_argmax, ondev_argmax=False):
    seen = _record_linear(monkeypatch)
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: t)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "clone", lambda t, **kw: t)
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices = 1
    m.args = None
    m.embd = lambda ids: types.SimpleNamespace(shape=(1, 1, 64))
    m._fuse = lambda a, b, mode: "fused"
    m.decoder = types.SimpleNamespace(forward=lambda *a, **kw: "hidden")
    m.head_norm = lambda x, mode, norm_config: "normed"
    m.lm_head_weight = BASE_W
    m.draft_lm_head_weight = DRAFT_W
    m._sharded_argmax = sharded_argmax
    m._ondev_argmax = ondev_argmax
    m._fp32_cfg = "fp32cfg"
    m._lm_head = lambda x, out_dtype=None, gather=True, weight=None: seen.append(
        (BASE_W if weight is None else weight, {"dtype": out_dtype})
    )
    return m, seen


@pytest.mark.parametrize(
    "sharded_argmax,sharded_lm_head,ondev",
    [(True, False, False), (False, True, False), (False, False, True), (False, False, False)],
)
def test_forward_decode_uses_draft_weight(monkeypatch, sharded_argmax, sharded_lm_head, ondev):
    m, seen = _fake_mtp(monkeypatch, sharded_argmax, ondev)
    m.forward_decode("h", "tok", "pos", "cos", "sin", "pt", sharded_lm_head=sharded_lm_head)
    assert len(seen) == 1 and seen[0][0] is DRAFT_W


def test_forward_decode_flag_off_is_shared_head(monkeypatch):
    m, seen = _fake_mtp(monkeypatch, True)
    m.draft_lm_head_weight = m.lm_head_weight  # what __init__ does with the flag off
    m.forward_decode("h", "tok", "pos", "cos", "sin", "pt")
    assert seen[0][0] is BASE_W
