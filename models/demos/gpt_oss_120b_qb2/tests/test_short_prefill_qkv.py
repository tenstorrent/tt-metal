# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.gpt_oss_120b_qb2.tt import attention_prefill
from models.demos.gpt_oss_120b_qb2.tt.multichip_decoder import _PhysicalHiddenCollectiveAttention
from models.demos.gpt_oss_120b_qb2.tt.precision import load_precision_config


class Tensor:
    def __init__(self, shape, owner, rows=None):
        self.shape = shape
        self.owner = owner
        self.rows = rows
        self.dead = False

    def deallocate(self, force):
        assert force is True and not self.dead
        self.dead = True


class Native:
    bfloat16 = object()
    DRAM_MEMORY_CONFIG = object()

    def __init__(self, fail=None):
        self.created = []
        self.calls = []
        self.fail = fail

    def slice(self, x, start, end, **kw):
        assert not x.dead and kw == dict(memory_config=self.DRAM_MEMORY_CONFIG)
        t = Tensor((1, 1, end[2] - start[2], 2880), "slice", range(start[2], end[2]))
        self.created.append(t)
        return t

    def linear(self, x, w, **kw):
        assert not x.dead
        self.calls.append((x, w, kw))
        if self.fail == len(self.calls):
            raise RuntimeError("linear")
        t = Tensor((*x.shape[:-1], w.shape[-1]), "linear", x.rows)
        self.created.append(t)
        return t

    def concat(self, outputs, **kw):
        assert kw == dict(dim=2, memory_config=self.DRAM_MEMORY_CONFIG)
        assert all(not t.dead for t in outputs)
        if self.fail == "concat":
            raise RuntimeError("concat")
        t = Tensor(
            (1, 1, sum(t.shape[2] for t in outputs), outputs[0].shape[-1]),
            "return",
            sum((list(t.rows) for t in outputs), []),
        )
        self.created.append(t)
        return t


def helper(native):
    return lambda x, w, b, c: attention_prefill._project_qkv(x, w, b, c)


@pytest.fixture(autouse=True)
def native_patch(monkeypatch):
    # Real module imports/config classes, CPU operation doubles only.
    yield monkeypatch


@pytest.mark.parametrize("layer", range(36))
def test_each_layer_forwarding_order_lifetime(layer, monkeypatch):
    n = Native()
    x = Tensor((1, 1, 512, 2880), "caller", range(512))
    w = SimpleNamespace(wqkv=Tensor((1, 1, 2880, 1024), "weight"), wqkv_bias=object())
    cfg = ttnn.init_device_compute_kernel_config(
        ttnn.Arch.BLACKHOLE,
        math_fidelity=load_precision_config().decoder_policy_for_layer(layer).prefill_projection_math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    monkeypatch.setattr(attention_prefill, "ttnn", n)
    y = helper(n)(x, w, 4, cfg)
    assert y.shape == (1, 1, 512, 1024) and y.rows == list(range(512)) and not y.dead and not x.dead and not w.wqkv.dead
    assert len(n.calls) == 4
    for slab, weight, kw in n.calls:
        assert weight is w.wqkv and kw == dict(bias=w.wqkv_bias, dtype=n.bfloat16, compute_kernel_config=cfg)
    assert all(t.dead for t in n.created if t is not y)


@pytest.mark.parametrize(
    "batch,seq,width", [(1, 128, 2880), (4, 256, 2880), (4, 64, 2880), (4, 128, 4096), (1, 1024, 2880)]
)
def test_unchanged_fallback(batch, seq, width, monkeypatch):
    n = Native()
    x = Tensor((1, 1, batch * seq, width), "caller")
    w = SimpleNamespace(wqkv=Tensor((1, 1, width, 1024), "weight"), wqkv_bias=object())
    cfg = object()
    monkeypatch.setattr(attention_prefill, "ttnn", n)
    y = helper(n)(x, w, batch, cfg)
    assert n.calls == [(x, w.wqkv, dict(bias=w.wqkv_bias, dtype=n.bfloat16, compute_kernel_config=cfg))]
    assert not x.dead and not y.dead and len(n.created) == 1


@pytest.mark.parametrize("failure", [1, 2, 3, 4, "concat"])
def test_exception_frees_only_owned(failure, monkeypatch):
    n = Native(failure)
    x = Tensor((1, 1, 512, 2880), "caller")
    w = SimpleNamespace(wqkv=Tensor((1, 1, 2880, 1024), "weight"), wqkv_bias=object())
    monkeypatch.setattr(attention_prefill, "ttnn", n)
    with pytest.raises(RuntimeError):  # allow-pytest.raises: CPU operation double, no native error.
        helper(n)(x, w, 4, None)
    assert not x.dead and not w.wqkv.dead and all(t.dead for t in n.created)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
def test_actual_prefill_call_owns_input_and_reshapes(batch, monkeypatch):
    n = Native()
    x = Tensor((1, 1, batch * 128, 2880), "caller", range(batch * 128))
    w = SimpleNamespace(wqkv=Tensor((1, 1, 2880, 1024), "weight"), wqkv_bias=object())
    cfg = object()
    caught = []

    def reshape(t, shape):
        assert x.dead and not t.dead
        t.shape = (batch, 1, 128, 1024)
        return t

    n.reshape = reshape
    monkeypatch.setattr(attention_prefill, "ttnn", n)

    class StopAtSplit(Exception):
        pass

    def split(t, *args):
        assert x.dead and not t.dead
        assert t.shape == ((batch, 1, 128, 1024) if batch > 1 else (1, 1, 128, 1024))
        caught.append(t)
        raise StopAtSplit

    monkeypatch.setattr(attention_prefill, "split_qkv_heads_prefill", split)
    obj = object.__new__(_PhysicalHiddenCollectiveAttention)
    obj.kv_cache = object()
    obj.weights = w
    obj.config = SimpleNamespace(num_heads=64, num_kv_heads=8, head_dim=64)
    obj.mesh_config = SimpleNamespace(shard_size=lambda v: v // 4)
    obj.mesh_device = object()
    obj.program_config = object()
    obj.transformation_mats = {"prefill": object()}
    obj.ccl_manager = object()
    obj.prefill_projection_input_dtype = object()
    obj.prefill_projection_compute_kernel_config = cfg
    with pytest.raises(StopAtSplit):  # allow-pytest.raises: CPU-only sentinel stops before device operations.
        obj(x, object(), is_decode=False, batch_size=batch)
    assert len(caught) == 1 and not w.wqkv.dead
    assert len(n.calls) == batch
    assert all(kw["compute_kernel_config"] is cfg for _, _, kw in n.calls)
    caught[0].deallocate(True)


def test_decode_never_enters_prefill(monkeypatch):
    obj = object.__new__(_PhysicalHiddenCollectiveAttention)
    obj.kv_cache = object()
    obj.transformation_mats = {"decode": object()}
    expected = object()
    calls = []
    obj._decode_forward = lambda *a, **kw: calls.append((a, kw)) or expected
    import models.demos.gpt_oss_120b_qb2.tt.multichip_decoder as decoder

    monkeypatch.setattr(decoder, "prefill_forward", lambda **kw: pytest.fail("decode entered prefill"))
    assert obj(object(), object(), is_decode=True) is expected
    assert len(calls) == 1
