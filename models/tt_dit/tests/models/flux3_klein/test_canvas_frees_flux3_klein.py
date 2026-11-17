# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the canvas encoder's explicit deallocations, on a recording ttnn stand-in: no tensor is freed
twice or used after its free, and the tensors a layer hands out but does not give away (the shortcut's persistent
reduce-scatter buffer, reused by the next reduce-scatter of that shape) are never freed. No device needed."""

from __future__ import annotations

import types

import pytest

import ttnn

from ....pipelines.flux3_klein import canvas_flux3_klein as cv


class _Tensor:
    count = 0

    def __init__(self, record: _Record, shape: tuple[int, ...]) -> None:
        _Tensor.count += 1
        self.id, self.shape, self._record = _Tensor.count, tuple(shape), record
        self.layout, self.dtype = ttnn.TILE_LAYOUT, ttnn.bfloat16

    def memory_config(self):
        return None

    def __add__(self, other: _Tensor) -> _Tensor:
        self._record.use(self, other)
        return _Tensor(self._record, self.shape)


class _Record:
    def __init__(self) -> None:
        self.freed: set[int] = set()

    def new(self, shape) -> _Tensor:
        return _Tensor(self, shape)

    def use(self, *tensors) -> None:
        for t in tensors:
            for u in t if isinstance(t, (list, tuple)) else (t,):
                assert not (isinstance(u, _Tensor) and u.id in self.freed), f"tensor {u.id} used after its free"

    def free(self, t: _Tensor) -> None:
        assert t.id not in self.freed, f"tensor {t.id} freed twice"
        self.freed.add(t.id)

    def op(self, shape_of=lambda args, kwargs: args[0].shape):
        def run(*args, **kwargs):
            self.use(*args)
            return self.new(shape_of(args, kwargs))

        return run


@pytest.fixture
def record(monkeypatch) -> _Record:
    rec = _Record()
    stub = types.SimpleNamespace(**{k: getattr(ttnn, k) for k in dir(ttnn) if not k.startswith("__")})
    stub.deallocate = rec.free
    stub.reshape = rec.op(lambda a, k: a[1])
    stub.to_layout = rec.op()
    stub.mesh_partition = rec.op(lambda a, k: (*a[0].shape[:2], a[0].shape[2] // 4, a[0].shape[3]))
    stub.concat = rec.op(lambda a, k: (*a[0][0].shape[:3], a[0][0].shape[3] + 32))
    stub.slice = rec.op(lambda a, k: tuple(a[2]))
    stub.transformer = types.SimpleNamespace(scaled_dot_product_attention=rec.op())
    monkeypatch.setattr(cv, "ttnn", stub)
    monkeypatch.setattr(cv, "_all_gather_hw", lambda ctx, x: rec.new((1, 32, 32, 512)))
    monkeypatch.setattr(cv, "_partition_hw", lambda ctx, x: rec.new((1, 8, 32, 512)))
    monkeypatch.setattr(cv, "sdpa_config_for_rows", lambda config, rows: config)
    return rec


def _encoder(record: _Record, *, tp: int) -> cv.CanvasVaeEncoder:
    encoder = object.__new__(cv.CanvasVaeEncoder)
    ccl = types.SimpleNamespace(all_gather=record.op()) if tp > 1 else None
    encoder.__dict__.update(
        _ctx=types.SimpleNamespace(h_factor=4, h_mesh_axis=1, ccl_manager=ccl, tp_axis=0 if tp > 1 else None),
        _constants={},
    )
    encoder._constant = lambda shape, value: record.new(shape)
    return encoder


@pytest.mark.parametrize("shortcut", [True, False])
def test_resnet_frees_only_its_own_intermediates(record, shortcut):
    # the shortcut is a RowParallelLinear: its output is the CCL manager's persistent reduce-scatter buffer
    persistent = record.new((1, 8, 32, 512))

    def reduce_scatter(x):
        record.use(x, persistent)
        return persistent

    layer = types.SimpleNamespace(forward=record.op())
    norm = types.SimpleNamespace(forward=lambda x, mask: record.op()(x))
    block = types.SimpleNamespace(
        norm1=norm,
        conv1=layer,
        norm2=norm,
        conv2=layer,
        conv_shortcut=types.SimpleNamespace(forward=reduce_scatter) if shortcut else None,
    )
    masks = types.SimpleNamespace(shard=lambda h, w: None)
    encoder = _encoder(record, tp=4)
    for _ in range(3):  # every call after the first reuses the persistent buffer
        x = record.new((1, 8, 32, 256))
        out = encoder._resnet(block, x, masks)
        record.use(x, out, persistent)


@pytest.mark.parametrize("tp", [4, 1])
def test_attention_frees_only_its_own_intermediates(record, tp):
    def qkv(x, **kwargs):
        record.use(x)
        return tuple(record.new((1, 1, 1024, 512)) for _ in range(3))

    attn = types.SimpleNamespace(
        _ccl_manager=types.SimpleNamespace(all_gather=record.op()) if tp > 1 else None,
        _tp_axis=0 if tp > 1 else None,
        _tp_factor=tp,
        norm=types.SimpleNamespace(forward=lambda x, mask: record.op()(x)),
        to_qkv=types.SimpleNamespace(forward=qkv),
        to_out=types.SimpleNamespace(forward=record.op()),
        _mm_compute_kernel_config=None,
        _sdpa_compute_kernel_config=None,
        _sdpa_program_config=types.SimpleNamespace(q_chunk_size=128),
    )
    key_bias = record.new((1, 1, 1024, 32))
    masks = types.SimpleNamespace(full=lambda h, w: None, key_bias=key_bias)
    x = record.new((1, 8, 32, 512))
    out = _encoder(record, tp=tp)._attention(attn, x, masks)
    record.use(x, out, key_bias)
