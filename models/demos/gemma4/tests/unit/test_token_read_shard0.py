# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GEMMA4_TOKEN_READ_SHARD0 (item 3a of the gemma4_12b_qb2 port): the one-shard decode readback hands the host the
same tokens and logits as the whole-mesh read, through the mixin's ``read_decode_output`` and the shared
``process_decode_output_host`` (blocking and event-based), and stays off where one shard is not the whole answer.

    pytest -k "1x4" models/demos/gemma4/tests/unit/test_token_read_shard0.py
"""

import types
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.generator import (
    TOKEN_READ_SHARD0_ENV,
    ChunkedPrefillPageTableGuardMixin,
    token_read_shard0_default,
)
from models.demos.gemma4.tt.model import Gemma4Model
from models.tt_transformers.tt.generator import Generator

from ...tests.test_factory import parametrize_mesh_with_fabric

VOCAB = 262144
ROWS = 32


class _Reader(ChunkedPrefillPageTableGuardMixin, Generator):
    """The served readback chain (mixin over the shared Generator) over a stub model, no device model built."""

    def __init__(self, model, mesh_device):
        self.model = [model]
        self.model_args = [SimpleNamespace(mesh_device=mesh_device, max_batch_size=ROWS)]
        self.data_parallel = 1


def _stub_model(mesh_device, *, lane_sharded=False, sampling_dp=1):
    tp = int(mesh_device.shape[1])
    model = SimpleNamespace(
        mesh_device=mesh_device,
        _lane_sharded=lane_sharded,
        sampling_dp=sampling_dp,
        mesh_config=SimpleNamespace(tp=tp),
        vocab_size=VOCAB,
        lane_device_indices=lambda: [0],
    )
    model.process_output_decode = types.MethodType(Gemma4Model.process_output_decode, model)
    return model


def _read(reader, outputs, *, is_tokens, async_read):
    if async_read:
        host, events = reader.read_decode_output(outputs, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
    else:
        host = reader.read_decode_output(outputs)
    return reader.process_decode_output_host(host, is_tokens=is_tokens)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
@pytest.mark.parametrize("async_read", [False, True], ids=["blocking", "events"])
def test_token_read_shard0_matches_mesh_read(mesh_device, device_params, async_read, monkeypatch):
    """Replicated tokens [1,1,1,32] and logits [1,1,32,V]: knob on == knob off, as tensors and as (tensor, None)."""
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tokens_torch = torch.randint(0, VOCAB, (1, 1, 1, ROWS), dtype=torch.int32)
    tokens = ttnn.from_torch(
        tokens_torch, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=replicate
    )
    logits_torch = torch.randn(1, 1, ROWS, VOCAB).to(torch.bfloat16)
    logits = ttnn.from_torch(
        logits_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=replicate
    )
    reader = _Reader(_stub_model(mesh_device), mesh_device)
    results = {}
    for knob in ("0", "1"):
        monkeypatch.setenv(TOKEN_READ_SHARD0_ENV, knob)
        results[knob] = (
            _read(reader, [tokens], is_tokens=True, async_read=async_read),
            _read(reader, [(tokens, None)], is_tokens=True, async_read=async_read),
            _read(reader, [logits], is_tokens=False, async_read=async_read),
        )
    for name, off, on in zip(("tokens", "(tokens, None)", "logits"), results["0"], results["1"]):
        off_t, on_t = (off[0], on[0]) if isinstance(off, tuple) else (off, on)
        assert torch.equal(off_t, on_t), f"{name}: one-shard read differs from the mesh read"
    toks = results["1"][0]
    toks = toks[0] if isinstance(toks, tuple) else toks
    assert torch.equal(toks.reshape(-1)[:ROWS].to(torch.int32), tokens_torch.reshape(-1))
    lg = results["1"][2][0]
    assert torch.equal(lg.reshape(ROWS, VOCAB), logits_torch.reshape(ROWS, VOCAB))


def test_token_read_shard0_gates(monkeypatch):
    """Host: the env var overrides, else the measured-shape policy; never under lanes / a row-sharded sampler."""
    mesh = SimpleNamespace(shape=(1, 4), get_num_devices=lambda: 4)
    monkeypatch.setenv(TOKEN_READ_SHARD0_ENV, "1")
    assert token_read_shard0_default(mesh) is True
    reader = _Reader(None, mesh)
    assert reader._g4_shard0_read_ok(SimpleNamespace(mesh_device=mesh, _lane_sharded=False, sampling_dp=1)) is True
    assert reader._g4_shard0_read_ok(SimpleNamespace(mesh_device=mesh, _lane_sharded=True, sampling_dp=1)) is False
    assert reader._g4_shard0_read_ok(SimpleNamespace(mesh_device=mesh, _lane_sharded=False, sampling_dp=4)) is False
    one = SimpleNamespace(shape=(1, 1), get_num_devices=lambda: 1)
    assert reader._g4_shard0_read_ok(SimpleNamespace(mesh_device=one, _lane_sharded=False, sampling_dp=1)) is False
    monkeypatch.setenv(TOKEN_READ_SHARD0_ENV, "0")
    assert token_read_shard0_default(mesh) is False
    monkeypatch.delenv(TOKEN_READ_SHARD0_ENV)
    from models.demos.gemma4.tt import generator as generator_module

    monkeypatch.setattr(generator_module, "TOKEN_READ_SHARD0_DEFAULT_SHAPES", ((1, 4), (1, 8)))
    monkeypatch.setattr(generator_module, "is_blackhole", lambda: True)
    assert token_read_shard0_default(mesh) is True
    assert token_read_shard0_default(SimpleNamespace(shape=(1, 8))) is True
    assert token_read_shard0_default(SimpleNamespace(shape=(8, 4))) is False
    monkeypatch.setattr(generator_module, "is_blackhole", lambda: False)  # Wormhole (T3K is (1,8) too)
    assert token_read_shard0_default(mesh) is False
    assert token_read_shard0_default(SimpleNamespace(shape=(1, 8))) is False
