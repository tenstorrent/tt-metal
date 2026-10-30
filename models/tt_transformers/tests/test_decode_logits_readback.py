# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Host readback of decode logits: only the live rows, from the first device. Host only, no device."""

from types import MethodType, SimpleNamespace

import pytest
import torch

import ttnn
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer

VOCAB = 128
PADDED_ROWS = 32


class FakeLogits:
    """Stands in for the row-major [1, 1, 32, vocab] decode logits replicated on every device of a mesh."""

    def __init__(self, data, replicas):
        self.data = data  # torch [1, 1, rows, vocab]
        self.replicas = replicas
        self.layout = ttnn.ROW_MAJOR_LAYOUT
        self.shape = list(data.shape)
        self.cpu_calls = []

    def cpu(self, blocking=True):
        self.cpu_calls.append(blocking)
        return ("host", self.data, self.replicas)


@pytest.fixture(autouse=True)
def fake_ttnn(monkeypatch):
    def fake_slice(tensor, start, end):
        assert list(start) == [0, 0, 0, 0]
        assert end[0] == 1 and end[1] == 1 and end[3] == tensor.shape[3]
        return FakeLogits(tensor.data[:, :, : end[2], :], tensor.replicas)

    monkeypatch.setattr(ttnn, "slice", fake_slice)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [FakeLogits(t.data, 1) for _ in range(t.replicas)])
    monkeypatch.setattr(ttnn, "is_tensor_storage_on_device", lambda t: isinstance(t, FakeLogits))
    # The generator dispatches on isinstance(x, ttnn.Tensor); let the stand-in pass that check.
    monkeypatch.setattr(ttnn, "Tensor", (ttnn.Tensor, FakeLogits))


def make_model(max_batch_size):
    model = SimpleNamespace(args=SimpleNamespace(max_batch_size=max_batch_size, num_devices=2))
    model.read_decode_logits = MethodType(Transformer.read_decode_logits, model)
    return model


def make_logits(rows=PADDED_ROWS, replicas=2):
    return FakeLogits(torch.arange(rows * VOCAB, dtype=torch.float32).reshape(1, 1, rows, VOCAB), replicas)


@pytest.mark.parametrize("max_batch_size, expected_rows", [(1, 1), (8, 8), (32, 32), (64, 32)])
def test_reads_only_the_live_rows_from_one_device(max_batch_size, expected_rows):
    logits = make_logits()
    kind, data, replicas = make_model(max_batch_size).read_decode_logits(logits)
    assert kind == "host" and replicas == 1, "read from a single device"
    torch.testing.assert_close(data, logits.data[:, :, :expected_rows, :], rtol=0, atol=0)
    assert logits.cpu_calls == [], "the full replicated tensor is not read"


def test_non_blocking_read_is_forwarded():
    logits = make_logits()
    model = make_model(8)
    model.read_decode_logits(logits, blocking=False)


def test_tile_layout_and_host_tensors_read_as_before():
    logits = make_logits()
    logits.layout = ttnn.TILE_LAYOUT
    assert make_model(8).read_decode_logits(logits) == ("host", logits.data, 2)
    assert logits.cpu_calls == [True]


def test_generator_uses_the_model_reader_and_falls_back_to_cpu():
    reading = make_model(8)
    plain = SimpleNamespace()  # a model without read_decode_logits (e.g. vocab-sharded logits)
    generator = SimpleNamespace(model=[reading, plain], model_args=[None, None], data_parallel=2)
    outputs = [make_logits(), make_logits()]
    host = Generator.read_decode_output(generator, outputs)
    assert host[0][1].shape[2] == 8 and host[0][2] == 1
    assert host[1][1].shape[2] == PADDED_ROWS and host[1][2] == 2
    assert outputs[0].cpu_calls == [] and outputs[1].cpu_calls == [True]

    # with logprobs the logits still go through the reader; logprobs are read as before
    lp = FakeLogits(torch.zeros(1, 1, 32, 5), 2)
    host = Generator.read_decode_output(generator, [(make_logits(), lp), (make_logits(), None)])
    assert host[0][0][1].shape[2] == 8 and host[0][1] == ("host", lp.data, 2)
    assert host[1][1] is None
