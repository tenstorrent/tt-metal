# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only decode formatting: batch buckets must never split vocabulary rows."""

from types import MethodType, SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gpt_oss.tt.model import Model as GptOssModel
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer


@pytest.fixture(autouse=True)
def host_conversions(monkeypatch):
    # Exercise the production formatters without constructing native tensors or devices.
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor: tensor)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda tensors: tensors)
    monkeypatch.setattr(ttnn, "reshape", lambda tensor, shape: tensor.reshape(tuple(shape)))


def make_model(model_class, tp, *, row_sharded=False):
    model = SimpleNamespace(
        vocab_size=100,
        args=SimpleNamespace(num_devices=tp),
        mesh_config=SimpleNamespace(tp=tp, get_config=lambda mode: SimpleNamespace(tp=tp)),
        users_row_sharded=row_sharded,
        concat_device_output=lambda tensor: tensor,
        concat_host_output=lambda tensor, *args: tensor,
    )
    model.process_output_decode = MethodType(model_class.process_output_decode, model)
    if model_class is GptOssModel:
        model._decode_host_shard = GptOssModel._decode_host_shard
    return model


def make_logits(rows, offset=0):
    # Distinct row/token values and a padded width catch row splitting and early trimming.
    return torch.arange(rows * 128, dtype=torch.float32).reshape(1, 1, rows, 128) + offset


def device_output(model_class, logits, tp):
    if tp == 1:
        return logits
    if model_class is GptOssModel:
        return list(logits.chunk(tp, dim=-1))
    # These models already gather logits on device, so TP ranks are replicas.
    return [logits.clone() for _ in range(tp)]


@pytest.mark.parametrize("model_class", [Transformer, Gemma4Model, GptOssModel])
@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("rows,limit", [(1, 32), (8, 32), (32, 32), (32, 7)])
def test_decode_preserves_available_rows_and_full_vocab(model_class, tp, rows, limit):
    model = make_model(model_class, tp)
    logits = make_logits(rows)
    actual = model.process_output_decode(device_output(model_class, logits, tp), B=limit)
    expected = logits[0, 0, :limit, :100].unsqueeze(1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_class", [Transformer, Gemma4Model, GptOssModel])
@pytest.mark.parametrize("widths", [(32, 1), (1, 32)])
def test_pending_outputs_keep_their_own_rows(model_class, widths):
    model = make_model(model_class, tp=4)
    generator = SimpleNamespace(model=[model], model_args=[SimpleNamespace(max_batch_size=32)], data_parallel=1)
    logits = [make_logits(rows, offset=10000 * i) for i, rows in enumerate(widths)]
    # Both outputs have been submitted before either is processed, as in async decode.
    pending = [device_output(model_class, output, 4) for output in logits]
    for output, expected in zip(pending, logits):
        actual, log_probs = Generator.process_decode_output_host(generator, [(output, None)], is_tokens=False)
        torch.testing.assert_close(actual, expected[0, 0, :, :100].unsqueeze(1), rtol=0, atol=0)
        assert log_probs is None


@pytest.mark.parametrize("row_sharded", [False, True])
def test_gpt_oss_tp_gather_distinguishes_dp_users_from_ep_replicas(row_sharded):
    model = make_model(GptOssModel, tp=4, row_sharded=row_sharded)
    first = make_logits(4)
    second = make_logits(4, offset=10000) if row_sharded else first.clone()
    shards = list(first.chunk(4, dim=-1)) + list(second.chunk(4, dim=-1))
    actual = model.process_output_decode(shards, B=32)
    expected = torch.cat([first, second], dim=-2) if row_sharded else first
    torch.testing.assert_close(actual, expected[0, 0, :, :100].unsqueeze(1), rtol=0, atol=0)


@pytest.mark.parametrize("model_class", [Transformer, Gemma4Model, GptOssModel])
@pytest.mark.parametrize("is_log_probs", [False, True])
def test_sampled_tokens_and_log_probs_keep_their_existing_shape(model_class, is_log_probs):
    model = make_model(model_class, tp=1)
    output = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32, 1)
    actual = model.process_output_decode(output, B=7, is_tokens=not is_log_probs, is_log_probs=is_log_probs)
    torch.testing.assert_close(actual, torch.arange(7, dtype=torch.float32), rtol=0, atol=0)
