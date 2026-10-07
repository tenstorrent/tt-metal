# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.model import Model
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator import Generator


@pytest.mark.parametrize("remap_slots", [False, True])
@pytest.mark.parametrize("reset_batch", [False, True])
def test_decode_keeps_feedback_from_every_mesh_row(monkeypatch, remap_slots, reset_batch):
    """New prefills must not roll continuing requests back to stale host tokens."""
    batch = 128
    device_tokens = torch.arange(batch) + 1000
    device_positions = torch.arange(batch) + 200
    remap = torch.arange(batch)
    if remap_slots:
        for row in range(4):
            remap[row * 32], remap[row * 32 + 1] = row * 32 + 1, row * 32
    host_tokens = torch.arange(batch).reshape(batch, 1)
    host_positions = device_positions[remap] - 1
    prefilled = {2, 35, 68, 101}
    inactive = [31, 63, 95, 127]
    host_positions[inactive] = -1
    expected_tokens = device_tokens[remap].clone()
    expected_positions = device_positions[remap].clone()
    keep_host = sorted(prefilled) + inactive
    expected_tokens[keep_host] = host_tokens.reshape(-1)[keep_host]
    expected_positions[keep_host] = host_positions[keep_host]

    # Each TP column carries the same 32-slot row shard. Tokens are 4-D;
    # positions are 1-D, as in Model.prepare_decode_inputs_host.
    token_shards = [part.reshape(1, 1, 1, 32) for part in device_tokens.chunk(4) for _ in range(8)]
    position_shards = [part for part in device_positions.chunk(4) for _ in range(8)]
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda tensor: tensor)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor: tensor)
    model = SimpleNamespace(
        mesh_device=SimpleNamespace(shape=(4, 8)), users_row_sharded=True, switch_mode=lambda _: None
    )
    model.read_decode_feedback = MethodType(Model.read_decode_feedback, model)
    generator = Generator.__new__(Generator)
    generator.model = [model]
    generator.data_parallel = 1
    generator.mode = Mode.DECODE if reset_batch else Mode.PREFILL
    generator.model_capabilities = {"supports_async_decode": True}
    generator.trace_ids_decode = {True: {0: 1}}
    generator.trace_inputs_decode = {True: [[token_shards, position_shards, None, None]]}
    generator._slots_prefilled_since_decode = prefilled
    observed = []

    def decode(tokens, current_pos, **kwargs):
        observed.extend([tokens[0], current_pos[0]])
        return []

    generator._decode_forward_trace_text = decode
    generator.decode_forward(
        host_tokens,
        host_positions,
        enable_trace=True,
        defer_device_sampling=True,
        reset_batch=reset_batch,
        slot_remap=remap.tolist() if remap_slots else None,
    )
    torch.testing.assert_close(observed[0].reshape(-1), expected_tokens)
    torch.testing.assert_close(observed[1], expected_positions)
    assert not generator._slots_prefilled_since_decode
