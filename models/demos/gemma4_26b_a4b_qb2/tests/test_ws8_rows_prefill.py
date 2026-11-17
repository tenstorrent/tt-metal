# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WS8: prefix-cache continuation as same-user decode rows (perf/PLAN.md).

The rows continuation must (1) give last-position logits that match the token-at-a-time continuation it replaces
(the same decode arithmetic one row at a time) and a cold prefill of the whole prompt, (2) write the suffix K/V so a
following decode sees the same context, and (3) leave the cached prefix and other requests' pages untouched."""

import pytest
import torch

from models.demos.gemma4_26b_a4b_qb2.tests.test_device_numerics import snapshot
from models.demos.gemma4_26b_a4b_qb2.tt import generator as generator_module
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.fixture
def generator(qb2_mesh):
    torch.manual_seed(42)
    torch.set_num_threads(8)
    generator = Gemma4Generator(qb2_mesh, max_seq_len=2048, layer_indices=(0, 5))
    try:
        yield generator
    finally:
        generator._release_trace()


def continuation(generator, monkeypatch, tokens, start, rows):
    monkeypatch.setattr(generator_module, "WS8_ROWS", rows)
    model = generator.model
    end = tokens.shape[1]
    caches, table = model.allocate_cache(slots=2, context=end + 1)
    active = table[:1]
    generator.prefill_forward(tokens[:, :start], page_table=active, kv_cache=caches, prompt_lens=[start])
    before = snapshot(caches)
    logits = generator.prefill_forward(tokens, page_table=active, kv_cache=caches, prompt_lens=[end], start_pos=[start])
    logits = generator._read_logits(logits).reshape(-1).clone()
    after = snapshot(caches)
    # One decode step after the continuation reads the suffix K/V it wrote.
    nxt = generator.decode_forward(
        tokens[:, end - 1 : end] * 0 + int(logits.argmax()),
        torch.tensor([end]),
        page_table=active,
        kv_cache=caches,
        return_logits=True,
    )
    nxt = generator._read_logits(nxt)[..., :1, :].reshape(-1).clone()
    generator._release_trace()
    for old_layer, new_layer in zip(before, after):
        for old, new in zip(old_layer, new_layer):
            assert torch.equal(old[table[1].long()], new[table[1].long()]), "Neighbor request pages changed"
            heads, width = old.shape[1], old.shape[-1]
            old_prefix = old[table[0].long()].permute(1, 0, 2, 3).reshape(heads, -1, width)[:, :start]
            new_prefix = new[table[0].long()].permute(1, 0, 2, 3).reshape(heads, -1, width)[:, :start]
            assert torch.equal(old_prefix, new_prefix), "Cached prefix changed"
    return logits, nxt


@pytest.mark.parametrize(
    "start,suffix",
    [(32, 9), (37, 40), (100, 64), (1100, 33)],
    ids=["page_aligned", "partial_page_two_chunks", "two_full_chunks", "past_sliding_window"],
)
def test_rows_continuation(generator, monkeypatch, start, suffix):
    tokens = torch.randint(10, 1000, (1, start + suffix))
    serial, serial_next = continuation(generator, monkeypatch, tokens, start, rows=False)
    count = generator.counters.get("ws8_rows_prefills", 0)
    rows, rows_next = continuation(generator, monkeypatch, tokens, start, rows=True)
    assert generator.counters.get("ws8_rows_prefills", 0) == count + 1, "rows path not taken"
    cold = cold_prefill(generator, tokens)
    print(
        f"WS8 start={start} suffix={suffix} last-logit PCC rows/serial {pcc(rows, serial):.5f} "
        f"rows/cold {pcc(rows, cold):.5f} serial/cold {pcc(serial, cold):.5f}, next rows/serial "
        f"{pcc(rows_next, serial_next):.5f}, top-1 rows {int(rows.argmax())} serial {int(serial.argmax())} "
        f"cold {int(cold.argmax())}"
    )
    # The rows path runs the batched (expert-parallel) layer, the serial path the batch-1 layer: equal to WS1's
    # batched-vs-serial tolerance, not bit-exact. Against a cold prefill it stays within the serial path's own
    # spread (2-layer logits on random tokens are flat: serial vs cold is 0.977-0.989 here), same top-1.
    assert pcc(rows, serial) >= 0.99
    assert pcc(rows, cold) >= pcc(serial, cold) - 0.01
    assert int(rows.argmax()) == int(serial.argmax())
    assert pcc(rows_next, serial_next) >= 0.999


def cold_prefill(generator, tokens):
    end = tokens.shape[1]
    caches, table = generator.model.allocate_cache(slots=1, context=end + 1)
    logits = generator.prefill_forward(tokens, page_table=table[:1], kv_cache=caches, prompt_lens=[end])
    out = generator._read_logits(logits).reshape(-1).clone()
    generator._release_trace()
    return out
