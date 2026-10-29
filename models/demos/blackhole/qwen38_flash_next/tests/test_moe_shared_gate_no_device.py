# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device gate: the composed (moe_post off) MoE combine gates a fused shared-expert pair before the add."""

from __future__ import annotations

from unittest import mock

import models.demos.blackhole.qwen38_flash_next.ttnn.moe as moe_module
from models.demos.blackhole.qwen38_flash_next.tests.test_production_moe_stage_fences import (
    _bare_moe,
    _FakeTensor,
    _forward_fixture,
    _install_forward_mocks,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import HIDDEN_SIZE, Qwen38TTNNSharedPartial


def _run_unfused_combine(monkeypatch, shared_partial):
    """Forward with moe_post off and the shared partial handed back as ``shared_partial``; returns the events."""

    module = _bare_moe()
    module.allreduce_links = 1
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)
    module._shared_partial = mock.Mock(return_value=shared_partial)
    gated = _FakeTensor("gated", tensors["shared_partial"].shape)
    monkeypatch.setattr(
        moe_module.fused.shared_expert,
        "gate_partial",
        lambda partial, sigmoid, rows: events.append(("gate", partial.name, sigmoid.name, rows)) or gated,
    )
    module.forward(tensors["hidden_sharded"], tensors["packed_w0_w1"], tensors["packed_w2"])
    return events


def test_unfused_post_gates_the_shared_partial_pair_before_the_add(monkeypatch) -> None:
    partial = _FakeTensor("partial", (1, 1, 1, HIDDEN_SIZE))
    pair = Qwen38TTNNSharedPartial(partial, _FakeTensor("sigmoid", (1, 1, 32, 32)))
    events = _run_unfused_combine(monkeypatch, pair)

    gate = events.index(("gate", "partial", "sigmoid", 1))
    add = [event for event in events if event[0] == "add"]
    assert add == [("add", "routed_partial", "gated")]
    assert gate < events.index(add[0])
    # the pair's two tensors and the routed partial are released; the gated tensor is released after the add
    assert ("deallocate", "partial") in events and ("deallocate", "sigmoid") in events
    assert ("deallocate", "gated") in events and ("deallocate", "routed_partial") in events


def test_unfused_post_passes_a_plain_shared_partial_through_unchanged(monkeypatch) -> None:
    plain = _FakeTensor("plain_shared", (1, 1, 1, HIDDEN_SIZE))
    events = _run_unfused_combine(monkeypatch, plain)

    assert not [event for event in events if event[0] == "gate"]
    assert [event for event in events if event[0] == "add"] == [("add", "routed_partial", "plain_shared")]
