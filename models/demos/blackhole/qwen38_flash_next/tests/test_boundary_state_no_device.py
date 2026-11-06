# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Boundary state piece table, offsets and preconditions with fake states (no device)."""

from types import SimpleNamespace

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import pool_blocks


def tensor(shape, dtype):
    return SimpleNamespace(padded_shape=shape, dtype=dtype)


def gdn(phase=0):
    return SimpleNamespace(
        recurrent=tensor((1, 6, 128, 128), ttnn.float32),
        conv=tuple(tensor((1, 1, 32, 1280), ttnn.bfloat16) for _ in range(4)),
        conv_phase=phase,
    )


def ple():
    return SimpleNamespace(conv=tuple(tensor((1, 1, 32, 320), ttnn.bfloat16) for _ in range(9)), token_context="ctx")


class Position:
    def __init__(self):
        self.calls = []

    def reset(self, position, rope_shift):
        self.calls.append((position, rope_shift))


def make_state(phase=0):
    layers = (
        SimpleNamespace(attention=gdn(phase), ple=None),
        SimpleNamespace(attention=gdn(phase), ple=ple()),
        SimpleNamespace(attention=gdn(phase), ple=None),
        SimpleNamespace(attention=SimpleNamespace(kv_staging="s", raw_key_ring="r"), ple=None),
    )
    return SimpleNamespace(layers=layers, position=Position())


def test_piece_table(expect_error):
    state = make_state()
    plain = model_module.boundary_state_pieces(state, 0)
    assert [offset for _, offset in plain] == [0, 393_216, 475_136, 557_056, 638_976]
    first = model_module.boundary_state_pieces(state, 1)
    assert [offset for _, offset in first[5:]] == [720_896 + j * 20_480 for j in range(9)]
    assert first[-1][1] + 20_480 == 905_216 <= 1024 * 1024
    with expect_error(ValueError, "not a GDN layer"):
        model_module.boundary_state_pieces(state, 3)


def test_shape_and_block_size_mismatch_fail_loudly(expect_error):
    state = make_state()
    state.layers[0].attention.recurrent = tensor((1, 12, 128, 128), ttnn.float32)
    with expect_error(RuntimeError, "393216"):
        model_module.boundary_state_pieces(state, 0)
    state = make_state()
    with expect_error(RuntimeError, "past the"):
        model_module.boundary_state_pieces(state, 1, block_size=512)


def patch_copies(monkeypatch):
    calls = []
    monkeypatch.setattr(pool_blocks, "state_to_block", lambda *a, **k: calls.append(("save", a[2], k["byte_offset"])))
    monkeypatch.setattr(pool_blocks, "block_to_state", lambda *a, **k: calls.append(("load", a[2], k["byte_offset"])))
    return calls


def test_save_copies_every_piece_and_checks_preconditions(monkeypatch, expect_error):
    calls = patch_copies(monkeypatch)
    state = make_state()
    blocks = {0: ("pool", 3), 1: ("pool", 4), 2: ("pool", 5)}
    model_module.save_boundary_state(state, blocks, block_size=1024, position=2048)
    assert len(calls) == 5 + 14 + 5
    assert [c for c in calls if c[1] == 4][0] == ("save", 4, 0)
    calls.clear()
    with expect_error(ValueError, "multiple"):
        model_module.save_boundary_state(state, blocks, block_size=1024, position=2052)
    with expect_error(ValueError, "exactly the GDN"):
        model_module.save_boundary_state(state, {0: ("pool", 3)}, block_size=1024, position=2048)
    assert calls == []
    # The traced decode step leaves the host phase behind the device ring, so a lagging phase still saves.
    state.layers[2].attention.conv_phase = 3
    model_module.save_boundary_state(state, blocks, block_size=1024, position=2048)
    assert len(calls) == 5 + 14 + 5


def test_restore_resets_phase_position_context_and_qsa(monkeypatch):
    calls = patch_copies(monkeypatch)
    state = make_state(phase=3)
    zeroed = []
    blocks = {0: ("pool", 3), 1: ("pool", 4), 2: ("pool", 5)}
    model_module.restore_boundary_state(
        state, blocks, block_size=1024, position=4096, zero_qsa=lambda i, s: zeroed.append(i)
    )
    assert len(calls) == 24 and all(c[0] == "load" for c in calls)
    assert [layer.attention.conv_phase for layer in state.layers[:3]] == [0, 0, 0]
    assert state.layers[1].ple.token_context is None
    assert zeroed == [3]
    assert state.position.calls == [(4096, 0)]
