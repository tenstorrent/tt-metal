# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only unit tests for the LTX LoRA adapter loader's module resolution.

No device, no checkpoint: these pin the loader's warn-and-skip contract for
targets that do not exist on the transformer variant being loaded. A LoRA file
authored against a deeper variant references block indices past this variant's
depth; resolution must return "absent" so the caller skips, not raise IndexError.
"""

from types import SimpleNamespace

from models.tt_dit.experimental.lora.ltx_adapter_loader import _get_attn, _resolve_singleton


def _fake_transformer(num_blocks: int):
    """A transformer stub with ``num_blocks`` blocks, each carrying an ``attn1``."""
    blocks = [SimpleNamespace(attn1=SimpleNamespace(name=f"attn1_{i}")) for i in range(num_blocks)]
    return SimpleNamespace(transformer_blocks=blocks)


def test_get_attn_out_of_range_returns_none():
    tf = _fake_transformer(2)
    assert _get_attn(tf, 5, "attn1") is None
    assert _get_attn(tf, -1, "attn1") is None


def test_get_attn_in_range_resolves():
    tf = _fake_transformer(2)
    assert _get_attn(tf, 1, "attn1") is tf.transformer_blocks[1].attn1
    # Missing attribute on an in-range block still degrades to None, not AttributeError.
    assert _get_attn(tf, 1, "attn2") is None


def test_resolve_singleton_out_of_range_returns_none_triple():
    tf = _fake_transformer(2)
    assert _resolve_singleton(tf, 9, "attn1", "to_out") == (None, None, None)
    assert _resolve_singleton(tf, -1, "attn1", "ff1") == (None, None, None)
