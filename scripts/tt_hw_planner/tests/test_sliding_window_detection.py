# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sliding-window attention is decided by the window the config declares.

`layer_types` used to be ORed into the predicate. It is a list of each layer's
attention kind, so it is non-empty for ANY config that enumerates its layers --
its presence says nothing about sliding. FLUX.2-klein's text_encoder declares 36
full-attention layers and no window, and was still reported as needing
sliding-window attention, which marked that building block PARTIAL and blocked the
bring-up.

The file's other predicates check FEATURE-SPECIFIC fields (`kv_lora_rank` for MLA,
`cross_attention_layers` for cross-attention) whose presence really does mean the
feature is in use. `sliding_window` is that field here; `layer_types` is not.
"""

from __future__ import annotations

from scripts.tt_hw_planner.compatibility import _is_sliding


def test_declared_window_is_sliding() -> None:
    assert _is_sliding({"sliding_window": 4096})


def test_no_window_is_not_sliding() -> None:
    assert not _is_sliding({"sliding_window": None})
    assert not _is_sliding({})


def test_zero_window_is_not_sliding() -> None:
    assert not _is_sliding({"sliding_window": 0})


def test_enumerated_layer_types_alone_is_not_sliding() -> None:
    """The regression: a config that lists its layers is not thereby sliding."""
    cfg = {"sliding_window": None, "layer_types": ["full_attention"] * 36}
    assert not _is_sliding(cfg), "listing layer kinds must not imply a sliding window"


def test_window_still_wins_when_layers_are_enumerated() -> None:
    cfg = {"sliding_window": 1024, "layer_types": ["full_attention", "sliding_attention"]}
    assert _is_sliding(cfg)


def test_nested_text_config_is_honoured() -> None:
    """Multimodal configs carry the text fields one level down."""
    assert _is_sliding({"text_config": {"sliding_window": 512}})
    assert not _is_sliding({"text_config": {"layer_types": ["full_attention"] * 8}})
