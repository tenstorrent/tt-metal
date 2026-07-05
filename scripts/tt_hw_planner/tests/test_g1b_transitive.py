# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from scripts.tt_hw_planner.commands.emit_e2e import _check_hf_fallback


def test_torch_compute_in_helper_reached_from_hot_is_flagged():
    src = """
import torch
class M:
    def __call__(self, x):
        return self._attn(x)

    def _attn(self, x):
        return torch.softmax(x, dim=-1)
"""
    hits = _check_hf_fallback(src)
    assert hits, "expected G1b to flag torch.softmax hidden in _attn helper"
    joined = " | ".join(hits)
    assert "softmax" in joined
    assert "via __call__" in joined or "[via " in joined


def test_top_level_hot_only_still_flagged():
    src = """
import torch
class M:
    def __call__(self, x):
        return torch.matmul(x, x)
"""
    hits = _check_hf_fallback(src)
    assert hits
    assert "matmul" in " | ".join(hits)


def test_pure_ttnn_stub_passes():
    src = """
import ttnn
class M:
    def __call__(self, x):
        return self._layer(x)

    def _layer(self, x):
        return ttnn.linear(x, self.w)
"""
    hits = _check_hf_fallback(src)
    assert hits == [], f"expected clean stub to pass; got: {hits}"


def test_nested_helper_chain_traversed():
    src = """
import torch
class M:
    def __call__(self, x):
        return self._a(x)

    def _a(self, x):
        return self._b(x)

    def _b(self, x):
        return torch.softmax(x, dim=-1)
"""
    hits = _check_hf_fallback(src)
    assert hits, "expected transitivity across two hops (__call__ -> _a -> _b)"
    assert "softmax" in " | ".join(hits)


def test_setup_named_helper_is_waived_even_when_reached_from_hot():
    src = """
import torch
class M:
    def __call__(self, x):
        return self.encode_trace_setup(x)

    def encode_trace_setup(self, x):
        return torch.softmax(x, dim=-1)
"""
    hits = _check_hf_fallback(src)
    assert hits == [], f"SETUP-named helpers keep their waiver even when reached " f"transitively; got hits: {hits}"


def test_non_hot_top_level_not_seeded():
    src = """
import torch
class M:
    def _helper(self, x):
        return torch.softmax(x, dim=-1)
"""
    hits = _check_hf_fallback(src)
    assert hits == [], f"_helper is not HOT and no HOT caller — should not seed; got: {hits}"
