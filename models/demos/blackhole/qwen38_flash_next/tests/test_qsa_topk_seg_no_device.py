# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_TOPK_SEG=0 (the default is now 1) makes qsa_topk_block_ids call topk_large_indices exactly as the old call sites did."""


import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module


class _Scores:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture
def spy(monkeypatch):
    calls = []

    def fake(tensor, *args, **kwargs):
        calls.append((tensor, args, kwargs))
        return "ids"

    monkeypatch.setattr(ttnn.experimental, "topk_large_indices", fake)
    return calls


@pytest.mark.parametrize("env", ["0"])  # unset is now on (test_seg_default_is_on)
@pytest.mark.parametrize("shape", [(1, 1, 1, 65536), (1, 1, 4, 65536), (1, 1, 8, 16384), (1, 1, 8, 8192)])
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"valid_length": 100}, {"valid_length_tensor": "t", "valid_length_offset": 32}],
)
def test_seg_off_is_the_single_call(monkeypatch, spy, env, shape, kwargs):
    if env is None:
        monkeypatch.delenv("QWEN38_TOPK_SEG", raising=False)
    else:
        monkeypatch.setenv("QWEN38_TOPK_SEG", env)
    scores = _Scores(shape)
    assert qsa_module.qsa_topk_block_ids(scores, k=qsa_module.BLOCK_TOPK, **kwargs) == "ids"
    assert spy == [(scores, (), {"k": qsa_module.BLOCK_TOPK, **kwargs})]


def test_seg_default_is_on(monkeypatch):
    monkeypatch.delenv("QWEN38_TOPK_SEG", raising=False)
    assert qsa_module._topk_seg_width(65536, 512) == 8192
    monkeypatch.setenv("QWEN38_TOPK_SEG", "0")
    assert qsa_module._topk_seg_width(65536, 512) == 0


@pytest.mark.parametrize("width", [8192, 16384])
def test_seg_on_narrow_rows_are_the_single_call(monkeypatch, spy, width):
    monkeypatch.setenv("QWEN38_TOPK_SEG", "1")
    scores = _Scores((1, 1, 8, width))
    qsa_module.qsa_topk_block_ids(scores, k=qsa_module.BLOCK_TOPK)
    assert spy == [(scores, (), {"k": qsa_module.BLOCK_TOPK})]


def test_seg_width_depends_on_width_only(monkeypatch):
    monkeypatch.setenv("QWEN38_TOPK_SEG", "1")
    assert qsa_module._topk_seg_width(65536, 512) == 8192
    assert qsa_module._topk_seg_width(32768, 512) == 8192
    assert qsa_module._topk_seg_width(20000, 512) == 0  # not a multiple of the segment width
    assert qsa_module._topk_seg_width(16384, 512) == 0
    assert qsa_module._topk_seg_width(65536, 1024) == 0
    monkeypatch.setenv("QWEN38_TOPK_SEG_W", "4096")
    assert qsa_module._topk_seg_width(65536, 512) == 4096
    monkeypatch.setenv("QWEN38_TOPK_SEG", "0")
    assert qsa_module._topk_seg_width(65536, 512) == 0
