# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The distributed verify top-k's width rule (QWEN38_VERIFY_DTOPK) without a device."""

from __future__ import annotations

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module


def test_off_when_env_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN38_VERIFY_DTOPK", "0")
    assert qsa_module._verify_dtopk_width(65536, 8) == 0


def test_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN38_VERIFY_DTOPK", raising=False)
    assert qsa_module._verify_dtopk_width(65536, 8) == 8192


@pytest.mark.parametrize("blocks, width", [(8192, 1024), (16384, 2048), (32768, 4096), (65536, 8192)])
def test_width_is_the_per_device_slice(monkeypatch: pytest.MonkeyPatch, blocks: int, width: int) -> None:
    monkeypatch.setenv("QWEN38_VERIFY_DTOPK", "1")
    assert qsa_module._verify_dtopk_width(blocks, 8) == width


@pytest.mark.parametrize(
    "blocks, tp",
    [
        (65536 + 32, 8),  # not divisible by tp * tile
        (2048, 8),  # slice 256 < k
        (24576, 8),  # slice 3072 is not a power of two
        (262144, 8),  # slice 32768 > TOPK_CLASSIC_MAX_WIDTH
        (0, 8),
    ],
)
def test_width_zero_when_not_applicable(monkeypatch: pytest.MonkeyPatch, blocks: int, tp: int) -> None:
    monkeypatch.setenv("QWEN38_VERIFY_DTOPK", "1")
    assert qsa_module._verify_dtopk_width(blocks, tp) == 0


def test_slice_equal_to_k_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN38_VERIFY_DTOPK", "1")
    assert qsa_module._verify_dtopk_width(4096, 8) == 512


def test_candidates_must_fit_one_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN38_VERIFY_DTOPK", "1")
    assert qsa_module._verify_dtopk_width(65536 * 4, 64) == 0  # 64 * 512 > 16384 candidates
