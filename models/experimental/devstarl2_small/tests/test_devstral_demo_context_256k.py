# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Context-length policy for Devstral TT demos (256K window on Blackhole).

These tests do **not** allocate a 256K-token prompt or run the model; they lock in the
budgeting rules from ``default_devstral_demo_max_seq_len`` / ``devstral_tt_kv_cache_max_seq_len``
so demos can take ``ModelArgs.max_seq_len`` up to **256_000** when ``--max-seq-len`` is omitted
on Blackhole (see ``multimodal_demo_helpers.default_devstral_demo_max_seq_len``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import ttnn

from models.experimental.devstarl2_small.devstral_utils import (
    DEVSTRAL_DEMO_BLACKHOLE_DEFAULT_MAX_SEQ_LEN,
    default_devstral_demo_max_seq_len,
    devstral_tt_kv_cache_max_seq_len,
)
from models.experimental.devstarl2_small.devstral_utils.multimodal_demo_helpers import tt_prefill_target_seqlen


@pytest.fixture
def fake_mesh_device() -> MagicMock:
    """Stand-in for ``ttnn.MeshDevice`` (only ``is_blackhole`` is consulted)."""
    return MagicMock(name="mesh_device")


def test_blackhole_context_floor_is_256k() -> None:
    """Contract: Blackhole demo default RoPE cap is at least 256K tokens."""
    assert DEVSTRAL_DEMO_BLACKHOLE_DEFAULT_MAX_SEQ_LEN == 256_000


def test_default_max_seq_len_blackhole_small_prompt_uses_256k_floor(fake_mesh_device, monkeypatch) -> None:
    """On Blackhole, a short prompt still gets ``max_seq_len`` = 256K (RoPE / allocation floor)."""
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda _mesh: True)
    need = 3_000
    assert default_devstral_demo_max_seq_len(fake_mesh_device, need) == 256_000


def test_default_max_seq_len_blackhole_large_prompt_exceeds_floor(fake_mesh_device, monkeypatch) -> None:
    """On Blackhole, if ``need`` exceeds 256K, the larger value wins (demo still fits run)."""
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda _mesh: True)
    need = 300_000
    assert default_devstral_demo_max_seq_len(fake_mesh_device, need) == 300_000


def test_default_max_seq_len_non_blackhole_uses_4096_floor(fake_mesh_device, monkeypatch) -> None:
    """Wormhole / other arches: floor 4096 to limit default KV/RoPE allocation."""
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda _mesh: False)
    assert default_devstral_demo_max_seq_len(fake_mesh_device, 100) == 4_096


def test_default_max_seq_len_non_blackhole_scales_with_need(fake_mesh_device, monkeypatch) -> None:
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda _mesh: False)
    need = 50_000
    assert default_devstral_demo_max_seq_len(fake_mesh_device, need) == 50_000


def test_tt_prefill_target_seqlen_accepts_256k_for_typical_kv_layout() -> None:
    """
    A 256K active length satisfies TT prefill padding (128 / KV tile / WO chunk) for a typical layout.

    Devstral-2 Small uses 8 KV heads on a 1-wide mesh in many demos (``cluster_shape[1] == 1``).
    """
    n_kv_heads = 8
    mesh_cols = 1
    target = tt_prefill_target_seqlen(256_000, n_kv_heads, mesh_cols)
    assert target == 256_000


def test_kv_cache_seq_dim_clamped_to_model_max_seq_len() -> None:
    """``devstral_tt_kv_cache_max_seq_len`` never exceeds ``model_args.max_seq_len``."""
    ma = MagicMock()
    ma.cluster_shape = (1, 1)
    ma.n_kv_heads = 8
    ma.max_seq_len = 256_000
    assert devstral_tt_kv_cache_max_seq_len(ma, 256_000) == 256_000

    ma.max_seq_len = 8_192
    assert devstral_tt_kv_cache_max_seq_len(ma, 256_000) <= 8_192
