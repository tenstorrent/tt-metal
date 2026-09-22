# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CSA lightning indexer and the sparse attend it drives.

Full-model decode hides a wrong index list inside one logit tensor. These tests
check the three contracts that tensor has to satisfy:

* ``DeepSeekV4Indexer.select`` scores ``sum_h ReLU(q_h · k) * w_h`` and keeps
  the top-k inside the closed-window prefix (a key past ``valid_length`` must
  not win, even when its score is the largest).
* ``index_list`` is ``[0, window)`` followed by ``window + selected``, and a
  sentinel stays a sentinel. That is the combined KV axis ``[sliding | compressed]``.
* ``sparse_sdpa`` on that index list matches a torch gather of the same rows,
  including the attention sink.

Run (ttnn venv)::

    pytest -s models/experimental/deepseek_v4_flash/tests/test_csa_indexer.py
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tt.attention_csa import INDEX_SENTINEL, DeepSeekV4Indexer

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

PCC_THRESHOLD = 0.99
TILE = 32


def _bare_indexer(device, *, topk: int, window: int = 128, compress_rate: int = 4) -> DeepSeekV4Indexer:
    """The fields ``select`` / ``index_list`` / ``_closed_windows`` read. No weights."""
    indexer = DeepSeekV4Indexer.__new__(DeepSeekV4Indexer)
    indexer.device = device
    indexer.index_topk = topk
    indexer.sliding_window = window
    indexer.compress_rate = compress_rate
    indexer._ag_sub_device_id = None
    indexer._window_ids = None
    return indexer


def _tile(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _rm(t: torch.Tensor, device, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _as_u32(t: torch.Tensor) -> torch.Tensor:
    """Host view of a uint32 device tensor. ``to_torch`` may return the sign bit as int32."""
    return t.reshape(-1).to(torch.int64) & 0xFFFFFFFF


def _reference_scores(query_row: torch.Tensor, keys: torch.Tensor, weights_row: torch.Tensor) -> torch.Tensor:
    """``sum_h ReLU(q_h · k_t) * w_h`` over every key. ``query_row`` is ``[Hi, D]``."""
    dots = torch.relu(query_row.float() @ keys.float().T)
    return (dots * weights_row.float()[:, None]).sum(dim=0)


def test_indexer_select_respects_closed_windows(device) -> None:
    """The largest key sits past the closed-window count and must not be selected."""
    torch.manual_seed(0)
    heads, dim, length, topk, valid = 32, 128, 64, 16, 24
    query = torch.zeros(1, heads, TILE, dim)
    query[0, :, TILE - 1, 0] = 1.0
    weights = torch.zeros(1, 1, TILE, heads)
    weights[0, 0, TILE - 1, :] = 1.0
    keys = torch.zeros(1, 1, length, dim)
    keys[0, 0, :, 0] = torch.arange(1, length + 1, dtype=torch.float32)
    keys[0, 0, 50, 0] = 1000.0

    scores = _reference_scores(query[0, :, TILE - 1], keys[0, 0], weights[0, 0, TILE - 1])
    scores[valid:] = float("-inf")
    expected = torch.topk(scores, topk).indices

    indexer = _bare_indexer(device, topk=topk)
    valid_length = _rm(torch.tensor([valid], dtype=torch.int32), device, ttnn.uint32)
    selected = indexer.select(
        _tile(query, device), _tile(keys, device), _tile(weights, device), valid_length_tensor=valid_length
    )
    row = ttnn.slice(selected, [0, 0, TILE - 1, 0], [1, 1, TILE, topk])
    got = _as_u32(ttnn.to_torch(row))

    logger.info(f"indexer select expected {expected.tolist()} got {got.tolist()}")
    assert got.tolist() == expected.tolist()
    assert 50 not in got.tolist()


def test_closed_windows_from_position(device) -> None:
    """``(pos + 1) // compress_rate`` is the number of keys the indexer may search."""
    indexer = _bare_indexer(device, topk=16, compress_rate=4)
    pos = _rm(torch.tensor([10240], dtype=torch.int32), device, ttnn.int32)
    got = int(ttnn.to_torch(indexer._closed_windows(pos)).reshape(-1)[0])
    assert got == (10240 + 1) // 4


def test_index_list_offsets_compressed_entries(device) -> None:
    """Sliding slots stay ``0..window``; compressed picks move by ``window``; sentinels do not."""
    window, topk = 128, 16
    selected = torch.zeros(1, 1, 1, topk, dtype=torch.uint32)
    selected[0, 0, 0, :4] = torch.tensor([0, 3, 7, 10], dtype=torch.uint32)
    selected[0, 0, 0, 4] = INDEX_SENTINEL
    indexer = _bare_indexer(device, topk=topk, window=window)
    got = _as_u32(ttnn.to_torch(indexer.index_list(_rm(selected, device, ttnn.uint32))))

    prefix = torch.arange(window, dtype=torch.int64)
    body = selected.reshape(-1).to(torch.int64).clone()
    real = body != INDEX_SENTINEL
    body[real] = body[real] + window
    expected = torch.cat([prefix, body])
    assert got.tolist() == expected.tolist()
    assert int(got[window + 4]) == INDEX_SENTINEL


def _sparse_reference(query, cache, indices, scale, sink):
    """Gather ``indices`` from ``cache`` and softmax with the raw sink as an extra logit.

    The device multiplies both the gathered scores and the sink tensor by ``scale``.
    The sink passed into the op is ``raw_sink / scale``, so the extra logit is the
    raw sink. A sentinel is not a cache row: replace it before the gather, then
    mask that score to -inf. ``clamp(min=0)`` leaves ``0xFFFFFFFF`` in range and
    ``index_select`` raises.
    """
    heads, dim = query.shape[1], query.shape[-1]
    flat = _as_u32(indices)
    masked = flat == INDEX_SENTINEL
    safe = torch.where(masked, torch.zeros_like(flat), flat)
    picked = cache[0, 0].index_select(0, safe)
    scores = (query[0, :, 0].float() @ picked.float().T) * scale
    scores[:, masked] = float("-inf")
    sink_logit = sink.float().reshape(heads, 1)
    probs = torch.softmax(torch.cat([scores, sink_logit], dim=-1), dim=-1)
    return (probs[:, :-1] @ picked.float()).reshape(1, heads, 1, dim)


def test_csa_sparse_attend_matches_reference(device) -> None:
    """Attend the sliding ring plus ``window + selected`` compressed rows."""
    torch.manual_seed(1)
    heads, dim, window, compressed, topk = 32, 128, 128, 32, 32
    length = window + compressed
    query = torch.randn(1, heads, 1, dim) * 0.1
    cache = torch.randn(1, 1, length, dim) * 0.1
    sink = torch.randn(heads) * 0.5
    scale = dim**-0.5
    chosen = torch.arange(0, compressed, compressed // topk)[:topk]
    indices = torch.cat([torch.arange(window), window + chosen]).reshape(1, 1, 1, window + topk).to(torch.uint32)
    indices[0, 0, 0, window + topk - 1] = INDEX_SENTINEL

    reference = _sparse_reference(query, cache, indices, scale, sink)
    got = ttnn.to_torch(
        ttnn.transformer.sparse_sdpa(
            _rm(query, device, ttnn.bfloat16),
            _rm(cache, device, ttnn.bfloat16),
            _rm(indices, device, ttnn.uint32),
            v_dim=dim,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=scale,
            k_chunk_size=32,
            attention_sink=_rm((sink / scale).reshape(1, 1, 1, heads), device, ttnn.bfloat16),
        )
    ).float()

    passing, message = comp_pcc(reference, got, pcc=PCC_THRESHOLD)
    logger.info(f"csa sparse attend {comp_allclose(reference, got)}")
    logger.info(f"csa sparse attend PCC: {message}")
    assert passing, f"CSA sparse attend PCC < {PCC_THRESHOLD}: {message}"
