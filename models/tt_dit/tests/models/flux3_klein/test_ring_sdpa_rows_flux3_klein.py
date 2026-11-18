# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only check of the ring SDPA chunks Flux3KleinAttention picks at every attention length of the served layouts
(each size, reference and text bucket: image, reference and text early blocks, single stage). No device.

SP4 x TP2 (DP4 groups) has no key in the shared ring SDPA table, whose default there is a 256 k chunk. Every DP4 length
must pick a 512 k chunk ((224, 512) at the 4096 image / single-stage rows), and SP8 x TP4 must pick what it picked before
SP4 x TP2 had its own chunks: Klein's table where it has the rows, else the shared table with q capped at 128.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ....blocks import attention_opt
from ....models.transformers.flux3_klein.transformer_block_flux3_klein import (
    RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
    SP4_TP2_RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
    Flux3KleinAttention,
)
from ....models.transformers.flux3_klein.transformer_flux3_klein import Flux3KleinSequenceLayout
from ....pipelines.flux3_klein.packing import TEXT_BUCKETS, TEXT_BUCKETS_BY_LAYOUT
from ....pipelines.flux3_klein.pipeline_flux3_klein import REFERENCE_TOKEN_BUCKETS, SIZES

SP8_ROWS = {
    2560: (160, 512),
    2688: (160, 512),
    2816: (160, 512),
    3072: (192, 512),
    8192: (224, 512),
    8320: (224, 512),
    8960: (160, 640),
}


def _lengths(sp: int, size: int) -> set[int]:
    """Every ring SDPA length (all devices) of a size's layouts."""
    lengths = set()
    for ref in (0, *REFERENCE_TOKEN_BUCKETS):
        for text in TEXT_BUCKETS_BY_LAYOUT.get((size, ref), TEXT_BUCKETS):
            layout = Flux3KleinSequenceLayout(sp, {"image": (size // 16) ** 2, "image_cond": ref, "text": text}, 1, ref)
            lengths |= {layout.padded[m] for m in layout.modes} | {layout.total_padded}
    return lengths


def _chunks(sp: int, tp: int, length: int) -> tuple[int, int]:
    attention = Flux3KleinAttention.__new__(Flux3KleinAttention)
    for name, value in {
        "parallel_config": SimpleNamespace(
            sequence_parallel=SimpleNamespace(factor=sp), tensor_parallel=SimpleNamespace(factor=tp)
        ),
        "ring_sdpa_program_config": {},
        "ring_sdpa_q_chunk_max": 128,
        "ring_sdpa_k_chunk_max": None,
        "ring_sdpa_exp_approx": True,
        "get_ring_sdpa_core_grid": lambda n: attention_opt.ttnn.CoreCoord(12, 9),
    }.items():
        object.__setattr__(attention, name, value)
    config = attention.get_ring_sdpa_program_config(length)
    return config.q_chunk_size, config.k_chunk_size


@pytest.fixture(autouse=True)
def _blackhole(monkeypatch):
    monkeypatch.setattr(attention_opt, "is_blackhole", lambda: True)


@pytest.mark.parametrize("size", SIZES)
def test_sp8_tp4_picks_are_unchanged(size):
    assert RING_SDPA_CHUNKS_BY_LOCAL_ROWS == SP8_ROWS
    shared = attention_opt.Attention.ring_sdpa_chunk_size_map[(True, 8, 4)]
    for n in _lengths(8, size):
        q, k = shared.get(n, shared[-1])
        assert _chunks(8, 4, n) == SP8_ROWS.get(n // 8, (min(q, 128), k)), n


@pytest.mark.parametrize("size", SIZES)
def test_dp4_picks_a_512_k_chunk(size):
    for n in _lengths(4, size):
        assert _chunks(4, 2, n)[1] == 512, n
        if size == 4096 and n // 4 in SP4_TP2_RING_SDPA_CHUNKS_BY_LOCAL_ROWS:
            assert _chunks(4, 2, n) == (224, 512), n


def test_dp4_4096_image_and_single_stage_rows_are_listed():
    rows = set()
    for ref in (0, *REFERENCE_TOKEN_BUCKETS):
        for text in TEXT_BUCKETS_BY_LAYOUT.get((4096, ref), TEXT_BUCKETS):
            layout = Flux3KleinSequenceLayout(4, {"image": 256 * 256, "image_cond": ref, "text": text}, 1, ref)
            rows |= {layout.rows["image"], layout.total_padded // 4}
    assert rows == set(SP4_TP2_RING_SDPA_CHUNKS_BY_LOCAL_ROWS)
