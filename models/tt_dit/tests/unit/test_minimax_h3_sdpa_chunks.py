# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for `MiniMaxH3Attention._sdpa_program_config`'s chunk-size selection.

Pins the three paths apart, because they are chosen by two booleans and a lookup and a mix-up is
silent -- a wrong chunk size costs 20-25% of the whole SDPA op (the shipped 1x1 canvases measure
116.77 -> 94.01 ms, 159.43 -> 121.05 ms and 454.77 -> 351.57 ms between q=256 and q=384) without
changing a single number in the output. No device: the method only reads instance attributes and
builds host-side `ttnn.SDPAProgramConfig` / `ttnn.CoreCoord` objects.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.tt_dit.models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention

# The padded packed lengths the p150 server exposes: 960x544 x124, 1344x768 x73 (the shipped
# default) and 1344x768 x124. None of them is in `measured_sdpa_chunk_sizes`, whose keys are
# Galaxy PER-DEVICE lengths, so all three must come from the plain-windowed rule.
P150_SERVED_LENGTHS = (19328, 22464, 37760)


def _attn_stub():
    """The narrowest object `_sdpa_program_config` can run against."""
    return SimpleNamespace(
        _sdpa_program_configs={},
        measured_sdpa_chunk_sizes=MiniMaxH3Attention.measured_sdpa_chunk_sizes,
        plain_windowed_sdpa_chunk_sizes=MiniMaxH3Attention.plain_windowed_sdpa_chunk_sizes,
        full_grid=ttnn.CoreCoord(11, 10),
        sdpa_worker_grid=(10, 10),
    )


def _config(seq_local, *, ring, windowed):
    return MiniMaxH3Attention._sdpa_program_config(_attn_stub(), seq_local, ring=ring, windowed=windowed)


@pytest.mark.parametrize("seq_local", P150_SERVED_LENGTHS)
def test_served_1x1_lengths_take_the_measured_windowed_chunks(seq_local):
    cfg = _config(seq_local, ring=False, windowed=True)
    assert (cfg.q_chunk_size, cfg.k_chunk_size) == MiniMaxH3Attention.plain_windowed_sdpa_chunk_sizes
    # The whole grid, not the ring path's reserved-column grid.
    assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (11, 10)


def test_token_refiner_keeps_the_generic_rule():
    """The other non-ring caller passes no window and must not inherit the 1x1 attention's chunks."""
    cfg = _config(512, ring=False, windowed=False)
    assert (cfg.q_chunk_size, cfg.k_chunk_size) == (256, 512)


def test_ring_lengths_are_untouched():
    """Every `measured_sdpa_chunk_sizes` entry still wins over both generic rules."""
    for seq_local, expected in MiniMaxH3Attention.measured_sdpa_chunk_sizes.items():
        cfg = _config(seq_local, ring=True, windowed=False)
        assert (cfg.q_chunk_size, cfg.k_chunk_size) == expected


def test_short_windowed_sequence_clamps_to_the_sequence():
    """A window shorter than the preferred q must not ask for a chunk longer than the sequence."""
    cfg = _config(128, ring=False, windowed=True)
    assert (cfg.q_chunk_size, cfg.k_chunk_size) == (128, 128)


def test_windowed_still_caps_k_at_one_mask_tile_budget():
    """A `measured_sdpa_chunk_sizes` hit reached with a window keeps the k cap that bounds the mask CB."""
    cfg = _config(9184, ring=False, windowed=True)
    assert cfg.k_chunk_size == 256
