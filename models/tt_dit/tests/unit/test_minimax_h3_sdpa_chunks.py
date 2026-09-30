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


def _attn_stub(*, is_sequence_parallel=True):
    """The narrowest object `_sdpa_program_config` can run against.

    `is_sequence_parallel` is what separates the DiT's attention from the token refiner's: the
    refiner is constructed `is_sequence_parallel=False` (token_refiner_minimax_h3.py), and BOTH pass
    a window, so it is the only thing that tells them apart.
    """
    return SimpleNamespace(
        _sdpa_program_configs={},
        measured_sdpa_chunk_sizes=MiniMaxH3Attention.measured_sdpa_chunk_sizes,
        plain_windowed_sdpa_chunk_sizes=MiniMaxH3Attention.plain_windowed_sdpa_chunk_sizes,
        is_sequence_parallel=is_sequence_parallel,
        full_grid=ttnn.CoreCoord(11, 10),
        sdpa_worker_grid=(10, 10),
    )


def _config(seq_local, *, ring, windowed, is_sequence_parallel=True):
    return MiniMaxH3Attention._sdpa_program_config(
        _attn_stub(is_sequence_parallel=is_sequence_parallel), seq_local, ring=ring, windowed=windowed
    )


@pytest.mark.parametrize("seq_local", P150_SERVED_LENGTHS)
def test_served_1x1_lengths_take_the_measured_windowed_chunks(seq_local):
    cfg = _config(seq_local, ring=False, windowed=True)
    assert (cfg.q_chunk_size, cfg.k_chunk_size) == MiniMaxH3Attention.plain_windowed_sdpa_chunk_sizes
    # The whole grid, not the ring path's reserved-column grid.
    assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (11, 10)


# The refiner's real presentation lengths: `MINIMAX_H3_REF2VA_PRESENTATION_RUNGS` pads to 1024 and
# above, and 512 is the short t2va case. All are above the generic q of 256, so all three would have
# silently taken (384, 256) under the first version of this rule.
REFINER_LENGTHS = (512, 1024, 5120)


@pytest.mark.parametrize("seq_local", REFINER_LENGTHS)
def test_token_refiner_keeps_the_generic_rule_even_though_it_passes_a_window(seq_local):
    """The refiner's REAL config, which the first version of this test did not exercise.

    It used to assert `windowed=False` for the refiner, on the strength of a comment saying the
    refiner passes no window. It does pass one -- `MiniMaxH3Pipeline._prompt_windows` fences off its
    padded text tail on every mesh -- so the assertion was testing the complement of the real
    configuration and the refiner was in fact inheriting the DiT's chunk sizes on every mesh,
    including ring meshes the commit claimed were untouched.
    """
    cfg = _config(seq_local, ring=False, windowed=True, is_sequence_parallel=False)
    assert (cfg.q_chunk_size, cfg.k_chunk_size) == (256, 256), "windowed still caps k at 256"


def test_a_non_windowed_non_ring_call_keeps_the_generic_rule():
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
