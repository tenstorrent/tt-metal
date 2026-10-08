# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks (no device) for the Qwen ring SDPA q_chunk picker and the ring program config built
from SDPA_PROGCFG. Picks were measured (ring SDPA micro-benchmark) only up to 40960; above that the
picker returns the default."""

import pytest

import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import (
    RING_SDPA_MAX_MEASURED_SEQ_LEN,
    qwen_ring_sdpa_program_config,
    ring_sdpa_balanced_q_chunk,
)

N_LOCAL_HEADS = 8  # Qwen3-32B: 64 q heads / 8 kv heads per device row
NUM_CORES = 70  # 7 x 10 SDPA grid


def base_cfg():
    # SDPA_PROGCFG(seqlen > 4096, chunk_start_idx=0) on Wormhole
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(7, 10), exp_approx_mode=False, q_chunk_size=256, k_chunk_size=512
    )


@pytest.mark.parametrize(
    "seq_len, q_chunk",
    [
        (8192, 64),
        (16384, 128),
        (24576, 192),
        (32768, 256),
        (40960, 320),
        # above the measured range: default, although the busiest-core model would pick otherwise
        (49152, 256),
        (65536, 256),
        (81920, 256),
        (131072, 256),
    ],
)
def test_picker_buckets(seq_len, q_chunk):
    assert ring_sdpa_balanced_q_chunk(seq_len, n_local_heads=N_LOCAL_HEADS, num_cores=NUM_CORES) == q_chunk


def test_picker_default_above_measured_range():
    assert RING_SDPA_MAX_MEASURED_SEQ_LEN == 40960
    for seq_len in range(RING_SDPA_MAX_MEASURED_SEQ_LEN + 2048, 262145, 2048):
        assert ring_sdpa_balanced_q_chunk(seq_len, N_LOCAL_HEADS, NUM_CORES) == 256, seq_len


def test_picker_valid_for_ring_sdpa():
    # ring_distributed SDPA (ring_size 4) needs S / 8 divisible by q_chunk, for every length the picker sees.
    for seq_len in range(6144, RING_SDPA_MAX_MEASURED_SEQ_LEN + 1, 2048):
        q = ring_sdpa_balanced_q_chunk(seq_len, N_LOCAL_HEADS, NUM_CORES)
        assert (seq_len // 8) % q == 0, (seq_len, q)


@pytest.mark.parametrize("seq_len", [8192, 16384, 24576, 32768, 40960, 49152])
def test_ring_program_config_keeps_base_fields(seq_len):
    cfg = qwen_ring_sdpa_program_config(base_cfg(), seq_len, N_LOCAL_HEADS)
    grid = cfg.compute_with_storage_grid_size
    assert (grid.x, grid.y) == (7, 10)
    assert cfg.k_chunk_size == 512
    assert cfg.exp_approx_mode is False
    assert cfg.q_chunk_size == ring_sdpa_balanced_q_chunk(seq_len, N_LOCAL_HEADS, NUM_CORES)
