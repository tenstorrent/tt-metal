# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the prefill SDPA program config (tt/attention/operations.py): the Blackhole wide config for
>= 8 Q heads per chip, the fixed configs otherwise, the env overrides, the diagnostic knobs. No device.

    pytest models/demos/gemma4/tests/unit/test_prefill_sdpa_config_host.py --noconftest -p no:cacheprovider
"""

import pytest

from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.attention import operations as ops

KNOBS = (
    "GEMMA4_PREFILL_SDPA_WIDE",
    "GEMMA4_PREFILL_SDPA_QCHUNK",
    "GEMMA4_PREFILL_SDPA_KCHUNK",
    "GEMMA4_PREFILL_SDPA_GRID",
    "GEMMA4_PREFILL_SDPA_EXP_APPROX",
    "GEMMA4_PREFILL_SDPA_PROGCFG",
)


def _grid(pc):
    return (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)


def test_wide_config_for_eight_heads_on_blackhole():
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8)
    if is_blackhole():
        assert _grid(pc) == ops.PREFILL_SDPA_WIDE_GRID and pc.q_chunk_size == ops.PREFILL_SDPA_WIDE_Q
    else:
        assert _grid(pc) == (8, 8) and pc.q_chunk_size == 256
    assert pc.k_chunk_size == 128 and pc.exp_approx_mode is False


def test_narrow_chips_take_the_op_defaults_on_blackhole(monkeypatch):
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=4)
    if is_blackhole():
        assert pc is None
        monkeypatch.setenv("GEMMA4_PREFILL_SDPA_PROGCFG", "1")  # opt back in to the fixed config
        pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=4)
    assert _grid(pc) == (8, 8) and (pc.q_chunk_size, pc.k_chunk_size) == (256, 128)
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_PROGCFG", "0")  # the op defaults everywhere
    assert ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8) is None


def test_fixed_configs_otherwise():
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=None)
    assert _grid(pc) == (8, 8) and (pc.q_chunk_size, pc.k_chunk_size) == (256, 128)
    pc = ops.prefill_sdpa_program_config(512, 2048, sliding_window=None, num_heads=8)
    assert _grid(pc) == (8, 4) and (pc.q_chunk_size, pc.k_chunk_size) == (128, 128)


def test_window_cap_and_short_sequences():
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=128, num_heads=8)
    assert pc.k_chunk_size == 64
    pc = ops.prefill_sdpa_program_config(256, 64, sliding_window=None, num_heads=8)
    assert pc.q_chunk_size == 64 and pc.k_chunk_size == 64


def test_env_overrides_and_knobs(monkeypatch):
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_WIDE", "0")
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8)
    assert _grid(pc) == (8, 8) and pc.q_chunk_size == 256
    monkeypatch.delenv("GEMMA4_PREFILL_SDPA_WIDE")
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_QCHUNK", "64")
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_GRID", "8x4")
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8)
    assert pc.q_chunk_size == 64 and _grid(pc) == (8, 4)
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_PROGCFG", "0")
    assert ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8) is None
