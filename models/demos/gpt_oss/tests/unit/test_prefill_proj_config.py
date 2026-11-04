# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the prefill QKV / o_proj 2D program configs (tt/attention/config.py).

prefill_proj_program_config sizes a 2D multicast matmul per sequence length from an L1 model of its circular
buffers. These tests sweep every M tile count up to 128K tokens on the Blackhole grids and check that each config is
well formed (block divisibility, subblock size, the grid covers the output) and that the modelled L1 stays within
PREFILL_PROJ_L1_BYTES. They also check that every case outside the measured envelope (Blackhole, TP = 4, bf8 in0 and
weight) gets None, which keeps the ttnn default. No device is needed.
"""

from unittest.mock import MagicMock

import pytest

import ttnn
from models.demos.gpt_oss.tt.attention.config import PREFILL_PROJ_L1_BYTES, PREFILL_PROJ_TP, prefill_proj_program_config
from models.demos.gpt_oss.tt.attention_configs import GPTOSSAttentionProgramConfig

# Per-device (K tiles, N tiles) of gpt-oss-20b at TP = 4: QKV [S, 2880] x [2880, 1280], o_proj [S, 1024] x [1024, 2880].
ROLE_KN = {"qkv": (90, 40), "o": (32, 90)}
# Output tile bytes per role: QKV writes bf16, o_proj bf8.
OUT_TILE = {"qkv": 2048, "o": 1088}
BF8_TILE = 1088
BF16_TILE = 2048
BH_GRIDS = [(11, 10), (12, 10), (13, 10)]
MAX_M_TILES = 128 * 1024 // 32
MAX_SUBBLOCK_TILES = 8


def modelled_l1_bytes(cfg, role, n_tiles):
    """L1 of the matmul's circular buffers, computed here independently of config.py: in0 and in1 blocks double
    buffered (bf8), the output block and a bf16 intermediate block."""
    in0 = 2 * cfg.out_block_h * cfg.in0_block_w * BF8_TILE
    in1 = 2 * cfg.in0_block_w * cfg.per_core_N * BF8_TILE
    out = cfg.out_block_h * cfg.per_core_N * (OUT_TILE[role] + BF16_TILE)
    return in0 + in1 + out


def config(grid, m_tiles, role, tp=PREFILL_PROJ_TP, in0_dtype=ttnn.bfloat8_b, weight_dtype=ttnn.bfloat8_b):
    k_tiles, n_tiles = ROLE_KN[role]
    return prefill_proj_program_config(
        ttnn.CoreCoord(*grid), m_tiles, k_tiles, n_tiles, role, tp=tp, in0_dtype=in0_dtype, weight_dtype=weight_dtype
    )


@pytest.mark.parametrize("grid", BH_GRIDS, ids=lambda g: f"{g[0]}x{g[1]}")
@pytest.mark.parametrize("role", ["qkv", "o"])
def test_configs_valid_for_every_m(grid, role):
    k_tiles, n_tiles = ROLE_KN[role]
    for m_tiles in range(1, MAX_M_TILES + 1):
        cfg = config(grid, m_tiles, role)
        assert cfg is not None, f"m_tiles {m_tiles}: no config"
        where = f"grid {grid}, {role}, m_tiles {m_tiles}: {cfg}"
        g = cfg.compute_with_storage_grid_size
        # Divisibility.
        assert k_tiles % cfg.in0_block_w == 0, where
        assert cfg.per_core_M % cfg.out_block_h == 0, where
        assert cfg.per_core_N % cfg.out_block_w == 0, where
        assert cfg.out_block_h % cfg.out_subblock_h == 0, where
        assert cfg.out_block_w % cfg.out_subblock_w == 0, where
        # Subblock fits the destination registers.
        assert cfg.out_subblock_h * cfg.out_subblock_w <= MAX_SUBBLOCK_TILES, where
        # The grid fits the device and covers the output exactly in N and without an idle row in M.
        assert g.x <= grid[0] and g.y <= grid[1], where
        assert g.x * cfg.per_core_N == n_tiles, where
        assert g.y * cfg.per_core_M >= m_tiles > (g.y - 1) * cfg.per_core_M, where
        # Circular buffers within the L1 budget.
        assert modelled_l1_bytes(cfg, role, n_tiles) <= PREFILL_PROJ_L1_BYTES, where


@pytest.mark.parametrize("role", ["qkv", "o"])
@pytest.mark.parametrize("m_tiles", [4, 32, 256, 1024, 4096])
def test_none_outside_envelope(role, m_tiles):
    grid = BH_GRIDS[0]
    assert config(grid, m_tiles, role) is not None
    # Other TP: the geometries were measured at TP = 4 only.
    for tp in (1, 2, 8):
        assert config(grid, m_tiles, role, tp=tp) is None, f"tp {tp}"
    # The L1 model assumes bf8 in0 and weight tiles.
    assert config(grid, m_tiles, role, in0_dtype=ttnn.bfloat16) is None
    assert config(grid, m_tiles, role, weight_dtype=ttnn.bfloat16) is None
    # A grid narrower than the role's columns (Wormhole 8x8).
    assert config((8, 8), m_tiles, role) is None
    # Other shapes.
    k_tiles, n_tiles = ROLE_KN[role]
    args = (ttnn.CoreCoord(*grid), m_tiles)
    common = dict(tp=PREFILL_PROJ_TP, in0_dtype=ttnn.bfloat8_b, weight_dtype=ttnn.bfloat8_b)
    assert prefill_proj_program_config(*args, k_tiles, n_tiles * 2, role, **common) is None
    assert prefill_proj_program_config(*args, k_tiles + 1, n_tiles, role, **common) is None


def _fake(shape, dtype):
    t = MagicMock()
    t.padded_shape = list(shape)
    t.dtype = dtype
    return t


@pytest.mark.parametrize("blackhole", [True, False])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_get_prefill_proj_config_gate(monkeypatch, blackhole, tp):
    """ProgramConfig.get_prefill_proj_config: a config only on Blackhole at TP = 4; the o_proj caller's in0_dtype
    override (apply_output_projection casts to bf8) decides over the bf16 tensor it passes."""
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda _: blackhole)
    mesh_device = MagicMock()
    mesh_device.compute_with_storage_grid_size.return_value = ttnn.CoreCoord(11, 10)
    pc = GPTOSSAttentionProgramConfig()
    m = 8192
    x_qkv = _fake((1, 1, m, 2880), ttnn.bfloat8_b)
    w_qkv = _fake((1, 1, 2880, 1280), ttnn.bfloat8_b)
    x_o = _fake((1, 1, m, 1024), ttnn.bfloat16)
    w_o = _fake((1, 1, 1024, 2880), ttnn.bfloat8_b)
    expect = blackhole and tp == PREFILL_PROJ_TP
    assert (pc.get_prefill_proj_config(mesh_device, "qkv", x_qkv, w_qkv, tp) is not None) == expect
    assert (pc.get_prefill_proj_config(mesh_device, "o", x_o, w_o, tp, in0_dtype=ttnn.bfloat8_b) is not None) == expect
    # A bf16 QKV input (no cast before the matmul) keeps the default at any length.
    assert pc.get_prefill_proj_config(mesh_device, "qkv", _fake((1, 1, m, 2880), ttnn.bfloat16), w_qkv, tp) is None
    assert pc.get_prefill_proj_config(mesh_device, "o", x_o, w_o, tp) is None
    # Batched [B, 1, S, K] inputs keep the default.
    assert pc.get_prefill_proj_config(mesh_device, "qkv", _fake((2, 1, m, 2880), ttnn.bfloat8_b), w_qkv, tp) is None
