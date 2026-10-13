# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the fused decode gate (tt/fused_decode/config.py).

fused_decode_layout_supported decides which decoder layers use the fused decode path. It must accept exactly the
validated envelope (gpt-oss-20b, Blackhole 1x4, TP=4, one token per device) and keep every other configuration on
the original decode path. No device is needed.
"""

from unittest.mock import MagicMock

import pytest

from models.demos.gpt_oss.tt.fused_decode.config import (
    FUSED_DECODE_DRAM_BANKS,
    FUSED_DECODE_MIN_GRID,
    fused_decode_layout_supported,
    fused_decode_supported,
)

# gpt-oss-20b on a QuietBox 2: Blackhole, mesh 1x4, TP=4, EP=1, 32 experts, batch 1, 11x10 grid, 8 DRAM banks.
SUPPORTED = dict(
    is_blackhole=True,
    mesh_shape=(1, 4),
    tp=4,
    ep=1,
    num_experts=32,
    use_throughput_experts=False,
    tokens=1,
    grid=(11, 10),
    dram_banks=8,
)


def test_supported_envelope():
    assert fused_decode_layout_supported(**SUPPORTED)


def test_minimum_grid_is_supported():
    assert fused_decode_layout_supported(**{**SUPPORTED, "grid": FUSED_DECODE_MIN_GRID})


@pytest.mark.parametrize(
    "change",
    [
        {"is_blackhole": False},
        {"mesh_shape": (1, 1), "tp": 1},
        {"mesh_shape": (1, 2), "tp": 2},
        {"mesh_shape": (1, 8), "tp": 8},
        {"mesh_shape": (4, 8), "tp": 8, "ep": 4},
        {"tp": 2},
        {"ep": 4},
        {"num_experts": 128},
        {"use_throughput_experts": True},
        {"tokens": 32},
        {"grid": (FUSED_DECODE_MIN_GRID[0] - 1, 10)},
        {"grid": (11, FUSED_DECODE_MIN_GRID[1] - 1)},
        {"dram_banks": FUSED_DECODE_DRAM_BANKS - 1},
    ],
    ids=[
        "wormhole",
        "mesh_1x1",
        "mesh_1x2",
        "mesh_1x8",
        "mesh_4x8",
        "tp_2",
        "ep_4",
        "gpt_oss_120b_experts",
        "throughput_experts",
        "batch_32",
        "grid_too_narrow",
        "grid_too_short",
        "seven_dram_banks",
    ],
)
def test_outside_envelope_keeps_original_path(change):
    assert not fused_decode_layout_supported(**{**SUPPORTED, **change})


def test_no_mesh_config_keeps_original_path():
    # Without a mesh config the device is never queried.
    mesh_device = MagicMock()
    assert not fused_decode_supported(mesh_device, None, MagicMock(), False, 1)
    mesh_device.compute_with_storage_grid_size.assert_not_called()


@pytest.mark.parametrize(
    "env, name",
    [(None, "bfp4"), ("", "bfp4"), ("bfp4", "bfp4"), ("bfp8", "bfp8"), (" BFP8 ", "bfp8")],
)
def test_expert_decode_weight_dtype(monkeypatch, env, name):
    import ttnn
    from models.demos.gpt_oss.tt.fused_decode.config import (
        ENV_FUSED_EXPERT_DTYPE,
        expert_decode_weight_dtype,
        expert_decode_weight_dtype_name,
    )

    if env is None:
        monkeypatch.delenv(ENV_FUSED_EXPERT_DTYPE, raising=False)
    else:
        monkeypatch.setenv(ENV_FUSED_EXPERT_DTYPE, env)
    assert expert_decode_weight_dtype_name() == name
    assert expert_decode_weight_dtype() == {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[name]


def test_expert_decode_weight_dtype_rejects_unknown(monkeypatch, expect_error):
    from models.demos.gpt_oss.tt.fused_decode.config import ENV_FUSED_EXPERT_DTYPE, expert_decode_weight_dtype

    monkeypatch.setenv(ENV_FUSED_EXPERT_DTYPE, "bf16")
    with expect_error(ValueError, "expected one of"):
        expert_decode_weight_dtype()


@pytest.mark.parametrize(
    "tile_bytes, packets",
    [
        (576, (25 * 576, 1, 0)),  # BFP4 down segment: one 14400 B packet, as before
        (1088, (13 * 1088, 1, 12 * 1088)),  # BFP8: 27200 B > the 16 KB burst -> 13 + 12 tiles
    ],
)
def test_down_segment_packets(tile_bytes, packets):
    from models.demos.gpt_oss.tt.fused_decode.stream import MAX_PACKET_BYTES, _segment_packets

    seg_tiles = 768 // 32 + 1  # gpt-oss-20b at TP=4: I_pad 768 -> 24 weight tiles + 1 bias tile
    packet, full, tail = _segment_packets(seg_tiles, tile_bytes)
    assert (packet, full, tail) == packets
    assert packet <= MAX_PACKET_BYTES and tail < packet
    assert full * packet + tail == seg_tiles * tile_bytes
