# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the all-gather matmul pins the H3 DiT resolves per compute grid, mode and path."""

from __future__ import annotations

import pytest

import ttnn
from models.tt_dit.models.transformers.minimax_h3.agmm_config import AgmmMode, agmm_pins

MODE_FLAG = "MINIMAX_H3_AGMM_MODE"
BLACKHOLE, WORMHOLE = ttnn.CoreCoord(12, 10), ttnn.CoreCoord(8, 9)
# (K, N) per device of the TP=4 DiT block's fused layers.
TO_QKV, FF1, TO_OUT = (5376, 5376), (5376, 7168), (7168, 1344)
ACCURATE, BALANCED = AgmmMode.ACCURATE, AgmmMode.BALANCED


def _pins(layer, grid=BLACKHOLE, **kwargs):
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    return agmm_pins(*layer, full_grid=grid, num_links=2, compute_kernel_config=compute_kernel_config, **kwargs)


def _program(pin):
    config = pin.matmul_config
    return (
        (config.M_block_size, config.K_block_size, config.N_block_size),
        (config.subblock_h, config.subblock_w),
        pin.transpose_core_grid,
        pin.compute_kernel_config.fp32_dest_acc_en,
    )


@pytest.fixture(autouse=True)
def _no_flags(monkeypatch):
    monkeypatch.delenv(MODE_FLAG, raising=False)
    for k, n in (TO_QKV, FF1, TO_OUT):
        monkeypatch.delenv(f"MINIMAX_H3_AGMM_PIN_{k}_{n}", raising=False)


@pytest.mark.parametrize(("layer", "blocking"), [(TO_QKV, (8, 7, 12)), (FF1, (8, 6, 16))], ids=["to_qkv", "ff1"])
def test_blackhole_balanced_pins(layer, blocking):
    """Blackhole's BALANCED to_qkv and ff1 on the bucketed path: the swept blocking on the transposed grid, with
    fp32 Dst accumulation off for an 8-tile subblock, still at HiFi2."""
    pins = _pins(layer, mode=BALANCED, dynamic_length=False)
    assert set(pins) == {"short", "long"}
    for pin in pins.values():
        assert _program(pin) == (blocking, (4, 2), True, False)
        assert pin.compute_kernel_config.math_fidelity == ttnn.MathFidelity.HiFi2
        assert (layer[0] // ttnn.TILE_SIZE // 4) % blocking[1] == 0  # K_block divides the K tiles per device


@pytest.mark.parametrize("layer", [TO_QKV, FF1, TO_OUT], ids=["to_qkv", "ff1", "to_out"])
def test_blackhole_balanced_leaves_the_dynamic_path(layer):
    """The Blackhole BALANCED pins were swept on the bucketed path only: dynamic length keeps ACCURATE's."""
    balanced, accurate = _pins(layer, mode=BALANCED), _pins(layer, mode=ACCURATE)
    assert {name: _program(pin) for name, pin in balanced.items()} == {
        name: _program(pin) for name, pin in accurate.items()
    }


def test_blackhole_accurate_keeps_its_pins():
    """ACCURATE (MINIMAX_H3_AGMM_MODE=accurate) is the way back: the dynamic-length pins as they were, fp32 Dst
    and 2x2 subblocks, and no pin on the bucketed path, which keeps its length-keyed table."""
    expected = {
        TO_QKV: {"short": ((4, 6, 16), False), "long": ((4, 6, 16), False)},
        FF1: {"short": ((6, 3, 16), False), "long": ((4, 3, 16), False)},
        TO_OUT: {"short": ((6, 8, 6), True), "long": ((6, 8, 6), True)},
    }
    for layer, classes in expected.items():
        pins = _pins(layer, mode=ACCURATE)
        for name, (blocking, transpose) in classes.items():
            assert _program(pins[name]) == (blocking, (2, 2), transpose, True), (layer, name)
        assert _pins(layer, mode=ACCURATE, dynamic_length=False) is None


@pytest.mark.parametrize("mode", [ACCURATE, BALANCED])
@pytest.mark.parametrize("grid", [WORMHOLE, ttnn.CoreCoord(11, 10)], ids=["wormhole", "unpinned_grid"])
def test_bucketed_path_elsewhere_keeps_the_table(grid, mode):
    """Only pins marked `bucketed` reach the bucketed path: none of Wormhole's, and none on an unpinned grid."""
    for layer in (TO_QKV, FF1, TO_OUT):
        assert _pins(layer, grid=grid, mode=mode, dynamic_length=False) is None
