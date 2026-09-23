# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Non-tile-aligned HW and C for groupnorm_sc_N_1_HW_C.

HW % 32 != 0: the image's last tile-row carries padded rows; pass 2 masks them through the
expansion matmul (masked group-mean tiles -> (0 - 0)^2 = 0), all counts use the true HW.
C % 32 != 0: the last channel block has c_valid < K*32 valid lanes; the membership rows and the
affine rows for lanes >= C are zero, the ROW_MAJOR leg reads / writes only the valid bytes of
each stick. Pinned in both layouts, both placements (interleaved + auto-like L1 block shards,
in_place both ways), the forced streaming regime (segmented pass-2 chunks) and the hardest
sub-case (C % 32 != 0 AND straddling groups, e.g. (1,1,64,200) G=8, Cg=25).
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C, config

from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C import (
    torch_groupnorm_n_1_hw_c,
    compute_pcc,
)
from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C_sharded import (
    block_shard_config,
)

PCC_THRESHOLD = 0.995
RMS_THRESHOLD = 0.02

LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="tile"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
]


def _weights(C, affine):
    gamma = beta = None
    if affine in ("gamma_beta", "gamma_only"):
        gamma = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    if affine == "gamma_beta":
        beta = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    return gamma, beta


def _run(
    device,
    shape,
    num_groups,
    *,
    layout,
    memory_config,
    in_place=False,
    affine="gamma_beta",
    mean_offset=0.0,
    rms_threshold=RMS_THRESHOLD,
):
    torch.manual_seed(42)
    N, _, HW, C = shape
    x = (torch.randn(shape, dtype=torch.float32) + mean_offset).to(torch.bfloat16)
    gamma, beta = _weights(C, affine)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)
    kwargs = {}
    for name, w in (("gamma", gamma), ("beta", beta)):
        if w is not None:
            kwargs[name] = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    tt_y = groupnorm_sc_N_1_HW_C(tt_x, num_groups, in_place=in_place, **kwargs)

    assert list(tt_y.shape) == list(shape)
    assert tt_y.layout == layout
    expected = torch_groupnorm_n_1_hw_c(x, num_groups, gamma=gamma, beta=beta)
    actual = ttnn.to_torch(tt_y)
    assert torch.isfinite(actual.float()).all(), "non-finite values in output"
    pcc = compute_pcc(actual.float(), expected.float())
    rms = float(((actual.float() - expected.float()) ** 2).mean().sqrt())
    tag = f"{shape} G={num_groups} {layout} {memory_config.memory_layout} in_place={in_place} {affine}"
    assert rms <= rms_threshold, f"RMS {rms:.4f} > {rms_threshold} for {tag}"
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD} for {tag}"
    return pcc


def _auto_block_shard(device, shape, layout):
    """Auto-shard-like block shard: extents rounded up to the layout granule
    (TILE 32x32; RM 1 x 8 for bf16 at 16 B L1 alignment), ceil-split over the live grid."""
    N, _, HW, C = shape
    grid = device.compute_with_storage_grid_size()
    hg_unit, wg_unit = (32, 32) if layout == ttnn.TILE_LAYOUT else (1, 8)
    hg = N * math.ceil(HW / hg_unit)
    wg = math.ceil(C / wg_unit)

    def split(count, cap):
        per = math.ceil(count / cap)
        return math.ceil(count / per), per

    ny, per_h = split(hg, grid.y)
    nx, per_w = split(wg, grid.x)
    return block_shard_config([per_h * hg_unit, per_w * wg_unit], (nx, ny))


# --------------------------------------------------------------------------- #
# Non-aligned HW / C cells
# --------------------------------------------------------------------------- #
HW_NON_ALIGNED = [
    pytest.param((1, 1, 17, 64), 1, id="hw17_c64_G1"),
    pytest.param((1, 1, 50, 128), 1, id="hw50_c128_G1"),
    pytest.param((1, 1, 47, 256), 1, id="hw47_c256_G1"),
    pytest.param((2, 1, 100, 128), 1, id="n2_hw100_c128_G1"),
    pytest.param((1, 1, 100, 64), 2, id="hw100_c64_G2"),
]

C_NON_ALIGNED = [
    pytest.param((1, 1, 64, 17), 1, id="c17_G1"),
    pytest.param((1, 1, 64, 50), 1, id="c50_G1"),
    pytest.param((1, 1, 128, 100), 1, id="c100_G1"),
    pytest.param((2, 1, 64, 47), 1, id="n2_c47_G1"),
    # C % 32 != 0 AND straddling groups (num_groups > 1): the hardest sub-case.
    pytest.param((1, 1, 64, 48), 2, id="c48_G2_straddle"),
    pytest.param((1, 1, 64, 80), 4, id="c80_G4_straddle"),
    pytest.param((1, 1, 128, 48), 3, id="c48_G3_straddle"),
    pytest.param((1, 1, 128, 144), 4, id="c144_G4_straddle"),
    pytest.param((1, 1, 64, 200), 8, id="c200_G8_straddle"),
    pytest.param((2, 1, 64, 48), 2, id="n2_c48_G2_straddle"),
]

BOTH_NON_ALIGNED = [
    # Both dims ragged at once.
    pytest.param((1, 1, 50, 50), 1, id="hw50_c50_G1"),
    pytest.param((2, 1, 47, 200), 8, id="n2_hw47_c200_G8"),
]


@pytest.mark.parametrize("shape,num_groups", HW_NON_ALIGNED + C_NON_ALIGNED + BOTH_NON_ALIGNED)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_non_aligned_interleaved(device, shape, num_groups, layout):
    _run(device, shape, num_groups, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 50, 128), 1, id="hw50_c128_G1"),
        pytest.param((1, 1, 64, 200), 8, id="c200_G8_straddle"),
        pytest.param((2, 1, 47, 200), 8, id="n2_hw47_c200_G8"),
    ],
)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "affine", [pytest.param("gamma_only", id="gamma_only"), pytest.param("no_affine", id="no_affine")]
)
def test_non_aligned_affine_variants(device, shape, num_groups, layout, affine):
    _run(device, shape, num_groups, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, affine=affine)


@pytest.mark.parametrize("shape,num_groups", HW_NON_ALIGNED + C_NON_ALIGNED + BOTH_NON_ALIGNED)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("in_place", [pytest.param(False, id="out"), pytest.param(True, id="in_place")])
def test_non_aligned_block_sharded(device, shape, num_groups, layout, in_place):
    mc = _auto_block_shard(device, shape, layout)
    _run(device, shape, num_groups, layout=layout, memory_config=mc, in_place=in_place)


@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 100, 64), 2, id="hw100_c64_G2"),
        pytest.param((2, 1, 100, 128), 1, id="n2_hw100_c128_G1"),
        pytest.param((1, 1, 200, 80), 4, id="hw200_c80_G4"),
    ],
)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_non_aligned_streaming_regime(device, monkeypatch, shape, num_groups, layout):
    """Streaming regime (input re-read per pass): the reader's pass-2 chunk order must follow
    compute's [body][tail] segments when the last tile-row is ragged."""
    monkeypatch.setattr(config, "FORCE_STREAMING", True)
    _run(device, shape, num_groups, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "shape,num_groups",
    [pytest.param((1, 1, 50, 128), 1, id="hw50_c128_G1"), pytest.param((1, 1, 64, 200), 8, id="c200_G8")],
)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_non_aligned_large_mean(device, shape, num_groups, layout):
    """mean = 10 sigma: an unmasked pad row would add (0 - mean)^2 = 100 sigma^2 per pad stick to the
    variance (~36x on (1,1,50,128)) and collapse the PCC — the masked-mean pass 2 must keep the
    stats exact. The RMS gate is the tf32-statistics baseline at this mean (3 roundings of |mean|,
    ~0.015-0.03 absolute), which the tile-aligned neighbours measure too."""
    _run(
        device,
        shape,
        num_groups,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mean_offset=10.0,
        rms_threshold=0.05,
    )
