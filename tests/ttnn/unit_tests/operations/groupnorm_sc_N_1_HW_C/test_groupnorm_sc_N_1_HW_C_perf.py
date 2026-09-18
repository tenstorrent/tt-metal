# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-measurement cases for groupnorm_sc_N_1_HW_C (run under --profile).

Not an acceptance test: correctness is checked loosely (PCC) so the device
kernel duration in the Tracy per-op CSV is the number of interest. Shapes are
the SDXL U-Net GroupNorms (TILE, DRAM interleaved; reference `ns_il` in
eval/golden_tests/groupnorm_sc_N_1_HW_C/perf_cases.py) plus two VAE decoder
shapes that force the streaming regime at large HW.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C import _run_case

SDXL_SHAPES = [
    pytest.param((1, 1, 16384, 320), 32, id="unet_16384x320_ref600us"),
    pytest.param((1, 1, 4096, 640), 32, id="unet_4096x640_ref225us"),
    pytest.param((1, 1, 1024, 1280), 32, id="unet_1024x1280_ref97us"),
    pytest.param((1, 1, 1024, 2560), 32, id="unet_1024x2560_ref120us"),
    pytest.param((1, 1, 4096, 1920), 32, id="unet_4096x1920_ref336us"),
    pytest.param((1, 1, 16384, 960), 32, id="unet_16384x960_ref818us"),
    pytest.param((1, 1, 16384, 640), 32, id="unet_16384x640_ref474us"),
]

VAE_SHAPES = [
    pytest.param((1, 1, 65536, 512), 32, id="vae_65536x512_ref1015us"),
    pytest.param((1, 1, 262144, 256), 32, id="vae_262144x256_ref2931us"),
]


@pytest.mark.parametrize("shape,num_groups", SDXL_SHAPES + VAE_SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_perf_shapes(device, shape, num_groups, layout):
    _run_case(device, shape, num_groups, layout=layout, affine="gamma_beta")


# ---------------------------------------------------------------------------------------------
# Refinement 3: the SDXL UNet sharded ROW_MAJOR contract (perf_cases.py: bf16 RM input on the
# model's 8x8 block shard, in_place, gamma+beta, G = 32) — one cell per --profile run:
#   scripts/run_safe_pytest.sh --profile <this file> -k direct-sdxl_rm_16384x320
# `staged` pins config.RM_SHARD_DIRECT_VIEW = False (the Refinement-1 staged-stick path) so the
# baseline stays measurable next to the direct-view number.
# ---------------------------------------------------------------------------------------------
from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C_sharded import run_sharded
from ttnn.operations.groupnorm_sc_N_1_HW_C import config as gn_config

# (hw, c, [shard_h, shard_w], achievable_ns at 1350 MHz) — perf_cases.py `_UNET`, ns_rm_model_shard
SDXL_RM_MODEL_SHARDS = [
    pytest.param((1, 1, 16384, 320), [2048, 40], 337_061, id="sdxl_rm_16384x320"),
    pytest.param((1, 1, 4096, 320), [512, 40], 102_861, id="sdxl_rm_4096x320"),
    pytest.param((1, 1, 4096, 640), [512, 80], 104_719, id="sdxl_rm_4096x640"),
    pytest.param((1, 1, 1024, 640), [128, 80], 45_043, id="sdxl_rm_1024x640"),
    pytest.param((1, 1, 1024, 1280), [128, 160], 44_467, id="sdxl_rm_1024x1280"),
    pytest.param((1, 1, 1024, 2560), [128, 320], 52_947, id="sdxl_rm_1024x2560"),
    pytest.param((1, 1, 1024, 1920), [128, 240], 54_141, id="sdxl_rm_1024x1920"),
    pytest.param((1, 1, 4096, 1920), [512, 240], 139_022, id="sdxl_rm_4096x1920"),
    pytest.param((1, 1, 4096, 1280), [512, 160], 95_819, id="sdxl_rm_4096x1280"),
    pytest.param((1, 1, 4096, 960), [512, 120], 106_423, id="sdxl_rm_4096x960"),
    pytest.param((1, 1, 16384, 960), [2048, 120], 389_687, id="sdxl_rm_16384x960"),
    pytest.param((1, 1, 16384, 640), [2048, 80], 381_651, id="sdxl_rm_16384x640"),
]


@pytest.mark.parametrize("shape,shard_shape,achievable_ns", SDXL_RM_MODEL_SHARDS)
@pytest.mark.parametrize("direct", [pytest.param(True, id="direct"), pytest.param(False, id="staged")])
def test_perf_sdxl_sharded_rm(device, monkeypatch, shape, shard_shape, achievable_ns, direct):
    monkeypatch.setattr(gn_config, "RM_SHARD_DIRECT_VIEW", direct)
    run_sharded(device, shape, 32, layout=ttnn.ROW_MAJOR_LAYOUT, shard_shape=shard_shape, grid=(8, 8), in_place=True)


# ---------------------------------------------------------------------------------------------
# Refinement 5: the DRAM-bound VAE decoder shapes (perf_cases.py `_VAE`, bf16 DRAM interleaved,
# gamma+beta, G = 32) — one cell per --profile run:
#   scripts/run_safe_pytest.sh --profile "<this file>::test_perf_vae_streaming[two_pass-tile-vae_262144x256_ref2931us]"
# `three_pass` pins config.STREAMING_TWO_PASS = False (the Phase-0 schedule: input read once per pass)
# so the baseline stays measurable next to the two-pass number. Measured (BH, 110 cores, device us,
# three_pass -> two_pass): TILE 262144x256 1402 -> 1072, RM 262144x256 1908 -> 1436, TILE 262144x512
# 2687 -> 2100, TILE 1048576x128 2720 -> 2096 (both schedules ~380 GB/s aggregate: the win is the 4V -> 3V
# traffic ratio). Kept small on purpose: the whole unit dir is re-run in plain mode after every phase.
# ---------------------------------------------------------------------------------------------
VAE_STREAMING_SHAPES = [
    pytest.param((1, 1, 262144, 256), 32, id="vae_262144x256_ref2931us"),
    pytest.param((1, 1, 262144, 512), 32, id="vae_262144x512_ref3846us"),
    pytest.param((1, 1, 1048576, 128), 32, id="vae_1048576x128_ref10168us"),
]


@pytest.mark.parametrize("shape,num_groups", VAE_STREAMING_SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("two_pass", [pytest.param(True, id="two_pass"), pytest.param(False, id="three_pass")])
def test_perf_vae_streaming(device, monkeypatch, shape, num_groups, layout, two_pass):
    monkeypatch.setattr(gn_config, "STREAMING_TWO_PASS", two_pass)
    _run_case(device, shape, num_groups, layout=layout, affine="gamma_beta")


# The 1.2 MB/core cell against the L1 budget: `budget_1M` streams (two_pass), `budget_1440K` makes the
# K = 4 assignment resident at Q = 1 (three passes over L1, 149 chunks of 4 tiles per pass) — the two
# regimes the residency-first split search chooses between on this shape; and STREAM_DEPTH 3 vs 2 on
# the streaming schedule (the reader prefetches through the combine).
@pytest.mark.parametrize(
    "budget,depth",
    [
        pytest.param(1_000_000, 2, id="budget_1M-depth2"),
        pytest.param(1_000_000, 3, id="budget_1M-depth3"),
        pytest.param(1_440_000, 2, id="budget_1440K-depth2"),
    ],
)
def test_perf_vae_262144x256_budget_depth(device, monkeypatch, budget, depth):
    monkeypatch.setattr(gn_config, "L1_CB_BUDGET_BYTES", budget)
    monkeypatch.setattr(gn_config, "STREAM_DEPTH", depth)
    _run_case(device, (1, 1, 262144, 256), 32, layout=ttnn.TILE_LAYOUT, affine="gamma_beta")
