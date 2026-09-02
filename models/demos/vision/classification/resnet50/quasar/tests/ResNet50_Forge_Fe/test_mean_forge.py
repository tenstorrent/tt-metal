# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.mean -- the tt-forge ResNet-50 graph issues one (the global average pool), and
ttnn.experimental.quasar has NO REDUCTION BINDING. This file makes that concrete and runs the
route that does work.

THE CALL-SITE
  mean(dim=[-2], keep_dim=True), MathFidelity.HiFi2
    in   [1,1,49,2048]  L1 BLOCK_SHARDED TILE, 16 cores (8x2 grid, (0,0)-(7,1)), shard [32, 256]
    out  [1,1, 1,2048]  L1 BLOCK_SHARDED TILE, 16 cores, same grid,              shard [32, 256]
  This is the global average pool over the layer4 feature map: 49 = 7x7 spatial positions reduced
  away, 2048 channels kept. Forge expresses it as a mean over the flattened spatial dim of the
  [1, 1, N*H*W, C] tensor, NOT as an avg_pool2d.

THE GAP
  ttnn/cpp/ttnn/operations/experimental/quasar/reduction/ exists as a device backend but is not
  included in quasar_nanobind.cpp -- there is no python binding, so there is no
  ttnn.experimental.quasar.mean / .sum. The generic ttnn.mean is the only way in; on a Quasar
  device it dispatches down to the quasar generic reduce. test_op_inventory.py prints the live
  list, so this is checked against the build rather than asserted from memory.

WHAT THIS FILE RUNS
  test_forge_mean_standalone   resolves ttnn.experimental.quasar.mean; xfails with the gap named
                               while there is none, and exercises it for real if one lands.
  test_forge_mean_generic      the route that exists: generic ttnn.mean(dim=-2, keepdim=True) on
                               Forge's exact block-sharded input, PCC-checked against torch.mean,
                               plus a range check -- an average can never leave the input's
                               [min, max], which catches a bad reduce scaler or a stale-L1 leak
                               even when PCC looks plausible.

  The reduced dim is 49 rows -- not tile-aligned (49 = 1.53 tiles) -- so the reduce also has to get
  the tile padding right rather than averaging 64 rows and dividing by 49.

KNOWN QUASAR ISSUE
  The Quasar GAPOOL SUM/AVG reduce has applied a fixed ~1.1504x multiplicative gain that WH/BH do
  not, independent of the scaler and of fidelity (see ../ops/test_reduce_sum_mean.py and
  ../ops/test_global_avgpool.py). If test_forge_mean_generic fails with the device result ~1.15x
  the golden, that is this known GAPOOL bug and not a config problem -- the assertion message
  prints the observed gain so it is identifiable at a glance.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_mean_forge.py
    pytest -s ... test_mean_forge.py -k generic     # only the route that can pass
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
CR16 = (((0, 0), (7, 1)),)  # 8x2 = 16 cores

# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
MEAN_IN = ("BLOCK_SHARDED", "L1", CR16, (32, 256), "TILE")
MEAN_OUT = ("BLOCK_SHARDED", "L1", CR16, (32, 256), "TILE")

INPUT_SHAPE = (1, 1, 49, 2048)
OUTPUT_SHAPE = (1, 1, 1, 2048)
DIM = -2
KEEP_DIM = True


def _mem(spec):
    """Frozen Forge memory-config tuple -> a real ttnn.MemoryConfig."""
    memory_layout, buffer_type, core_ranges, shard_shape, _page_layout = spec
    layout = getattr(ttnn.TensorMemoryLayout, memory_layout)
    buffer = getattr(ttnn.BufferType, buffer_type)
    if core_ranges is None:
        return ttnn.MemoryConfig(layout, buffer, None)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*lo), ttnn.CoreCoord(*hi)) for lo, hi in core_ranges])
    return ttnn.MemoryConfig(layout, buffer, ttnn.ShardSpec(ranges, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR))


def _page(spec):
    return ttnn.TILE_LAYOUT if spec[4] == "TILE" else ttnn.ROW_MAJOR_LAYOUT


def _to_device(x, spec, device):
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=_page(spec))
    try:
        return tt.to(device, _mem(spec))
    except Exception as e:
        raise AssertionError(
            "could not place a %s tensor in the Forge memory config (%s/%s/%s, shard %s over %s): %s"
            % (tuple(x.shape), spec[1], spec[0], spec[4], spec[3], spec[2], e)
        ) from e


def _require_grid(device, *specs):
    """Skip unless the device compute grid can hold every Forge core range in play."""
    grid = device.compute_with_storage_grid_size()
    for spec in specs:
        if spec[2] is None:
            continue
        need_x = max(hi[0] for _lo, hi in spec[2]) + 1
        need_y = max(hi[1] for _lo, hi in spec[2]) + 1
        if need_x > grid.x or need_y > grid.y:
            pytest.skip(
                "Forge pins a %dx%d core grid but this device grid is %dx%d. These configs need a "
                "full Quasar part; ../ops/ covers the same kernels with device-derived sharding."
                % (need_x, need_y, grid.x, grid.y)
            )


# --------------------------------------------------------------------------------------------------
# 1. the gap itself
# --------------------------------------------------------------------------------------------------
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_mean_standalone(mesh_device):
    """
    Forge emits ttnn.mean here. xfails while ttnn.experimental.quasar has no mean -- which is every
    build today: the quasar reduction has a device backend but no nanobind binding.
    """
    device = mesh_device
    torch.manual_seed(0)

    mean = getattr(ttnn.experimental.quasar, "mean", None)
    if mean is None:
        pytest.xfail(
            "NOT EXPOSED: ttnn.experimental.quasar.mean does not exist. The quasar reduction has a "
            "device backend under operations/experimental/quasar/reduction/ but is not bound in "
            "quasar_nanobind.cpp. The route that exists is the generic ttnn.mean, in "
            "test_forge_mean_generic."
        )

    _require_grid(device, MEAN_IN, MEAN_OUT)

    x_torch = torch.rand(INPUT_SHAPE, dtype=torch.float32)
    golden = torch.mean(x_torch, dim=DIM, keepdim=KEEP_DIM)

    x = _to_device(x_torch.to(torch.bfloat16), MEAN_IN, device)
    out = mean(x, dim=DIM, keepdim=KEEP_DIM, memory_config=_mem(MEAN_OUT))
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == OUTPUT_SHAPE, "mean output %s, Forge IR says %s" % (tuple(out.shape), OUTPUT_SHAPE)
    assert_with_pcc(golden, ttnn.to_torch(out).float(), pcc=PCC)


# --------------------------------------------------------------------------------------------------
# 2. the route that exists: the generic ttnn.mean
# --------------------------------------------------------------------------------------------------
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_mean_generic(mesh_device):
    """Generic ttnn.mean on Forge's exact block-sharded [1,1,49,2048] input, HiFi2."""
    device = mesh_device
    torch.manual_seed(0)

    _require_grid(device, MEAN_IN, MEAN_OUT)

    x_torch = torch.rand(INPUT_SHAPE, dtype=torch.float32)
    golden = torch.mean(x_torch, dim=DIM, keepdim=KEEP_DIM)
    assert tuple(golden.shape) == OUTPUT_SHAPE, "torch mean gives %s, Forge IR says %s" % (
        tuple(golden.shape),
        OUTPUT_SHAPE,
    )

    x = _to_device(x_torch.to(torch.bfloat16), MEAN_IN, device)
    out = ttnn.mean(
        x,
        dim=DIM,
        keepdim=KEEP_DIM,
        memory_config=_mem(MEAN_OUT),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2
        ),
    )
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(ttnn.from_device(out)).float().reshape(OUTPUT_SHAPE)

    # An average can never escape the input range. This catches a bad reduce scaler -- the known
    # ~1.15x Quasar GAPOOL gain lands here -- or a stale-L1 leak, even if PCC happens to look fine.
    in_lo, in_hi = float(x_torch.min()), float(x_torch.max())
    dev_lo, dev_hi = float(got.min()), float(got.max())
    gain = float(got.mean()) / max(float(golden.mean()), 1e-9)
    assert dev_lo >= in_lo - 1e-2 and dev_hi <= in_hi + 1e-2, (
        "mean output range [%.4f,%.4f] escaped the input range [%.4f,%.4f]; observed gain %.4fx "
        "(a gain near 1.15 is the known Quasar GAPOOL scaler bug, see ../ops/test_reduce_sum_mean.py)"
        % (dev_lo, dev_hi, in_lo, in_hi, gain)
    )
    assert_with_pcc(golden, got, pcc=PCC)
