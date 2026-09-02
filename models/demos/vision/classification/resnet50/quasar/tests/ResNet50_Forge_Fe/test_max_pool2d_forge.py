# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for the single ttnn.max_pool2d the tt-forge ResNet-50 graph issues, run on Quasar via
ttnn.experimental.quasar.max_pool2d.

  batch 1, 64 channels, 112x112 in, kernel 3x3, stride 2, padding 1, dilation 1,
  ceil_mode=False, reallocate_halo_output=False, config_tensor_in_dram=True
    in   [1,1,12544,64]  L1 HEIGHT_SHARDED TILE       56 cores (0,0)-(7,6), shard [224, 64]
    out  [1,1, 3136,64]  L1 HEIGHT_SHARDED ROW_MAJOR  56 cores (0,0)-(7,6), shard [ 56, 64]
  output spatial: (112 - 3 + 2*1)/2 + 1 = 56  ->  56*56 = 3136 rows.

HOW THIS DIFFERS FROM ../ops/test_max_pool2d.py
  The sibling test covers the same logical stem pool but builds its own grid-adaptive sharding and
  passes no memory_config. This one pins what FORGE chose, and the differences matter:
    * the INPUT is TILE-layout height-sharded over 56 cores with a [224, 64] shard -- Forge feeds
      the pool the tilized stem-conv output directly, it does not untilize first;
    * the OUTPUT is pinned ROW_MAJOR height-sharded with a [56, 64] shard, i.e. the pool is asked
      to change page layout as part of the op (this is what makes layer1's conv1 and downsample
      the only two convs in the graph with a row-major activation);
    * reallocate_halo_output=False (the binding's default is True) and config_tensor_in_dram=True
      (default False), both of which change where the halo / config tensors live.

LAYOUT / GOLDEN CONVERSION
  ttnn pools are channels-last flattened: [1, 1, N*H*W, C]. torch.nn.functional.max_pool2d is
  NCHW. So: build NCHW, take the torch golden, permute NCHW->NHWC and flatten for the device
  input, and flatten the golden the same way for the compare. Max-pool pads with -inf in both
  torch and ttnn, so the padding never pollutes a window max.

KNOWN QUASAR ISSUE
  max_pool2d has HUNG on Quasar in the pool-reduce dest handshake in compute_pool_2d.cpp (pack
  tile_regs_wait / math WFD / unpack UPTW never rendezvous) -- see ../ops/test_max_pool2d.py. A
  hang, not a failure, is the expected symptom today, hence the module timeout.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_max_pool2d_forge.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99  # max-pool moves values, it does not compute them

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
CR56 = (((0, 0), (7, 6)),)  # 8x7 = 56 cores

# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
POOL_IN = ("HEIGHT_SHARDED", "L1", CR56, (224, 64), "TILE")
POOL_OUT = ("HEIGHT_SHARDED", "L1", CR56, (56, 64), "ROW_MAJOR")

BATCH, CHANNELS, INPUT_HW = 1, 64, 112
KERNEL, STRIDE, PADDING, DILATION = (3, 3), (2, 2), (1, 1), (1, 1)
CEIL_MODE = False
REALLOCATE_HALO_OUTPUT = False
CONFIG_TENSOR_IN_DRAM = True


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


def _flat_nhwc(x_nchw):
    n, c, h, w = x_nchw.shape
    return x_nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c).contiguous()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_max_pool2d(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    _require_grid(device, POOL_IN, POOL_OUT)

    # ---- torch golden (NCHW) --------------------------------------------------------------------
    x_nchw = torch.rand((BATCH, CHANNELS, INPUT_HW, INPUT_HW), dtype=torch.bfloat16)
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(),
        kernel_size=KERNEL,
        stride=STRIDE,
        padding=PADDING,
        dilation=DILATION,
        ceil_mode=CEIL_MODE,
    )
    oh, ow = golden_nchw.shape[2], golden_nchw.shape[3]
    assert (oh, ow) == (56, 56), "IR says the pool output is 56x56, torch says %dx%d" % (oh, ow)

    # ---- the op, in Forge's exact placement -----------------------------------------------------
    x = _to_device(_flat_nhwc(x_nchw), POOL_IN, device)

    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=BATCH,
        input_h=INPUT_HW,
        input_w=INPUT_HW,
        channels=CHANNELS,
        kernel_size=KERNEL,
        stride=STRIDE,
        padding=PADDING,
        dilation=DILATION,
        ceil_mode=CEIL_MODE,
        memory_config=_mem(POOL_OUT),
        reallocate_halo_output=REALLOCATE_HALO_OUTPUT,
        config_tensor_in_dram=CONFIG_TENSOR_IN_DRAM,
        dtype=ttnn.bfloat16,
        output_layout=_page(POOL_OUT),
    )
    ttnn.synchronize_device(device)

    # ---- checks ---------------------------------------------------------------------------------
    got_layout = out.memory_config().memory_layout
    assert got_layout == getattr(ttnn.TensorMemoryLayout, POOL_OUT[0]), "pool landed in %s but Forge asked for %s" % (
        got_layout,
        POOL_OUT[0],
    )
    assert out.layout == _page(POOL_OUT), "pool output page layout is %s but Forge asked for %s" % (
        out.layout,
        POOL_OUT[4],
    )
    assert out.shape[-2] == BATCH * oh * ow, "pool output rows: got %d, Forge IR says %d" % (
        out.shape[-2],
        BATCH * oh * ow,
    )

    golden_flat = _flat_nhwc(golden_nchw.to(torch.bfloat16)).float()
    got = ttnn.to_torch(out).float().reshape(1, 1, BATCH * oh * ow, -1)[:, :, :, :CHANNELS]
    assert tuple(got.shape) == tuple(golden_flat.shape), "pool output %s, golden %s" % (
        tuple(got.shape),
        tuple(golden_flat.shape),
    )
    assert_with_pcc(golden_flat, got, pcc=PCC)
