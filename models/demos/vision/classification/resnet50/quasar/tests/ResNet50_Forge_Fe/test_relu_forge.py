# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.relu -- the tt-forge ResNet-50 graph issues 16 of them, and ttnn.experimental.quasar has NO
STANDALONE RELU. This file makes that concrete and runs the route that does work.

THE CALL-SITES
  Forge places relu as a SEPARATE op after each residual add, one per bottleneck:
        add(conv3_out, skip)
        relu(...)                <-- 16 of these, no quasar equivalent
  Shapes and memory configs are identical to the adds they consume:
        layer1  3 x [1,1,3136, 256]  HEIGHT_SHARDED, 49 cores, shard [ 64,  256]
        layer2  4 x [1,1, 784, 512]  HEIGHT_SHARDED, 25 cores, shard [ 32,  512]
        layer3  6 x [1,1, 196,1024]  BLOCK_SHARDED,  56 cores, shard [ 32,  128]
        layer4  3 x [1,1,  49,2048]  BLOCK_SHARDED,  16 cores, shard [ 32,  256]

THE GAP
  The Quasar namespace binds data movement, conv2d, the pools, the matmul family and a BINARY
  front-end (add / subtract / multiply / comparisons / ...). It binds no plain unary activation --
  there is no quasar relu, sigmoid or gelu. (`prelu`, `pow` and `polyval` are the only
  unary-with-param ops bound, and none of them is relu.) test_op_inventory.py prints the live
  list, so this is checked against the build rather than asserted from memory.

  Forge DOES fuse relu into 33 of the 53 convs via Conv2dConfig.activation, and that path works on
  Quasar -- test_conv2d_forge.py covers those. It is only the 16 post-add relus that have no home.

WHAT THIS FILE RUNS
  test_forge_relu_standalone      resolves ttnn.experimental.quasar.relu. Today there is none, so
                                  it xfails with the gap named. If a future build adds one, the
                                  test runs the real op and PCC-checks it with no edit needed.
  test_forge_relu_fused_into_add  the route that DOES exist: collapse the add + relu pair into one
                                      quasar.add(a, b, activations=[UnaryWithParam(RELU)])
                                  which is exactly what the hand-written metal quasar resnet50
                                  does (resnet50Bottleneck.__call__ uses add_ with a fused RELU).
                                  If Forge is to target Quasar, this fusion is the thing the
                                  compiler has to do.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_relu_forge.py
    pytest -s ... test_relu_forge.py -k fused        # only the workaround that can pass
    pytest -s ... test_relu_forge.py -k "not dup"    # the 4 distinct configs
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99  # relu is a clamp, not a computation

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
CR56 = (((0, 0), (7, 6)),)  # 8x7     = 56 cores
CR49 = (((0, 0), (7, 5)), ((0, 6), (0, 6)))  # 8x6 + 1 = 49 cores
CR25 = (((0, 0), (7, 2)), ((0, 3), (0, 3)))  # 8x3 + 1 = 25 cores
CR16 = (((0, 0), (7, 1)),)  # 8x2     = 16 cores

# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
# fmt: off
HS49_64x256 = ("HEIGHT_SHARDED", "L1", CR49, (64, 256), "TILE")
HS25_32x512 = ("HEIGHT_SHARDED", "L1", CR25, (32, 512), "TILE")
BS56_32x128 = ("BLOCK_SHARDED",  "L1", CR56, (32, 128), "TILE")
BS16_32x256 = ("BLOCK_SHARDED",  "L1", CR16, (32, 256), "TILE")

# (idx, tag,          shape,               input_and_output_mem)
# Forge gives the relu the same memory config in and out, matching the add it follows.
RELU_CASES = [
    ( 0, "layer1.0", (1, 1, 3136,  256), HS49_64x256),
    ( 1, "layer1.1", (1, 1, 3136,  256), HS49_64x256),
    ( 2, "layer1.2", (1, 1, 3136,  256), HS49_64x256),
    ( 3, "layer2.0", (1, 1,  784,  512), HS25_32x512),
    ( 4, "layer2.1", (1, 1,  784,  512), HS25_32x512),
    ( 5, "layer2.2", (1, 1,  784,  512), HS25_32x512),
    ( 6, "layer2.3", (1, 1,  784,  512), HS25_32x512),
    ( 7, "layer3.0", (1, 1,  196, 1024), BS56_32x128),
    ( 8, "layer3.1", (1, 1,  196, 1024), BS56_32x128),
    ( 9, "layer3.2", (1, 1,  196, 1024), BS56_32x128),
    (10, "layer3.3", (1, 1,  196, 1024), BS56_32x128),
    (11, "layer3.4", (1, 1,  196, 1024), BS56_32x128),
    (12, "layer3.5", (1, 1,  196, 1024), BS56_32x128),
    (13, "layer4.0", (1, 1,   49, 2048), BS16_32x256),
    (14, "layer4.1", (1, 1,   49, 2048), BS16_32x256),
    (15, "layer4.2", (1, 1,   49, 2048), BS16_32x256),
]
# fmt: on


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


def _mem_tag(spec):
    kind = {"HEIGHT_SHARDED": "HS", "BLOCK_SHARDED": "BS", "WIDTH_SHARDED": "WS"}[spec[0]]
    cores = sum((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) for lo, hi in spec[2])
    return "%s%d" % (kind, cores)


def _id(case):
    idx, tag, shape, mem = case
    first = next(c[0] for c in RELU_CASES if c[3] is mem)
    return "%02d_%s_%dx%d_%s%s" % (
        idx,
        tag.replace("layer", "L").replace(".", "_"),
        shape[2],
        shape[3],
        _mem_tag(mem),
        "_dup%d" % first if first != idx else "",
    )


# --------------------------------------------------------------------------------------------------
# 1. the gap itself
# --------------------------------------------------------------------------------------------------
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("case", RELU_CASES, ids=[_id(c) for c in RELU_CASES])
def test_forge_relu_standalone(mesh_device, case):
    """
    Forge emits a standalone ttnn.relu here. xfails while ttnn.experimental.quasar has no relu --
    which is every build today. If one lands, this test starts exercising it for real.
    """
    device = mesh_device
    torch.manual_seed(0)

    relu = getattr(ttnn.experimental.quasar, "relu", None)
    if relu is None:
        pytest.xfail(
            "NOT EXPOSED: ttnn.experimental.quasar.relu does not exist. The quasar namespace binds "
            "no plain unary activation, so the 16 post-add relus in the Forge graph have no direct "
            "route. The one that exists is the fusion in test_forge_relu_fused_into_add."
        )

    idx, tag, shape, mem = case
    _require_grid(device, mem)

    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    golden = torch.relu(x_torch.float())

    x = _to_device(x_torch, mem, device)
    out = relu(x, memory_config=_mem(mem))
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == shape, "relu%d output %s, IR says %s" % (idx, tuple(out.shape), shape)
    assert_with_pcc(golden, ttnn.to_torch(out).float(), pcc=PCC)


# --------------------------------------------------------------------------------------------------
# 2. the route that exists: fuse it into the preceding add
# --------------------------------------------------------------------------------------------------
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("case", RELU_CASES, ids=[_id(c) for c in RELU_CASES])
def test_forge_relu_fused_into_add(mesh_device, case):
    """
    add + relu collapsed into one quasar.add with a fused RELU activation.

    Inputs are torch.randn (genuinely negative-valued), so the RELU actually clamps and a dropped
    activation fails both the PCC and the explicit non-negativity check rather than sneaking
    through.
    """
    device = mesh_device
    torch.manual_seed(0)

    idx, tag, shape, mem = case
    _require_grid(device, mem)

    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    b_torch = torch.randn(shape, dtype=torch.bfloat16)
    golden = torch.relu(a_torch.float() + b_torch.float())

    a = _to_device(a_torch, mem, device)
    b = _to_device(b_torch, mem, device)

    out = ttnn.experimental.quasar.add(
        a,
        b,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        memory_config=_mem(mem),
        dtype=ttnn.bfloat16,
    )
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out).float()
    assert tuple(out.shape) == shape, "fused add+relu%d output %s, IR says %s" % (idx, tuple(out.shape), shape)
    # a real RELU can never emit a negative; this catches the activation being silently dropped
    assert float(got.min()) >= -1e-2, "fused RELU produced %.4f < 0 -- the activation was not applied" % float(
        got.min()
    )
    assert_with_pcc(golden, got, pcc=PCC)
