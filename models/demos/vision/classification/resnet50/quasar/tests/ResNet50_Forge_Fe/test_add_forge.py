# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-call-site PCC test for every ttnn.add the tt-forge ResNet-50 graph issues, run on Quasar via
ttnn.experimental.quasar.add.

16 cases -- the residual add at the end of each of the 16 bottlenecks. Shapes and memory configs
are exactly the [3, 4, 6, 3] bottleneck outputs:

    layer1  3 x [1,1,3136, 256]  HEIGHT_SHARDED, 49 cores (8x6 + 1), shard [ 64,  256]
    layer2  4 x [1,1, 784, 512]  HEIGHT_SHARDED, 25 cores (8x3 + 1), shard [ 32,  512]
    layer3  6 x [1,1, 196,1024]  BLOCK_SHARDED,  56 cores (8x7),     shard [ 32,  128]
    layer4  3 x [1,1,  49,2048]  BLOCK_SHARDED,  16 cores (8x2),     shard [ 32,  256]

NOT FUSED HERE. The Forge graph emits a bare ttnn.add followed by a SEPARATE ttnn.relu (16 of
each); it does not fuse the activation. So this test issues a plain add and the golden is a + b
with no clamp. The fused form -- which is how the hand-written metal quasar model does it, and the
only route available on Quasar since there is no standalone quasar relu -- is exercised in
test_relu_forge.py::test_forge_relu_fused_into_add.

Both operands carry the SAME memory config in every case (the two input layout aliases are
identical for all 16 adds), so no reshard is needed before the add. Only 4 of the 16 configs are
distinct -- one per layer.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_add_forge.py
    pytest -s ... test_add_forge.py -k "not dup"     # the 4 distinct configs
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99  # elementwise bf16 add is near-exact

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

# (idx, tag,          shape,               operand_and_output_mem)
# Forge gives both operands and the output the same memory config in every case.
ADD_CASES = [
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
    """Short memory-config tag for a test id, e.g. 'HS49' / 'BS16'."""
    kind = {"HEIGHT_SHARDED": "HS", "BLOCK_SHARDED": "BS", "WIDTH_SHARDED": "WS"}[spec[0]]
    cores = sum((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) for lo, hi in spec[2])
    return "%s%d" % (kind, cores)


def _id(case):
    idx, tag, shape, mem = case
    first = next(c[0] for c in ADD_CASES if c[3] is mem)
    return "%02d_%s_%dx%d_%s%s" % (
        idx,
        tag.replace("layer", "L").replace(".", "_"),
        shape[2],
        shape[3],
        _mem_tag(mem),
        "_dup%d" % first if first != idx else "",
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("case", ADD_CASES, ids=[_id(c) for c in ADD_CASES])
def test_forge_add(mesh_device, case):
    device = mesh_device
    torch.manual_seed(0)

    idx, tag, shape, mem = case
    _require_grid(device, mem)

    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    b_torch = torch.randn(shape, dtype=torch.bfloat16)
    golden = a_torch.float() + b_torch.float()

    a = _to_device(a_torch, mem, device)
    b = _to_device(b_torch, mem, device)

    out = ttnn.experimental.quasar.add(a, b, memory_config=_mem(mem), dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == shape, "add%d (%s) output %s, Forge IR says %s" % (idx, tag, tuple(out.shape), shape)
    got_layout = out.memory_config().memory_layout
    assert got_layout == getattr(ttnn.TensorMemoryLayout, mem[0]), "add%d landed in %s but Forge asked for %s" % (
        idx,
        got_layout,
        mem[0],
    )

    assert_with_pcc(golden, ttnn.to_torch(out).float(), pcc=PCC)
