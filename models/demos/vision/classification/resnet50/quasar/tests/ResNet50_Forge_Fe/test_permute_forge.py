# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.permute -- the tt-forge ResNet-50 graph issues one, and ttnn.experimental.quasar has NO
GENERIC PERMUTE. This file makes that concrete and runs the route that does work.

THE CALL-SITE
  permute([0, 2, 3, 1])   [1,3,224,224] DRAM INTERLEAVED TILE
                       -> [1,224,224,3] L1   INTERLEAVED TILE,  pad_value 0.0
  i.e. NCHW -> NHWC on the tilized input image, feeding the reshape that flattens it to
  [1, 1, 50176, 3] for the stem conv. This is the only permute in the graph.

THE GAP
  The Quasar namespace binds `transpose` -- a TWO-AXIS SWAP, transpose(input, dim1, dim2) -- and
  nothing more general. test_op_inventory.py prints the live list, so this is checked against the
  build rather than asserted from memory.

  [0,2,3,1] is a 3-cycle on axes (1,2,3), so it cannot be expressed as one transpose. It DOES
  decompose into exactly two:
        (N,C,H,W) --transpose(1,2)--> (N,H,C,W) --transpose(2,3)--> (N,H,W,C)
  which is what test_forge_permute_via_transpose runs. If Forge is to target Quasar, that
  decomposition (or a real quasar permute) is what the compiler has to emit.

WHAT THIS FILE RUNS
  test_forge_permute_standalone     resolves ttnn.experimental.quasar.permute; xfails with the gap
                                    named while there is none, and exercises it for real if one
                                    lands.
  test_forge_permute_via_transpose  the 2-transpose decomposition, PCC-checked against the torch
                                    permute, with the intermediate shape asserted so a wrong-axis
                                    transpose cannot pass by accident.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_permute_forge.py
    pytest -s ... test_permute_forge.py -k transpose     # only the workaround that can pass
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.999  # pure data movement -- values must round-trip

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
PERMUTE_IN = ("INTERLEAVED", "DRAM", None, None, "TILE")
PERMUTE_OUT = ("INTERLEAVED", "L1", None, None, "TILE")

INPUT_SHAPE = (1, 3, 224, 224)
PERMUTATION = (0, 2, 3, 1)
OUTPUT_SHAPE = (1, 224, 224, 3)


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
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=_page(spec)).to(device, _mem(spec))


# --------------------------------------------------------------------------------------------------
# 1. the gap itself
# --------------------------------------------------------------------------------------------------
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_permute_standalone(mesh_device):
    """
    Forge emits ttnn.permute([0,2,3,1]) here. xfails while ttnn.experimental.quasar has no permute
    -- which is every build today. If one lands, this starts exercising it for real.
    """
    device = mesh_device
    torch.manual_seed(0)

    permute = getattr(ttnn.experimental.quasar, "permute", None)
    if permute is None:
        exposed = sorted(n for n in dir(ttnn.experimental.quasar) if "transpose" in n or "permute" in n)
        pytest.xfail(
            "NOT EXPOSED: ttnn.experimental.quasar.permute does not exist -- the namespace binds "
            "only %s, a two-axis swap. [0,2,3,1] is a 3-cycle, so it needs two of them; that route "
            "is test_forge_permute_via_transpose." % (exposed or ["nothing of that shape"])
        )

    x_torch = torch.rand(INPUT_SHAPE, dtype=torch.bfloat16)
    golden = x_torch.permute(*PERMUTATION).contiguous()

    x = _to_device(x_torch, PERMUTE_IN, device)
    out = permute(x, list(PERMUTATION), memory_config=_mem(PERMUTE_OUT))
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == OUTPUT_SHAPE, "permute output %s, Forge IR says %s" % (tuple(out.shape), OUTPUT_SHAPE)
    assert_with_pcc(golden, ttnn.to_torch(out).to(torch.bfloat16), pcc=PCC)


# --------------------------------------------------------------------------------------------------
# 2. the route that exists: two transposes
# --------------------------------------------------------------------------------------------------
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_permute_via_transpose(mesh_device):
    """
    NCHW -> NHWC as quasar.transpose(1, 2) then quasar.transpose(2, 3).

    Only the FINAL memory config is pinned to Forge's (L1 interleaved); the intermediate is left to
    the op, because Forge never materialises one -- it emits a single permute.
    """
    device = mesh_device
    torch.manual_seed(0)

    n, c, h, w = INPUT_SHAPE
    x_torch = torch.rand(INPUT_SHAPE, dtype=torch.bfloat16)
    golden = x_torch.permute(*PERMUTATION).contiguous()

    x = _to_device(x_torch, PERMUTE_IN, device)

    # (N,C,H,W) -> (N,H,C,W)
    mid = ttnn.experimental.quasar.transpose(x, 1, 2)
    assert tuple(mid.shape) == (n, h, c, w), "transpose(1,2) gave %s, expected %s" % (tuple(mid.shape), (n, h, c, w))

    # (N,H,C,W) -> (N,H,W,C)
    out = ttnn.experimental.quasar.transpose(mid, 2, 3, memory_config=_mem(PERMUTE_OUT))
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == OUTPUT_SHAPE, "2-transpose permute gave %s, Forge IR says %s" % (
        tuple(out.shape),
        OUTPUT_SHAPE,
    )
    assert_with_pcc(golden, ttnn.to_torch(out).to(torch.bfloat16), pcc=PCC)
