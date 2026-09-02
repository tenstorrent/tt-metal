# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for the single ttnn.to_layout the tt-forge ResNet-50 graph issues, run on Quasar via
ttnn.experimental.quasar.to_layout.

  [1,3,224,224]  DRAM INTERLEAVED ROW_MAJOR -> DRAM INTERLEAVED TILE

This is the very first op in the graph: the bf16 input image arrives row-major and is tilized
before the NCHW->NHWC permute. to_layout is value-preserving, so the golden is the input itself
(PCC ~1.0) and the check is that the output really landed in TILE layout.

WHY THIS ONE IS NOT TRIVIAL ON QUASAR
  The tensor is [1, 3, 224, 224] -- the tilized dims are the last two (224x224, both multiples of
  32, so no padding is needed), but the tensor is rank 4 with a leading 3, so it tilizes 3 separate
  224x224 planes. The Quasar tilize is a ported kernel (fast_tilize is unported; see
  ../ops/test_tilize_width_quasar.py and ../ops/test_tilize_wh_control.py), so this exercises the
  plain tilize_block path on a batched rank-4 input.

The reverse direction (TILE -> ROW_MAJOR) is NOT in the Forge graph and is not tested here;
../ops/test_to_layout.py covers both directions on the hand-written model's shapes.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_to_layout_forge.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.999  # pure layout change -- values must round-trip

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
TL_IN = ("INTERLEAVED", "DRAM", None, None, "ROW_MAJOR")
TL_OUT = ("INTERLEAVED", "DRAM", None, None, "TILE")

INPUT_SHAPE = (1, 3, 224, 224)


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


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_to_layout(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    assert _page(TL_IN) != _page(TL_OUT), "the Forge to_layout case is a no-op?"

    x_torch = torch.rand(INPUT_SHAPE, dtype=torch.bfloat16)
    x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=_page(TL_IN)).to(device, _mem(TL_IN))

    out = ttnn.experimental.quasar.to_layout(x, _page(TL_OUT), memory_config=_mem(TL_OUT))
    ttnn.synchronize_device(device)

    assert out.layout == _page(TL_OUT), "to_layout landed in %s but Forge asked for %s" % (out.layout, TL_OUT[4])
    assert tuple(out.shape) == INPUT_SHAPE, "to_layout changed shape: %s -> %s" % (INPUT_SHAPE, tuple(out.shape))
    assert out.memory_config().memory_layout == getattr(ttnn.TensorMemoryLayout, TL_OUT[0])

    assert_with_pcc(x_torch, ttnn.to_torch(out).to(torch.bfloat16), pcc=PCC)
