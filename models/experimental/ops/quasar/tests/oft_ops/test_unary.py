# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: unary SFPU ops the OFT encoder decode ran (``eltwise_sfpu`` variants).

Shape source (kernel cache): each op has two ``eltwise_sfpu`` variants whose ct[0]
(tiles per core) is 2 and 1 — 25 tiles split over 20 cores, i.e. a [1, 159, 159] bf16 TILE
tensor — except ``exp`` with 4 and 3 tiles per core (75 tiles = [1, 3, 159, 159]). All bf16,
L1 interleaved in and out (unary reader/writer accessors), HiFi4, approx off.         [H]

  * ``exp``: dims decode ``exp(dim_offsets * std + mean)``                     (126815 / 126361)
  * ``reciprocal``, ``atan``, ``gtz``, ``ltz``, ``eqz``: the pieces of the composite
    ``ttnn.atan2`` (covered as a whole in ``test_atan2.py``)
    (recip 183849 / 171905, atan 143616 / 751423, gtz 183839 / 126521,
     ltz 885732 / 863660, eqz 805151 / 989732)

There is no standalone ``relu`` kernel in cache #1: every ReLU was fused into conv,
matmul or binary ops.

Second cache (in-tree OFT, unary_ng factory of the fork, ~``628a4797bf7``): one ``eltwise_sfpu``
``relu_tile`` compute variant (105908; tile counts are runtime args) and ten
``reader_unary_ng`` / ``writer_unary_ng`` pairs, one per memory config. Their TensorAccessor
args give the page size (2048 B = bf16 tile, 512 / 128 B = bf16 ROW_MAJOR row of 256 / 64
channels), bank count and shard shape in pages. All bf16, HiFi4. The relus are the standalone
``ttnn.relu`` calls of ``tt_resnet.py`` / ``tt_oftnet.py`` (in place via ``output_tensor=`` after
each BasicBlock's bn1 and the stem concat; out of place after the residual add and the lateral
GroupNorms):                                                                                [H]

  * stem concat slice HS (3072, 64) 20 cores TILE, in place                       (159604)
  * layer1 / layer2 / layer3 HS (1536, 64) / (384, 128) / (96, 256) TILE           (145689 / 150488 / 789612)
  * layer4 BS (96, 128) COL_MAJOR 5x4 TILE (bn1, in place); HS (32, 512) on 15 cores (after the add)
                                                                                    (224891 / 110654)
  * lateral bn8 / bn16 HS (384, 256) / (96, 256) ROW_MAJOR; bn32 BS (96, 64) ROW_MAJOR (105769 / 180653 / 945826)
  * topdown bn1 output [1, 1, 25344, 256] TILE DRAM interleaved, in place          (302917) [H mem / M shape]

The 11th reader variant (150910, L1 interleaved TILE) is not mapped to a relu: it most likely
serves the encoder's ``exp`` (covered above).
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

DIMS = (1, 3, U.BEV, U.BEV)


def _away_from_zero(x):
    return torch.where(x.abs() < 0.1, torch.full_like(x, 0.5), x)


def _with_zeros(x):
    x = x.clone()
    x.view(-1)[::7] = 0  # comparison-to-zero ops need exact zeros in the input
    return x


# (op name, ttnn op, torch ref, input transform, exact?)
_OPS = {
    "exp": (ttnn.exp, torch.exp, lambda x: x * 0.5, False),
    "reciprocal": (ttnn.reciprocal, torch.reciprocal, _away_from_zero, False),
    "atan": (ttnn.atan, torch.atan, None, False),
    "gtz": (ttnn.gtz, lambda x: (x > 0).float(), _with_zeros, True),
    "ltz": (ttnn.ltz, lambda x: (x < 0).float(), _with_zeros, True),
    "eqz": (ttnn.eqz, lambda x: (x == 0).float(), _with_zeros, True),
}


@U.with_default_mesh()
@pytest.mark.parametrize(
    "op, shape",
    [
        pytest.param("exp", DIMS, id="exp-1x3x159x159-126815"),
        pytest.param("reciprocal", U.HEATMAP, id="reciprocal-1x159x159-183849"),
        pytest.param("atan", U.HEATMAP, id="atan-1x159x159-143616"),
        pytest.param("gtz", U.HEATMAP, id="gtz-1x159x159-183839"),
        pytest.param("ltz", U.HEATMAP, id="ltz-1x159x159-885732"),
        pytest.param("eqz", U.HEATMAP, id="eqz-1x159x159-805151"),
    ],
)
def test_unary(ttnn_mesh_device, reset_seeds, op, shape):
    mesh = ttnn_mesh_device
    tt_op, torch_op, transform, exact = _OPS[op]
    x = U.torch_rand(shape)
    if transform is not None:
        x = transform(x)
    tt_x = U.to_tt_mem(x, mesh, "l1")
    out = tt_op(tt_x, memory_config=ttnn.L1_MEMORY_CONFIG)
    ref = torch_op(x.float())
    if exact:
        U.assert_lossless(ref, out, mesh_device=mesh)
    else:
        U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)


# fmt: off
# (shape, memory spec, row-major?, in place?)  spec: "dram" | (kind, shard, (grid_x, grid_y))
RELU_CASES = [
    pytest.param((1, 1, 61440, 64), ("hs", (3072, 64), (5, 4)), False, True, id="relu_-stem-concat-hs-3072x64-159604"),
    pytest.param((1, 1, 30720, 64), ("hs", (1536, 64), (5, 4)), False, True, id="relu_-layer1-hs-1536x64-145689"),
    pytest.param((1, 1, 7680, 128), ("hs", (384, 128), (5, 4)), False, True, id="relu_-layer2-hs-384x128-150488"),
    pytest.param((1, 1, 1920, 256), ("hs", (96, 256), (5, 4)), False, False, id="relu-layer3-hs-96x256-789612"),
    pytest.param((1, 1, 480, 512), ("bs", (96, 128), (5, 4)), False, True, id="relu_-layer4-bs-96x128-224891"),
    pytest.param((1, 1, 480, 512), ("hs", (32, 512), (5, 3)), False, False, id="relu-layer4-hs15-32x512-110654"),
    pytest.param((1, 1, 7680, 256), ("hs", (384, 256), (5, 4)), True, False, id="relu-bn8-rm-hs-384x256-105769"),
    pytest.param((1, 1, 1920, 256), ("hs", (96, 256), (5, 4)), True, False, id="relu-bn16-rm-hs-96x256-180653"),
    pytest.param((1, 1, 480, 256), ("bs", (96, 64), (5, 4)), True, False, id="relu-bn32-rm-bs-96x64-945826"),
    pytest.param((1, 1, 25344, 256), "dram", False, True, id="relu_-topdown-dram-302917"),
]
RELU_PROXY_CASES = [
    pytest.param((1, 1, 128, 64), ("hs", (64, 64), (2, 1)), False, True, id="proxy-relu_-hs-tile"),
    pytest.param((1, 1, 128, 256), ("hs", (64, 256), (2, 1)), True, False, id="proxy-relu-hs-rm"),
    pytest.param((1, 1, 64, 128), ("bs", (64, 64), (1, 2)), True, False, id="proxy-relu-bs-rm"),
    pytest.param((1, 1, 1024, 256), "dram", False, True, id="proxy-relu_-dram"),
]
# fmt: on


def _relu_memcfg(mesh, spec):
    """HS ROW_MAJOR or BS COL_MAJOR (the model's GroupNorm block sharding) over a grid_x x grid_y grid."""
    kind, shard, grid = spec
    cores = U.model_grid(mesh, *grid)
    if kind == "hs":
        shard_spec = ttnn.ShardSpec(cores, list(shard), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    shard_spec = ttnn.ShardSpec(cores, list(shard), ttnn.ShardOrientation.COL_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


def _run_relu(mesh, shape, spec, rm, inplace):
    x = U.torch_rand(shape)
    layout = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
    tt_x = U.to_tt(x, mesh, layout=layout)
    if spec != "dram":
        tt_x = U.op("to_memory_config")(tt_x, _relu_memcfg(mesh, spec))
    if inplace:
        ttnn.relu(tt_x, output_tensor=tt_x)  # ``ttnn.relu(out, output_tensor=out)``
        out = tt_x
    else:
        out = ttnn.relu(tt_x)
        assert out.is_sharded() == tt_x.is_sharded(), "relu changed the memory layout"
    U.assert_lossless(torch.relu(x.float()), out, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape, spec, rm, inplace", RELU_CASES)
def test_relu_oft2(ttnn_mesh_device, reset_seeds, shape, spec, rm, inplace):
    _run_relu(ttnn_mesh_device, shape, spec, rm, inplace)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, spec, rm, inplace", RELU_PROXY_CASES)
def test_relu_oft2_small(ttnn_mesh_device, reset_seeds, shape, spec, rm, inplace):
    _run_relu(ttnn_mesh_device, shape, spec, rm, inplace)
