# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: the BasicBlock residual add of the in-tree OFT ResNet / topdown (``tt_resnet.py``):
``out += x`` (``ttnn.add(out, x)`` with the output in ``out``'s memory config), or for layer4
(``gn_shard="BS"``) ``ttnn.add(out, x, memory_config=HS15)`` after resharding both operands to
``create_sharded_memory_config([32, 512], CoreGrid(y=3, x=5), HEIGHT, ROW_MAJOR)``. The ReLU that
follows is a separate ``ttnn.relu`` (see ``test_unary.py``), so these adds have no fused
activation, unlike cache #1's residual adds in ``test_binary.py`` (which all carry PACK_RELU).

Shape source: second kernel cache (in-tree OFT, binary_ng factory at ``628a4797bf7``). Compute
``eltwise_binary_no_bcast`` ADD, no PROCESS_*_ACTIVATIONS, HiFi4; reader / writer TensorAccessor
args give the sharding (20 or 15 banks, shard volume in tiles) or DRAM interleaving:

  * layer1.0: a bf16 HEIGHT_SHARDED 96 tiles = (1536, 64), b **bfloat8_b DRAM interleaved** (the
    max-pool output moved to DRAM), out sharded like a (reader 128633, compute 119192)          [H]
  * layer1 (1536, 64) / layer2 (384, 128) / layer3 (96, 256): bf16 a, b, out HEIGHT_SHARDED on
    20 cores (readers 158813 / 173286 / 113314)                                                [H]
  * layer4: bf16 HEIGHT_SHARDED 16 tiles on 15 cores = (32, 512) (reader 165439)               [H]
  * topdown: bf16 TILE DRAM + DRAM -> DRAM (reader 233744); [1, 1, 25344, 256], the GroupNormDRAM
    output tilized to 25344 rows                                                             [H mem / M shape]

Already covered in ``test_binary.py`` and not repeated: bf16+bfp8 both sharded (legacy path),
float32 layer2, and the 128-channel topdown HS + DRAM add, all with fused ReLU.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U


def _memcfg(mesh, spec):
    """ "dram" / "l1" / ("hs", shard, (grid_x, grid_y)) -> MemoryConfig (HS is ROW_MAJOR, like the model)."""
    if spec in ("dram", "l1"):
        return U.resolve_memcfg(spec, mesh)
    _, shard, grid = spec
    shard_spec = ttnn.ShardSpec(U.model_grid(mesh, *grid), list(shard), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _upload(x, mesh, spec, dt):
    tt = U.to_tt(x, mesh, dtype=U.DTYPES[dt])
    return tt if spec == "dram" else U.op("to_memory_config")(tt, _memcfg(mesh, spec))


def _run(mesh, shape, a_dt, b_dt, a_mem, b_mem, out_mem):
    a = U.torch_rand(shape)
    b = U.torch_rand(shape)
    tt_a = _upload(a, mesh, a_mem, a_dt)
    tt_b = _upload(b, mesh, b_mem, b_dt)
    kwargs = {} if out_mem is None else {"memory_config": _memcfg(mesh, out_mem)}
    out = U.op("add")(tt_a, tt_b, **kwargs)
    if out_mem is None:  # ``out += x`` keeps ``out``'s memory config
        assert out.is_sharded() == tt_a.is_sharded(), "add did not keep the input memory layout"
    ref = U.seen(tt_a, a, mesh) + U.seen(tt_b, b, mesh)  # bfloat8_b quantization included
    U.assert_pcc(ref, out, pcc=0.99 if "bfp8" in (a_dt, b_dt) else 0.999, mesh_device=mesh)


G5x4, G5x3 = (5, 4), (5, 3)
_ARGS = "shape, a_dt, b_dt, a_mem, b_mem, out_mem"

# fmt: off
CASES = [
    pytest.param((1, 1, 30720, 64), "bf16", "bfp8", ("hs", (1536, 64), G5x4), "dram", None, id="layer1.0-hs+bfp8dram-128633"),
    pytest.param((1, 1, 30720, 64), "bf16", "bf16", ("hs", (1536, 64), G5x4), ("hs", (1536, 64), G5x4), None, id="layer1-hs-1536x64-158813"),
    pytest.param((1, 1, 7680, 128), "bf16", "bf16", ("hs", (384, 128), G5x4), ("hs", (384, 128), G5x4), None, id="layer2-bf16-hs-384x128-173286"),
    pytest.param((1, 1, 1920, 256), "bf16", "bf16", ("hs", (96, 256), G5x4), ("hs", (96, 256), G5x4), None, id="layer3-hs-96x256-113314"),
    pytest.param((1, 1, 480, 512), "bf16", "bf16", ("hs", (32, 512), G5x3), ("hs", (32, 512), G5x3), ("hs", (32, 512), G5x3), id="layer4-hs15-32x512-165439"),
    pytest.param((1, 1, 25344, 256), "bf16", "bf16", "dram", "dram", None, id="topdown-dram+dram-233744"),
]

PROXY_CASES = [
    pytest.param((1, 1, 128, 64), "bf16", "bfp8", ("hs", (64, 64), (2, 1)), "dram", None, id="proxy-hs+bfp8dram"),
    pytest.param((1, 1, 128, 128), "bf16", "bf16", ("hs", (64, 128), (2, 1)), ("hs", (64, 128), (2, 1)), None, id="proxy-hs+hs"),
    pytest.param((1, 1, 1024, 256), "bf16", "bf16", "dram", "dram", None, id="proxy-dram+dram"),
]
# fmt: on


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(_ARGS, CASES)
def test_resnet_residual_add(ttnn_mesh_device, reset_seeds, shape, a_dt, b_dt, a_mem, b_mem, out_mem):
    _run(ttnn_mesh_device, shape, a_dt, b_dt, a_mem, b_mem, out_mem)


@U.with_default_mesh()
@pytest.mark.parametrize(_ARGS, PROXY_CASES)
def test_resnet_residual_add_small(ttnn_mesh_device, reset_seeds, shape, a_dt, b_dt, a_mem, b_mem, out_mem):
    _run(ttnn_mesh_device, shape, a_dt, b_dt, a_mem, b_mem, out_mem)
