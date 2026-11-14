# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The one-shot decode TP all-reduce (GEMMA4_CCL_ONESHOT, tt/ccl.py): the row-parallel partial sums of one row tile,
L1 width-sharded in the DRAM-sharded decode matmul's output layout, summed over the TP group by ONE
``all_reduce_async`` through a scratch allocated around the op instead of the reduce-scatter + all-gather composite.

Gates, on the 12B decode width (H = 3840) in both layouts the 12B row-parallel matmuls write at the mesh's TP (o_proj
and down_proj; at TP=4: 8 cores x 480 and 40 cores x 96), on the 4-chip (1,4) mesh and the (1,8) Galaxy DP4 line:
* the bf16 payload matches the torch sum of the partials at PCC > 0.999 on every chip, and so does the composite
  path's result from the same partials (the two paths agree);
* the bfp8 payload's PCC is recorded (printed as RESULT lines) and gated loosely (> 0.99);
* a one-row logical input (batch 1 under the row view) comes back as one logical row;
* repeated use of one slot (the semaphore pair in turn) and of both slots alternately never hangs and keeps the
  result.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt import ccl as ccl_mod
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allreduce
from models.demos.gemma4.tt.dram_sharded import TILE_SIZE, _decode_core_grid, _padded_n_tiles

H = 3840  # the 12B hidden size
HEADS, HEAD_DIM, INTER = 16, 256, 15360  # the 12B o_proj K (heads x head_dim) and down_proj K (intermediate)
TILE = ttnn.TILE_SIZE
PCC_BF16 = 0.999
PCC_BFP8 = 0.99


def _decode_out_layout(k, n):
    """(cores, shard width) of the DRAM-sharded decode matmul's L1 width-sharded output for a per-chip [k, n] weight,
    as ttnn derives it from ``decode_progcfg``: ``ceil(N_tiles / per_core_N)`` cores, shard ``per_core_N * 32``."""
    _, _, num_cores = _decode_core_grid(k, n)
    per_core_n = math.ceil(n / (TILE_SIZE * num_cores))
    return math.ceil(_padded_n_tiles(n) / per_core_n), per_core_n * TILE_SIZE


def _layouts(tp):
    return {"o": _decode_out_layout(HEADS * HEAD_DIM // tp, H), "down": _decode_out_layout(INTER // tp, H)}


def _sharded_mem(mesh_device, num_cores, shard_w):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (TILE, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _pcc(a, b):
    return float(comp_pcc(a, b, 0.0)[1])


def test_oneshot_knobs(monkeypatch, expect_error):
    """Host-only: the policy default, the env override, the payload dtype and the build-level gate."""
    monkeypatch.delenv("GEMMA4_CCL_ONESHOT", raising=False)
    monkeypatch.delenv("GEMMA4_CCL_ONESHOT_DTYPE", raising=False)
    assert ccl_mod.ccl_oneshot_enabled(mesh_shape=None) is False
    monkeypatch.setattr(ccl_mod, "is_blackhole", lambda: True)
    for shape in ((1, 4), (4, 1), (1, 8), (8, 4), (1, 1)):
        assert ccl_mod.ccl_oneshot_enabled(mesh_shape=shape) == ccl_mod.default_ccl_oneshot(shape)
        assert ccl_mod.default_ccl_oneshot(shape) == (shape in ccl_mod._ONESHOT_DEFAULT_ON_SHAPES)
    assert ccl_mod.default_ccl_oneshot((1, 4)) and not ccl_mod.default_ccl_oneshot((1, 8))
    monkeypatch.setattr(ccl_mod, "is_blackhole", lambda: False)  # Wormhole (T3K is (1,8) too): never by default
    assert not any(ccl_mod.default_ccl_oneshot(s) for s in ((1, 4), (1, 8)))
    monkeypatch.setattr(ccl_mod, "is_blackhole", lambda: True)
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT", "0")
    assert not any(ccl_mod.ccl_oneshot_enabled(mesh_shape=s) for s in ((1, 4), (1, 8)))
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT", "1")
    assert all(ccl_mod.ccl_oneshot_enabled(mesh_shape=s) for s in ((1, 4), (1, 8), None))
    assert ccl_mod.ccl_oneshot_dtype() == ttnn.bfloat16
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT_DTYPE", "bfp8")
    assert ccl_mod.ccl_oneshot_dtype() == ttnn.bfloat8_b
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT_DTYPE", "fp32")
    with expect_error(ValueError, "expected bf16 or bfp8"):
        ccl_mod.ccl_oneshot_dtype()
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT_DTYPE", "bfp8")
    manager = object()
    for tp, shape in ((4, (1, 4)), (8, (1, 8)), (4, (4, 1))):
        cfg = MeshConfig(shape, decode=ModeConfig(tp=tp), tp_axis=0 if shape == (4, 1) else 1)
        assert ccl_mod.oneshot_wanted(cfg, manager), shape
        assert ccl_mod.oneshot_matmul_kwargs(cfg, manager) == {"keep_sharded": True, "dtype": ttnn.bfloat8_b}
    assert not ccl_mod.oneshot_wanted(MeshConfig((1, 4), decode=ModeConfig(tp=4)), None)
    assert not ccl_mod.oneshot_wanted(MeshConfig((1, 1), decode=ModeConfig(tp=1)), manager)
    assert not ccl_mod.oneshot_wanted(MeshConfig((1, 3), decode=ModeConfig(tp=3)), manager), "odd ring size"
    fractured = MeshConfig((8, 4), decode=ModeConfig(tp=8, ep=4), tp_axis=0, weight_fracture=True)
    assert not ccl_mod.oneshot_wanted(fractured, manager)
    lanes = MeshConfig((1, 4), decode=ModeConfig(tp=4))
    lanes.lane_sharded = True
    assert not ccl_mod.oneshot_wanted(lanes, manager)
    assert ccl_mod.oneshot_matmul_kwargs(lanes, manager) == {}
    assert _layouts(4) == {"o": (8, 480), "down": (40, 96)}


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4), (1, 8)])
@pytest.mark.parametrize("payload", ["bf16", "bfp8"])
@pytest.mark.parametrize("rows", [32, 1], ids=["rows32", "rows1"])
def test_oneshot_all_reduce(mesh_device, monkeypatch, payload, rows):
    """Both slots' layouts at the mesh's TP, ``rows`` logical rows of one row tile: one-shot vs the torch sum and vs
    the composite."""
    shape = tuple(int(x) for x in mesh_device.shape)
    if shape not in ((1, 4), (1, 8)):
        pytest.skip(f"the plain-TP line meshes are (1,4) and (1,8); got {shape}")
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT", "1")
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT_DTYPE", payload)
    tp = shape[1]
    cfg = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    mgr = CCLManager(mesh_device)
    assert set(mgr._oneshot_sems) == set(ccl_mod.ONESHOT_SLOTS), "the slot semaphores are created with the manager"
    layouts = _layouts(tp)
    dtype = ttnn.bfloat8_b if payload == "bfp8" else ttnn.bfloat16
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    torch.manual_seed(0)
    per_device = torch.randn(tp, 1, rows, H, dtype=torch.bfloat16)
    ref = per_device.float().sum(0, keepdim=True)  # [1, 1, rows, H]

    def partials(mem, dt):
        return ttnn.from_torch(
            per_device,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dt,
            memory_config=mem,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

    results = {}
    # the slots in the layer's order (o, down, o, down, ...): three rounds exercise the semaphore rotation of each
    for round_idx in range(3):
        for slot, (num_cores, shard_w) in layouts.items():
            mem = _sharded_mem(mesh_device, num_cores, shard_w)
            x = partials(mem, dtype)
            assert ccl_mod.ccl_oneshot_applicable(x, cfg, mgr), f"{slot}: the one-shot gate refused the layout"
            y = ccl_allreduce(x, cfg, mgr, slot=slot, rows=rows)
            assert not y.is_sharded() and y.dtype == ttnn.bfloat16
            assert tuple(int(d) for d in y.shape) == (1, 1, rows, H), y.shape
            out = ttnn.to_torch(y, mesh_composer=composer).float()  # [tp, 1, rows, H]
            y.deallocate(True)
            pcc = min(_pcc(out[d : d + 1], ref) for d in range(tp))
            results[slot] = min(results.get(slot, 1.0), pcc)

    # the composite path (knob off) from the same bf16 partials, and the two paths against each other
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT", "0")
    x = partials(ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
    z = ccl_allreduce(x, cfg, mgr)
    composite = ttnn.to_torch(z, mesh_composer=composer).float()
    z.deallocate(True)
    pcc_composite = min(_pcc(composite[d : d + 1], ref) for d in range(tp))
    monkeypatch.setenv("GEMMA4_CCL_ONESHOT", "1")
    mem = _sharded_mem(mesh_device, *layouts["o"])
    y = ccl_allreduce(partials(mem, dtype), cfg, mgr, slot="o", rows=rows)
    oneshot = ttnn.to_torch(y, mesh_composer=composer).float()
    y.deallocate(True)
    pcc_paths = min(_pcc(oneshot[d : d + 1], composite[d : d + 1]) for d in range(tp))
    ttnn.synchronize_device(mesh_device)
    print(
        f"RESULT oneshot mesh={shape} payload={payload} rows={rows} layouts={layouts} pcc_vs_torch o={results['o']:.6f} "
        f"down={results['down']:.6f} composite_vs_torch={pcc_composite:.6f} oneshot_vs_composite={pcc_paths:.6f}"
    )
    gate = PCC_BF16 if payload == "bf16" else PCC_BFP8
    for slot, pcc in results.items():
        assert pcc > gate, f"{slot} one-shot ({payload}) vs torch sum: PCC {pcc:.6f} <= {gate}"
    assert pcc_composite > PCC_BF16, f"composite vs torch sum: PCC {pcc_composite:.6f}"
    assert pcc_paths > gate, f"one-shot ({payload}) vs composite: PCC {pcc_paths:.6f} <= {gate}"
