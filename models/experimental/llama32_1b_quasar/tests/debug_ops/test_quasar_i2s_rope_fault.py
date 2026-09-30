# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar interleaved_to_sharded (i2s) fault seen during llama32_1b decode.

During llama32_1b decode on the Quasar simulator, RoPE cos/sin sharding
(models/experimental/llama32_1b_quasar/modules/rope/rope_1d.py:189-190):

    cos = ttnn.interleaved_to_sharded(cos, cfg.cos_sin_shard_mem_config)

tripped a watcher assert:

    Device 0 worker core(x= 1,y= 0): DM1 hardware fault at PC 0x23c. Cause: UNALIGNED_LOAD,
    faulting address 0x04a46b43. Current kernel: blank.

i.e. the single-core i2s (shard core (0,0)) issued a misaligned / stray NOC access that corrupted a
*neighbouring* core (1,0), whose blank kernel then faulted on an unaligned load. The exact op from the
device log:

    InterleavedToShardedDeviceOperation
      input  : [1, 1, 1, 64] bf16 TILE, DRAM INTERLEAVED   (decode: seq=1, head_dim=64)
      output : HEIGHT_SHARDED L1, grid [(0,0)..(0,0)], shard [32, 64], ROW_MAJOR, keep_l1_aligned=false

[1,1,1,64] is degenerate: logical height 1, padded to a single 32x32 tile. This reproduces that op in
isolation so i2s can be debugged without the whole model. Two entry points are exercised for the same
config:

  * test_i2s_mainline        -> ttnn.interleaved_to_sharded            (the op that faulted)
  * test_i2s_quasar          -> ttnn.experimental.quasar.interleaved_to_sharded  (Gen2-native op)

A non-degenerate [1,1,32,64] is also parametrized to check whether the height-1 padding is the trigger
(if the tall case passes and the height-1 case faults, the bug is in the degenerate-tile address math).

Run on the sim as the model does, e.g.:

    TT_METAL_WATCHER=12 MESH_DEVICE=N150 TT_METAL_SIMULATOR=~/sim/libttsim.so \
        pytest tests/ttnn/unit_tests/operations/test_quasar_i2s_rope_fault.py
"""

import pytest
import torch
from loguru import logger

import ttnn


def _readback(tt, mesh_device):
    try:
        num = mesh_device.get_num_devices()
    except Exception:
        num = 1
    if num > 1:
        return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return ttnn.to_torch(tt)


def _tile_dram_interleaved(t_bf16, mesh_device):
    """Build a bf16 TILE, DRAM-interleaved tensor WITHOUT the mainline from_torch(TILE) tilize (that
    mainline device Tilize deadlocks on the Quasar sim). Upload row-major, then tilize with the
    Gen2-native quasar op where available; fall back to mainline tilize (WH/BH). The torch tensor's
    height MUST be tile-aligned (>=32, %32==0) -- tilize rejects a partial-height row-major input."""
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # no device tilize on upload
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    try:
        return ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    except (AttributeError, RuntimeError) as e:
        logger.info(f"[i2s-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _tile_input_logical_h(h, w, mesh_device):
    """Produce the i2s input the way the model does: a TILE tensor whose LOGICAL height is `h` but whose
    PADDED height is a full 32-tall tile. A row-major height-h (<32) tensor cannot be tilized (tilize
    needs height %32==0), so build a full [1,1,32,w] tile then ``ttnn.slice`` to logical [1,1,h,w] -- this
    is exactly how the model's RoPE cos/sin [1,1,1,64] i2s input is formed (slice of a taller TILE
    tensor). Returns (tt_tensor, torch_ref) where torch_ref is the first h rows.

    Returns (None, None) if the slice itself faults/raises, so the caller can surface that separately from
    the i2s under test.
    """
    t_full = torch.randn(1, 1, 32, w, dtype=torch.bfloat16)
    x_full = _tile_dram_interleaved(t_full, mesh_device)
    if h == 32:
        return x_full, t_full
    x = ttnn.slice(x_full, [0, 0, 0, 0], [1, 1, h, w])  # logical [1,1,h,w], padded tile stays 32 tall
    return x, t_full[:, :, :h, :]


def _single_core_height_sharded_memcfg(shard_h, shard_w):
    """HEIGHT_SHARDED L1, single core (0,0), shard [shard_h, shard_w], ROW_MAJOR — exactly the config the
    faulting RoPE cos/sin i2s used (cos_sin_shard_mem_config)."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        (shard_h, shard_w),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


# (logical H, W). head_dim=64. The first row is the exact decode case that faulted; the second is the
# non-degenerate control (a full tile tall).
_SHAPES = [
    (1, 64),  # decode: seq=1, head_dim=64 -> degenerate height-1 tile (THE faulting case)
    (32, 64),  # non-degenerate control: a full 32-row tile
]


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_i2s_mainline(mesh_device, shape):
    """ttnn.interleaved_to_sharded — the exact op that tripped the watcher UNALIGNED_LOAD."""
    h, w = shape
    torch.manual_seed(0)

    logger.info(f"[i2s-repro][mainline] build input logical (1,1,{h},{w}) bf16 TILE DRAM (padded 32 tall)")
    x, t = _tile_input_logical_h(h, w, mesh_device)
    memcfg = _single_core_height_sharded_memcfg(32, w)  # shard height padded to a tile (32), as the model did

    logger.info(f"[i2s-repro][mainline] ttnn.interleaved_to_sharded (1,1,{h},{w}) -> HS L1 [32,{w}] core (0,0)")
    xs = ttnn.interleaved_to_sharded(x, memcfg)
    logger.info("[i2s-repro][mainline] i2s returned; reading back")
    out = _readback(xs, mesh_device).float()
    logger.info("[i2s-repro][mainline] readback complete")

    ref = t.float()
    assert torch.isfinite(out).all(), "non-finite after mainline i2s"
    assert torch.allclose(out.reshape(ref.shape), ref, atol=0.05, rtol=0.05), "value mismatch after mainline i2s"


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_i2s_quasar(mesh_device, shape):
    """ttnn.experimental.quasar.interleaved_to_sharded — same config via the Gen2-native op."""
    h, w = shape
    torch.manual_seed(0)

    logger.info(f"[i2s-repro][quasar] build input logical (1,1,{h},{w}) bf16 TILE DRAM (padded 32 tall)")
    x, t = _tile_input_logical_h(h, w, mesh_device)
    memcfg = _single_core_height_sharded_memcfg(32, w)

    logger.info(f"[i2s-repro][quasar] quasar.interleaved_to_sharded (1,1,{h},{w}) -> HS L1 [32,{w}] core (0,0)")
    xs = ttnn.experimental.quasar.interleaved_to_sharded(x, memcfg)
    logger.info("[i2s-repro][quasar] i2s returned; reading back")
    out = _readback(xs, mesh_device).float()
    logger.info("[i2s-repro][quasar] readback complete")

    ref = t.float()
    assert torch.isfinite(out).all(), "non-finite after quasar i2s"
    assert torch.allclose(out.reshape(ref.shape), ref, atol=0.05, rtol=0.05), "value mismatch after quasar i2s"


# ---------------------------------------------------------------------------------------------------
# Address-shifting variants.
#
# The degenerate [1,1,1,64] i2s PASSES in isolation, but the SAME op tripped an UNALIGNED_LOAD in the
# model. The fault was a STRAY NOC write that corrupted a *neighbour* core (1,0) of the shard core (0,0),
# so whether it is destructive depends on the i2s input's actual DRAM address and the L1/NOC geometry.
# These tests deliberately vary that placement -- pre-allocated DRAM spacers, fragmentation, the output
# shard core, and allocation churn across a loop -- to try to land on the destructive address the model
# hit. A failure (fault / value mismatch) here is a REPRO; all-pass means the fault needs still more of
# the model's specific allocator state.
# ---------------------------------------------------------------------------------------------------


def _get_i2s(op):
    if op == "mainline":
        return ttnn.interleaved_to_sharded
    fn = getattr(getattr(ttnn.experimental, "quasar", None), "interleaved_to_sharded", None)
    if fn is None:
        pytest.skip("ttnn.experimental.quasar.interleaved_to_sharded not available")
    return fn


def _alloc_dram_spacers(mesh_device, count, cols=4096):
    """Allocate `count` bf16 DRAM row-major tensors and return them (caller MUST keep the list alive).
    Each live spacer pushes the subsequently-allocated i2s input to a different DRAM address."""
    spacers = []
    for _ in range(count):
        spacers.append(
            ttnn.from_torch(
                torch.zeros(1, 1, 32, cols, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
            )
        )
    return spacers


def _check(out, t, tag):
    assert torch.isfinite(out).all(), f"non-finite after i2s ({tag})"
    assert torch.allclose(out.reshape(t.float().shape), t.float(), atol=0.05, rtol=0.05), f"value mismatch ({tag})"


@pytest.mark.parametrize("op", ["mainline", "quasar"])
@pytest.mark.parametrize("prealloc", [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
def test_i2s_addr_shift(mesh_device, op, prealloc):
    """Push the degenerate [1,1,1,64] i2s input to a different DRAM address by holding `prealloc` DRAM
    spacers alive, then run i2s. Sweeps the input address to hunt the model's destructive placement."""
    w = 64
    i2s = _get_i2s(op)
    spacers = _alloc_dram_spacers(mesh_device, prealloc)  # held alive for the whole test
    x, t = _tile_input_logical_h(1, w, mesh_device)
    memcfg = _single_core_height_sharded_memcfg(32, w)
    logger.info(f"[i2s-repro][{op}] addr-shift prealloc={prealloc}")
    xs = i2s(x, memcfg)
    out = _readback(xs, mesh_device).float()
    _check(out, t, f"{op} prealloc={prealloc}")
    del spacers


@pytest.mark.parametrize("op", ["mainline", "quasar"])
@pytest.mark.parametrize("cols", [1024, 2048, 4096])
def test_i2s_frag(mesh_device, op, cols):
    """Fragment DRAM (allocate 32 spacers, free every other one) so the i2s input lands in a hole at an
    unusual address, then run the degenerate i2s."""
    w = 64
    i2s = _get_i2s(op)
    spacers = _alloc_dram_spacers(mesh_device, 32, cols=cols)
    for i in range(0, len(spacers), 2):
        ttnn.deallocate(spacers[i])
    kept = [s for i, s in enumerate(spacers) if i % 2 == 1]  # hold the odd ones alive -> holes between them
    x, t = _tile_input_logical_h(1, w, mesh_device)
    memcfg = _single_core_height_sharded_memcfg(32, w)
    logger.info(f"[i2s-repro][{op}] fragmentation cols={cols}")
    xs = i2s(x, memcfg)
    out = _readback(xs, mesh_device).float()
    _check(out, t, f"{op} frag cols={cols}")
    del kept


@pytest.mark.parametrize("op", ["mainline", "quasar"])
@pytest.mark.parametrize("core", [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (3, 1)])
def test_i2s_shard_core(mesh_device, op, core):
    """Vary the OUTPUT shard core. The model fault was a stray NOC write to a NEIGHBOUR of the shard core,
    so a different shard core changes the NOC geometry and which neighbour a stray write would hit."""
    cx, cy = core
    grid = mesh_device.compute_with_storage_grid_size()
    if cx >= grid.x or cy >= grid.y:
        pytest.skip(f"core {core} outside compute grid {grid.x}x{grid.y}")
    w = 64
    i2s = _get_i2s(op)
    x, t = _tile_input_logical_h(1, w, mesh_device)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy))}),
        (32, w),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    logger.info(f"[i2s-repro][{op}] shard core=({cx},{cy})")
    xs = i2s(x, memcfg)
    out = _readback(xs, mesh_device).float()
    _check(out, t, f"{op} core=({cx},{cy})")


@pytest.mark.parametrize("op", ["mainline", "quasar"])
def test_i2s_alloc_churn(mesh_device, op):
    """Run the degenerate i2s many times with churning DRAM allocations between iterations, so the input
    lands at many different addresses within one test -- catches an intermittent address-dependent fault
    that a single placement misses."""
    w = 64
    i2s = _get_i2s(op)
    memcfg = _single_core_height_sharded_memcfg(32, w)
    for it in range(40):
        junk = _alloc_dram_spacers(mesh_device, it % 7, cols=512 + 128 * (it % 5))
        x, t = _tile_input_logical_h(1, w, mesh_device)
        logger.info(f"[i2s-repro][{op}] churn iter={it} junk={len(junk)}")
        xs = i2s(x, memcfg)
        out = _readback(xs, mesh_device).float()
        _check(out, t, f"{op} churn iter={it}")
        ttnn.deallocate(xs)
        for j in junk:
            ttnn.deallocate(j)
