# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-safe metadata path of ring_indexer_score_dsa.

`chunk_start_idx` is a hash-EXCLUDED host runtime argument: one cached program serves every prefill chunk,
and `override_runtime_arguments` re-patches the derived causal fields (chunk_start_tiles, straddle_*,
kv_len_tiles) on each dispatch. A ttnn trace replay never runs that host patch, so a captured scalar stays
frozen at its capture-time value and every later chunk is scored against the FIRST chunk's causal window --
silently wrong scores, not an error. `chunk_start_idx_tensor` moves the value into a 1-element uint32 DRAM
tensor the reader reads on-device, so one captured program is correct for every chunk.

The matrix is scalar-vs-metadata x eager-vs-traced. The comparisons here are DEVICE-TO-DEVICE and
bit-exact (`torch.equal`), not PCC-vs-torch: the scalar path's numerics are already covered by
test_ring_indexer_score_dsa.py, and what needs proving is the narrower claim that the tensor path computes
the same thing the scalar path does.

The load-bearing test is `test_metadata_trace_replay`. A replay that only re-runs the CAPTURED chunk_start
passes even if the kernel ignores the metadata tensor entirely, so each replay here uses a DIFFERENT value
and is checked against that value's own eager result.
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_indexer_score import (
    glx_config,
    _global_inputs,
    _to_slab,
    QB_SQ,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.ring_indexer_score_test_utils import (
    _open_ring4_ccl,
    _close_ring4_ccl,
    _persistent_buffer,
    _shard_k,
    RING,
    SP_AXIS,
    CHUNK_GLOBAL,
    T,
)

pytestmark = [
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() < 8, reason="ring-of-4 needs the 8-chip LoudBox (2x4)"),
]

HEADS = 16  # one representative head count; the causal derivation is head-independent
TRACE_REGION = 32 * 1024 * 1024

# chunk_start values, all tile-aligned and within the causal window bound (max_cs + Sq <= T):
#   0        slab-aligned, boundary_chip 0, no straddle -- the degenerate case
#   320      MID-SLAB (320 % chunk_local(640) != 0) -> the boundary chip straddles a slab boundary, so
#            straddle_q_tile / straddle_jump_tiles are non-zero. This is the case a naive linear
#            chunk_start formula gets wrong, so it is the one that actually exercises the closed form.
#   640      chip-aligned but not slab-aligned -> boundary_chip rotation with zero offset
#   2560     exactly one global slab on
CHUNK_STARTS = [0, 320, 640, CHUNK_GLOBAL]


def _meta_scalar(mesh, value: int, *, on_device: bool = True):
    """The 1-element uint32 metadata tensor: [1,1,1,1] ROW_MAJOR, replicated, DRAM."""
    t = torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1)
    kw = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    if on_device:
        kw.update(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(t, **kw)


def _dev_inputs(submesh, q_g, w_g, k_host):
    shard = ttnn.ShardTensorToMesh(submesh, dim=2)
    q_dev = ttnn.from_torch(q_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    w_dev = ttnn.from_torch(w_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    k_local = _shard_k(submesh, k_host, dtype=ttnn.bfloat16)
    k_gathered = _persistent_buffer(submesh, torch.zeros_like(k_host[:1]), dtype=ttnn.bfloat16)
    return q_dev, w_dev, k_local, k_gathered


def _score(submesh, q_dev, k_gathered, w_dev, k_local, sems, subdev, *, chunk_start=None, meta=None):
    """One fused-ring score. Exactly one of chunk_start / meta is supplied.

    kv_len is left UNSET on both paths on purpose. The metadata path pins it to the full allocated width
    on-device (so the score tail is causally -inf rather than stale, which is what lets top-k stay
    unbounded), and a scalar kv_len would bound the scalar run differently -- the two would then legitimately
    differ in the tail and the bit-exact comparison would be measuring the wrong thing.
    """
    return ttnn.experimental.ring_indexer_score_dsa(
        q_dev,
        k_gathered,
        w_dev,
        k_local,
        sems,
        cluster_axis=SP_AXIS,
        topology=ttnn.Topology.Linear,
        num_links=1,
        ag_sub_device_id=subdev,
        chunk_start_idx=chunk_start,
        chunk_start_idx_tensor=meta,
        program_config=glx_config(HEADS),
        block_cyclic_sp_axis=SP_AXIS,
        block_cyclic_chunk_local=QB_SQ,
    )


def _to_host(submesh, out):
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))


@pytest.mark.parametrize("chunk_start", CHUNK_STARTS, ids=[f"cs{c}" for c in CHUNK_STARTS])
def test_metadata_matches_scalar_eager(chunk_start):
    """Eager: the tensor path must reproduce the scalar path BIT-EXACTLY at the same chunk_start."""
    submesh, parent, sems, subdev, stall = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(HEADS, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL)
        q_dev, w_dev, k_local, k_gathered = _dev_inputs(submesh, q_g, w_g, k_host)

        scalar_out = _score(submesh, q_dev, k_gathered, w_dev, k_local, sems, subdev, chunk_start=chunk_start)
        ttnn.synchronize_device(submesh, sub_device_ids=stall)
        scalar_t = _to_host(submesh, scalar_out)

        meta = _meta_scalar(submesh, chunk_start)
        meta_out = _score(submesh, q_dev, k_gathered, w_dev, k_local, sems, subdev, meta=meta)
        ttnn.synchronize_device(submesh, sub_device_ids=stall)
        meta_t = _to_host(submesh, meta_out)

        ndiff = int((scalar_t != meta_t).sum())
        assert ndiff == 0, (
            f"metadata path differs from scalar at chunk_start={chunk_start}: {ndiff}/{scalar_t.numel()} "
            f"elements, max |diff| {float((scalar_t - meta_t).abs().max()):.3e}"
        )
        logger.info(f"ring4 metadata cs={chunk_start}: bit-exact vs scalar ({scalar_t.numel()} elements)")
    finally:
        _close_ring4_ccl(parent, submesh, stall)


def test_metadata_trace_replay():
    """Capture ONCE, then replay with a DIFFERENT chunk_start each time and check every replay against that
    value's own eager result.

    Replaying only the captured value would pass even if the kernel ignored the metadata tensor, so the
    replay order deliberately includes a descending pass and a shuffled pass: a stale read then lands on a
    value that is too LARGE as well as too small, and cannot coincidentally match by always trailing by one.
    """
    # Eager references first, in their own device session, so nothing from the traced session can leak in.
    submesh, parent, sems, subdev, stall = _open_ring4_ccl()
    refs = {}
    try:
        q_g, k_nat, w_g = _global_inputs(HEADS, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL)
        q_dev, w_dev, k_local, k_gathered = _dev_inputs(submesh, q_g, w_g, k_host)
        for cs in CHUNK_STARTS:
            out = _score(submesh, q_dev, k_gathered, w_dev, k_local, sems, subdev, chunk_start=cs)
            ttnn.synchronize_device(submesh, sub_device_ids=stall)
            refs[cs] = _to_host(submesh, out)
    finally:
        _close_ring4_ccl(parent, submesh, stall)

    submesh, parent, sems, subdev, stall = _open_ring4_ccl(trace_region_size=TRACE_REGION)
    try:
        q_g, k_nat, w_g = _global_inputs(HEADS, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL)
        q_dev, w_dev, k_local, k_gathered = _dev_inputs(submesh, q_g, w_g, k_host)

        meta = _meta_scalar(submesh, CHUNK_STARTS[0])
        host_meta = {cs: _meta_scalar(submesh, cs, on_device=False) for cs in CHUNK_STARTS}

        # Warm/compile the metadata program BEFORE capture: a program-cache miss inside begin_trace_capture
        # would compile into the capture rather than being replayed from it.
        _score(submesh, q_dev, k_gathered, w_dev, k_local, sems, subdev, meta=meta)
        ttnn.synchronize_device(submesh, sub_device_ids=stall)

        tid = ttnn.begin_trace_capture(submesh, cq_id=0)
        traced_out = _score(submesh, q_dev, k_gathered, w_dev, k_local, sems, subdev, meta=meta)
        ttnn.end_trace_capture(submesh, tid, cq_id=0)
        ttnn.synchronize_device(submesh, sub_device_ids=stall)

        order = CHUNK_STARTS + CHUNK_STARTS[::-1] + [CHUNK_STARTS[-1], CHUNK_STARTS[0], CHUNK_STARTS[1]]
        for i, cs in enumerate(order):
            ttnn.copy_host_to_device_tensor(host_meta[cs], meta)
            ttnn.execute_trace(submesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(submesh, sub_device_ids=stall)
            got = _to_host(submesh, traced_out)
            ndiff = int((got != refs[cs]).sum())
            if ndiff:
                # Name the most likely cause rather than just reporting a mismatch: matching a DIFFERENT
                # chunk_start's reference is the signature of a stale/ignored metadata read.
                matches = [o for o in CHUNK_STARTS if int((got != refs[o]).sum()) == 0]
                raise AssertionError(
                    f"replay {i} (chunk_start={cs}) differs from its eager reference in {ndiff}/{got.numel()} "
                    f"elements; it matches chunk_start(s) {matches or 'none'} instead "
                    f"-- {'stale metadata read (value not picked up on replay)' if matches else 'unrelated'}"
                )
        ttnn.release_trace(submesh, tid)
        logger.info(
            f"ring4 metadata trace replay: {len(order)} replays over {len(CHUNK_STARTS)} chunk_starts, all bit-exact"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall)


def test_metadata_rejects_conflicting_args(expect_error):
    """The tensor and the scalar bounds are mutually exclusive: kv_len is DERIVED on the metadata path, so
    accepting a caller-supplied one would silently ignore it."""
    submesh, parent, sems, subdev, stall = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(HEADS, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL)
        q_dev, w_dev, k_local, k_gathered = _dev_inputs(submesh, q_g, w_g, k_host)
        meta = _meta_scalar(submesh, 0)

        with expect_error(RuntimeError, "kv_len"):
            ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                sems,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdev,
                chunk_start_idx_tensor=meta,
                kv_len=CHUNK_GLOBAL,  # rejected: derived on-device from chunk_start_idx
                program_config=glx_config(HEADS),
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
            )

        # The derivation needs the slab geometry, so the block-cyclic layout is required.
        with expect_error(RuntimeError, "block-cyclic"):
            ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                sems,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdev,
                chunk_start_idx_tensor=meta,
                program_config=glx_config(HEADS),
            )
        logger.info("ring4 metadata: conflicting-argument guards fire as expected")
    finally:
        _close_ring4_ccl(parent, submesh, stall)
