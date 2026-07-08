# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
STEP B of the ring-fused indexer_score roadmap (see ring_indexer_score_fusion_design.md).

The FIRST real fusion: ONE op (ttnn.experimental.indexer_score_dsa_fused) co-schedules the ring_attention
all-gather + the indexer compute into a single program, with a producer->consumer semaphore handshake so the
reader COARSE-BARRIERS on the whole gather before scoring. This is the delta from the passing Step-A2 flow
(all_gather then a separate indexer_score): one program instead of two, exercising the signaler wiring,
per-device Linear fwd/bwd thresholds, the forward-writer +1 pre-signal, buffer sharing, and the reader's
FUSED_RING coarse barrier -- the places a first bring-up would hang.

Step B keeps the compute path byte-identical to A2 and HOST-SEEDS the local slab into the gathered buffer
(the device-side prologue copy / dual-source local sourcing is deferred to Steps C/D). No overlap win yet --
the coarse barrier waits for the entire gather; Step B only proves the plumbing is correct and does not hang.

Run:  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score_lb_ring4_fused.py
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    assert_indexer_match,
    glx_config,
    _global_inputs,
    _per_sp_ref,
    _to_slab,
    QB_HISTORY,
    QB_SQ,
    QB_CASES,
    QB_IDS,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score_lb_ring4_ag_equiv import (
    _open_ring4_ccl,
    _close_ring4_ccl,
    _persistent_buffer,
    _shard_k,
    RING,
    SP_AXIS,
    CHUNK_GLOBAL,
    T,
)

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")


def _run_fused(heads, *, block_cyclic):
    """SP-shard q/w/k_local, seed the gathered buffer, run the ONE fused op, check vs the per-SP reference."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL) if block_cyclic else k_nat

        shard = ttnn.ShardTensorToMesh(submesh, dim=2)
        q_dev = ttnn.from_torch(q_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)

        # k_local = this chip's SP shard [1,1,sll,D] (the all-gather INPUT); k_gathered = the full [1,1,T,D]
        # persistent AG OUTPUT buffer, seeded with ZEROS. The AG writes only the REMOTE bands; the local band
        # stays zero and the reader dual-sources it from k_local on device (Step D). A correct score therefore
        # PROVES device-side local sourcing (a stale/zero local band would fail the -inf map + PCC).
        k_local = _shard_k(submesh, k_host)
        k_gathered = _persistent_buffer(submesh, torch.zeros_like(k_host))

        bc_kwargs = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=QB_SQ) if block_cyclic else {}
        out = ttnn.experimental.indexer_score_dsa_fused(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            program_config=glx_config(heads),
            **bc_kwargs,
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _per_sp_ref(q_g, k_nat, w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"ring4 fused {layout} (heads={heads}): fused all-gather + dual-source score matched reference")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused(case_id, heads, block_cyclic):
    """Step B: single fused all-gather + indexer_score op on a ring of 4, checked vs the per-SP DSA reference."""
    _run_fused(heads, block_cyclic=block_cyclic)
