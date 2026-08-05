# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: does RingJointSDPA's chunked sliding path accept Gemma4's geometry?

The sliding path was added for GPT-OSS (W=128, 8Q:1K:1V, D64) and
``validate_on_program_cache_miss`` gates on exactly those values. The
*implementation* underneath looks general — ``build_sliding_q_work_plan`` and
``chunked_sliding_halo_tile_rows`` derive everything from runtime window /
k_chunk / slab sizes, and the program factory's CB sizing scales with DHt/vDHt —
so the question is whether those gates are a tested-configuration allowlist or a
real limit.

Gemma4-31B sliding layers want W=1024, and at TP=8 hold 4Q:2K:2V local heads at
head_dim 256. Two things make that interesting:

  * the halo lands exactly on the slab boundary. halo_tiles =
    ceil((W-1)/k_chunk)*k_chunk_tiles = 32 for W=1024/k_chunk=128, against a
    32-tile (1024-token) local slab. The guards admit equality
    (``halo_tile_rows > q_local_tile_rows`` rejects), but GPT-OSS at W=128 sits
    at 4 tiles against a 20-tile slab, so the equality case is never exercised
    upstream. That is where a bug would live.
  * D256 with 4 Q heads is ~4x GPT-OSS's per-head data at D64, so L1/CB budget is
    the other plausible failure.

Run against the harness's native galaxy ring (SP=8) with chunk 8192, which gives
per-device q_local = 1024 = the same slab/halo equality Gemma4 hits at SP=4 /
chunk 4096. Ring size itself is already allowed to be 4 or 8.
"""

import contextlib
import os
from dataclasses import replace
from unittest import mock

import pytest

import ttnn

from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    CHUNKED_PREFILL_CHUNK_ID_ENV,
    GPT_OSS_CHUNKED_MODEL,
    MESH_CONFIG,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
    run_ring_joint_sdpa_chunked,
)
from tests.nightly.sdpa_perf_utils import MeshConfig

# The suite's own galaxy default is SP=8 / TP=4, which does not open on this
# machine: open_ring_joint_sdpa_runtime asks for FABRIC_1D_RING with sp_axis=1,
# i.e. a ring of 8 along the columns, but this galaxy's torus is on the 4-wide
# dim (single_bh_galaxy_torus_x: dims [8,4], dim_types [LINE, RING]). The
# upstream GPT-OSS sliding tests fail here with the same MGD mapping error, so
# that is an environment limit rather than anything about the geometry below.
#
# SP=4 / TP=8 puts the ring on the 4-wide torus dim, giving mesh (8,4) — and it
# happens to be exactly Gemma4's CP=4 x TP=8 split.
GEMMA4_MESH_CONFIG = MeshConfig(
    is_galaxy=MESH_CONFIG.is_galaxy,
    num_devices=MESH_CONFIG.num_devices,
    sp_size=4,
    tp_size=8,
    grid_cols=MESH_CONFIG.grid_cols,
    grid_rows=MESH_CONFIG.grid_rows,
)

# Gemma4-31B sliding layer, TP=8: 32 attention heads / 8 = 4 local Q,
# 16 KV heads / 8 = 2 local K and V, head_dim 256, window 1024.
GEMMA4_SLIDING_WINDOW = 1024
GEMMA4_LOCAL_Q_HEADS = 4
GEMMA4_LOCAL_KV_HEADS = 2
GEMMA4_HEAD_DIM = 256

# Gemma4's real prefill chunk. At ring=4 this gives per-device q_local = 1024,
# so halo_tiles == q_local_tiles == 32: the exact slab boundary.
PROBE_CHUNK = 4096
PROBE_TOTAL_SEQ = 2 * PROBE_CHUNK  # the path needs a complete predecessor Q group


@contextlib.contextmanager
def torus_x_fabric():
    """Redirect the harness's FABRIC_1D_RING request onto a torus-X fabric.

    On a UBB galaxy ``get_fabric_type`` maps FABRIC_1D_RING to TORUS_XY — rings on
    *both* axes — and this machine cannot provide that: the torus_xy descriptor
    fails MGD mapping while torus_x maps and opens. A ring does exist on the
    4-wide axis, which is the axis the harness uses for SP (sp_axis=1) and the one
    Gemma4 would use for CP=4.

    Also downgrades STRICT_INIT: the stock torus descriptors declare
    ``channels { count: 2 policy: STRICT }`` and four chips here report 4 eth
    channels in a direction, so STRICT refuses. Pair this with
    TT_MESH_GRAPH_DESC_PATH pointing at a RELAXED-channel torus_x copy.
    """
    real_set_fabric_config = ttnn.set_fabric_config

    def redirect(fabric_config, reliability_mode=None, *args, **kwargs):
        if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING:
            fabric_config = ttnn.FabricConfig.FABRIC_2D_TORUS_X
            reliability_mode = ttnn.FabricReliabilityMode.RELAXED_INIT
        return real_set_fabric_config(fabric_config, reliability_mode, *args, **kwargs)

    with mock.patch.object(ttnn, "set_fabric_config", redirect):
        yield


@pytest.mark.parametrize("q_chunk,k_chunk", [(64, 128), (128, 128)], ids=["q64k128", "q128k128"])
def test_gemma4_sliding_geometry_probe(q_chunk, k_chunk):
    """Does the chunked sliding path run at Gemma4's window / heads / head_dim?

    Expected before relaxing the allowlist: TT_FATAL naming the GPT-OSS window,
    head counts or head_dim. That failure is the useful result — it confirms the
    gate is the only blocker. After relaxing it to the structural constraints,
    this either passes (usable) or fails on L1 / the halo equality edge (concrete
    feedback rather than speculation).
    """
    model = replace(
        GPT_OSS_CHUNKED_MODEL,
        name="gemma4_sliding_probe",
        nhq=GEMMA4_LOCAL_Q_HEADS,
        nhk=GEMMA4_LOCAL_KV_HEADS,
        nhv=GEMMA4_LOCAL_KV_HEADS,
        d_q=GEMMA4_HEAD_DIM,
        d_k=GEMMA4_HEAD_DIM,
        d_v=GEMMA4_HEAD_DIM,
        q_chunk_sizes=[q_chunk],
        k_chunk_sizes=[k_chunk],
        seq_len=PROBE_CHUNK,
    )

    with torus_x_fabric():
        runtime = open_ring_joint_sdpa_runtime(GEMMA4_MESH_CONFIG)
    # The opener hardcodes num_links=2, but under the RELAXED-channel torus_x
    # descriptor some pairs on this machine expose only one ethernet channel
    # ("Requested link index 1 is out of bounds ... 1 ethernet channels available
    # b/w (M0, D0) and (M0, D1)"). One link is enough to exercise the geometry.
    runtime = replace(runtime, num_links=1)
    runtime.mesh_device.enable_program_cache()

    # Only the last chunk. Chunk 0 has logical_n == q_group_size, so it has no
    # predecessor Q group and the chunked sliding path refuses it by design
    # ("requires a complete predecessor and current Q group"). The upstream
    # native-ring test isolates the final chunk the same way.
    final_chunk = PROBE_TOTAL_SEQ // PROBE_CHUNK - 1
    try:
        with mock.patch.dict(os.environ, {CHUNKED_PREFILL_CHUNK_ID_ENV: str(final_chunk)}):
            run_ring_joint_sdpa_chunked(
                GEMMA4_MESH_CONFIG,
                model,
                batch_size=1,
                chunk_size=PROBE_CHUNK,
                total_seq=PROBE_TOTAL_SEQ,
                qk_configs=[(q_chunk, k_chunk)],
                persistent_buffer_mode="exact_per_chunk",
                sliding_window_size=GEMMA4_SLIDING_WINDOW,
                runtime=runtime,
            )
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)


# 16K target: 4 chunks of 4096. Gemma4 prefills 256k this way, so the multi-chunk
# behaviour matters more than the 2-chunk case above.
PROBE_16K_CHUNKS = 4
PROBE_16K_TOTAL_SEQ = PROBE_CHUNK * PROBE_16K_CHUNKS


@pytest.mark.parametrize("chunk_id", [1, 2, 3], ids=["chunk1", "chunk2", "chunk3"])
def test_gemma4_sliding_16k_multichunk(chunk_id):
    """Every chunk of a 16K / 4x4096 prefill that the chunked sliding path can serve.

    Chunk 0 is excluded on purpose: logical_n == q_group_size there, so it has no
    predecessor Q group and this mode refuses it ("requires chunked prefill"). M3
    handles that case with a separate non-chunked ring_joint over the chunk's own
    SP-sharded K/V; Gemma4 already has an equivalent in its mask-based CP path.

    Uses persistent_buffer_mode="reuse_max" — one max-length K/V buffer reused
    across chunks, which is the production shape. M3 records that sizing the
    ring-gather buffer to logical_n instead of max_seq_len happens to work for a
    2-chunk run (where the final chunk's logical_n == max_seq_len) and fails beyond
    it with "gather dim 2 too small", so a 4-chunk run is the case that catches it.
    """
    model = replace(
        GPT_OSS_CHUNKED_MODEL,
        name="gemma4_sliding_16k",
        nhq=GEMMA4_LOCAL_Q_HEADS,
        nhk=GEMMA4_LOCAL_KV_HEADS,
        nhv=GEMMA4_LOCAL_KV_HEADS,
        d_q=GEMMA4_HEAD_DIM,
        d_k=GEMMA4_HEAD_DIM,
        d_v=GEMMA4_HEAD_DIM,
        q_chunk_sizes=[64],
        k_chunk_sizes=[128],
        seq_len=PROBE_CHUNK,
    )

    with torus_x_fabric():
        runtime = open_ring_joint_sdpa_runtime(GEMMA4_MESH_CONFIG)
    runtime = replace(runtime, num_links=1)
    runtime.mesh_device.enable_program_cache()
    try:
        with mock.patch.dict(os.environ, {CHUNKED_PREFILL_CHUNK_ID_ENV: str(chunk_id)}):
            run_ring_joint_sdpa_chunked(
                GEMMA4_MESH_CONFIG,
                model,
                batch_size=1,
                chunk_size=PROBE_CHUNK,
                total_seq=PROBE_16K_TOTAL_SEQ,
                qk_configs=[(64, 128)],
                persistent_buffer_mode="reuse_max",
                sliding_window_size=GEMMA4_SLIDING_WINDOW,
                runtime=runtime,
            )
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)
