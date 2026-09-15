"""4-rank topology + D2D probe for the [8,1] intra-galaxy pipeline. NO weights, NO model.

Validates exactly the pieces the new topology config introduces, in the order the real runner does:
  1. the rank binding + single_bh_galaxy_4x8x1_z_chain MGD let each rank open its own [8,1] mesh,
  2. the Z-link MeshSockets between consecutive column meshes come up (this is the step that fails
     if the topology mapper cannot realise a column as an 8x1 LINE, or if the inter-mesh links are
     not where the descriptor claims),
  3. one real activation-shaped chunk actually traverses all three rank boundaries, and
  4. the shutdown sentinel drains the pipeline the way the runner's teardown relies on.

Deliberately imports the runner's OWN helpers rather than re-implementing them, so a pass here is
evidence about the production path and not about a lookalike.
"""
import os
import time

import torch
from loguru import logger

import ttnn

# Importing the runner module reads the PREFILL_* env (SP/TP/CHUNK_SIZE/...) at import time and picks
# the adapter -- the same module-level config the real runner would take from this rank binding.
from models.demos.common.prefill.runners import prefill_runner as R
from models.demos.common.prefill.runners.runner_utils import open_mesh_device

if not ttnn.distributed_context_is_initialized():
    ttnn.init_distributed_context()
rank = int(ttnn.distributed_context_get_rank())
num_ranks = int(ttnn.distributed_context_get_size())
tag = f"[probe rank {rank}/{num_ranks}]"
logger.info(
    f"{tag} mesh_shape={R.GLOBAL_MESH_SHAPE} chunk={R.CHUNK_SIZE} TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')}"
)

mesh = open_mesh_device(R.GLOBAL_MESH_SHAPE, R.MODEL_CFG, l1_small_size=R.ADAPTER.l1_small_size)
logger.info(f"{tag} PROBE mesh open: shape={mesh.shape} device_ids={mesh.get_device_ids()}")
ttnn.distributed_context_barrier()

hidden = R.ADAPTER.load_hf_config().hidden_size  # same source as the runner's d2d_activation_width
d2d_in, d2d_out = R.build_d2d_pipeline_endpoints(mesh, rank, num_ranks, R.CHUNK_SIZE, hidden)
logger.info(f"{tag} PROBE d2d endpoints up (in={d2d_in is not None} out={d2d_out is not None})")
ttnn.distributed_context_barrier()

# One chunk down the whole pipeline, timed per hop.
#
# The fabric-link LEASE PROTOCOL is what makes a transfer actually happen, and its naming is
# inverted from the intuition: release_fabric_links() GRANTS the links to the socket service (so it
# may ship/drain), wait_for_fabric_links() RECLAIMS them once it is done. Skipping the grant does not
# error -- the push returns, nothing moves, and every downstream rank blocks forever in a recv that
# has no timeout. So mirror the runner's own _lease_reclaim() + _compute_and_send() ordering exactly:
#   per chunk:  reclaim in+out, GRANT inbound, recv, compute, send, GRANT outbound.
meta = {"slot_id": 0, "actual_start": 0, "actual_end": R.CHUNK_SIZE}


def hop(payload, label):
    """One pipeline traversal of `payload` on this rank, following the runner's lease ordering."""
    R._lease_reclaim(d2d_in, d2d_out)
    got = None
    act = payload
    if d2d_in is not None:
        t0 = time.perf_counter()
        act, got, _md = R._d2d_recv(d2d_in)
        ttnn.synchronize_device(mesh)
        logger.info(f"{tag} PROBE {label} recv {(time.perf_counter()-t0)*1000:.2f}ms meta={got} shape={act.shape}")
    if d2d_out is not None:
        t0 = time.perf_counter()
        R._d2d_send(d2d_out, act, rank, meta, deallocate=False)
        d2d_out.release_fabric_links()  # the grant that ships it
        ttnn.synchronize_device(mesh)
        logger.info(f"{tag} PROBE {label} send {(time.perf_counter()-t0)*1000:.2f}ms")
    return got


payload = None
if rank == 0:
    payload = ttnn.from_torch(
        torch.randn(1, 1, R.CHUNK_SIZE, hidden),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(mesh, R.D2D_MAPPER_CONFIG),
    )

got = hop(payload, "chunk")
if got is not None:
    assert got["actual_end"] == R.CHUNK_SIZE, f"{tag} metadata corrupted across the hop: {got}"
ttnn.distributed_context_barrier()
logger.info(f"{tag} PROBE chunk traversed the pipeline")

# A second traversal: the first pays kernel compile, so only the second is a usable transport number.
got = hop(payload, "chunk2")
ttnn.distributed_context_barrier()
logger.info(f"{tag} PROBE second chunk done")

# Drain with the sentinel exactly as the runner's teardown does, so we learn now (not during a long
# model run) whether shutdown releases cleanly on this topology.
R._lease_reclaim(d2d_in, d2d_out)
if d2d_in is not None:
    _a, s, _m = R._d2d_recv(d2d_in)
    logger.info(f"{tag} PROBE got sentinel={s}")
if d2d_out is not None:
    R._forward_shutdown(d2d_out, rank, hidden)  # sends + grants + releases

ttnn.distributed_context_barrier()
logger.info(f"{tag} PROBE OK -- topology, sockets, two chunks and shutdown all validated")
ttnn.close_mesh_device(mesh)
