# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Bring-up probe for a 4-stage pipeline on a SINGLE 32-chip BH galaxy.

The full prefill runner loads ~119B of weights before it ever reaches fabric init, so a topology bug
surfaces many minutes in and tangled up with model errors. This probe exercises only the two things
that are actually novel about single-galaxy PP, in order, and nothing else:

  stage "mesh" — do 4 ranks on one host each open their own 8-chip (8,1) mesh from the per-rank
                 TT_VISIBLE_DEVICES carve, with 2D fabric trained across all four?
  stage "d2d"  — can consecutive ranks stand up D2D endpoints and pass a tensor 0->1->2->3 intact?

Everything here mirrors the real runner's calls (it imports the runner's own open_mesh_device and
activation_global_spec) so a pass means the transport layer is genuinely viable, not just that some
simplified stand-in worked. It deliberately does NOT import prefill_runner: that module resolves the
model adapter at import time, which would drag the model in and defeat the point.

Run under tt-run with the 4-rank binding, e.g.
    python3 ttnn/ttnn/distributed/ttrun.py --rank-binding mistral4_bringup/pp4/pp4_single_galaxy_rank_bindings.yaml \
      --mpi-args "--host <this-host>:4 --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH" \
      -- python3 -m mistral4_bringup.pp4.probe_pp4
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.runner_utils import activation_global_spec, open_mesh_device

SP = int(os.environ.get("PREFILL_SP", 8))
TP = int(os.environ.get("PREFILL_TP", 1))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5120))
HIDDEN_SIZE = int(os.environ.get("PROBE_HIDDEN_SIZE", 4096))  # Mistral-Small-4 hidden_size
STAGE = os.environ.get("PROBE_STAGE", "mesh")  # "mesh" | "d2d"

# Same worker core / metadata sizing the runner uses; the D2D ops assume both sides agree.
SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
METADATA_SIZE_BYTES = 12
D2D_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", 256))

# tp is 1 here, so the emb axis is replicated rather than sharded; seq shards across the sp rows.
D2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()])


class _ProbeCfg:
    """Stand-in for the model config; open_mesh_device only reads FABRIC_PAYLOAD_SIZE off it."""

    FABRIC_PAYLOAD_SIZE = HIDDEN_SIZE


def _build_endpoints(mesh_device, rank: int, num_ranks: int):
    """Inbound-from-rank-1 / outbound-to-rank+1, built inbound-first so the chained socket rendezvous
    unblocks stage by stage instead of deadlocking (same ordering rule as the runner)."""
    global_spec = activation_global_spec(CHUNK_SIZE, HIDDEN_SIZE)

    def _common():
        # Mapper is MOVED into the create call, so a middle rank building both endpoints needs a fresh
        # one per call rather than a shared instance.
        return dict(
            global_spec=global_spec,
            mapper=ttnn.create_mesh_mapper(mesh_device, D2D_MAPPER_CONFIG),
            fifo_size_bytes=D2D_FIFO_SIZE_BYTES,
            sender_worker_cores=SYNC_WORKER_CORES,
            receiver_worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
            share_fabric_links=True,
            socket_buffer_type=ttnn.BufferType.L1,
        )

    inbound = None
    if rank > 0:
        logger.info(f"[probe rank {rank}] creating inbound receiver from rank {rank - 1}")
        inbound = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh_device, sender_rank=rank - 1, receiver_rank=rank, **_common()
        )
    outbound = None
    if rank < num_ranks - 1:
        logger.info(f"[probe rank {rank}] creating outbound sender to rank {rank + 1}")
        outbound = ttnn.D2DStreamService.create_sender(
            sender_mesh=mesh_device, sender_rank=rank, receiver_rank=rank + 1, **_common()
        )
    return inbound, outbound


def _send(outbound, activation, meta: dict) -> None:
    backing = outbound.get_backing_tensor()
    md_tensor = ttnn.from_torch(
        torch.tensor([meta["slot_id"], meta["actual_start"], meta["actual_end"]], dtype=torch.int32).reshape(
            1, 1, 1, -1
        ),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=backing.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            backing.device(),
            ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]),
        ),
    )
    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(outbound, activation, metadata=md_tensor)


def _recv(inbound):
    act, metadata_device = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        inbound, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(metadata_device)[0]).view(torch.int32).flatten()
    return act, {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}


def main() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())

    logger.info(
        f"[probe rank {rank}/{num_ranks}] stage={STAGE} mesh=({SP},{TP}) "
        f"TT_MESH_ID={os.environ.get('TT_MESH_ID')} "
        f"TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')} "
        f"fabric={os.environ.get('PREFILL_FABRIC_MODE')}"
    )

    t0 = time.perf_counter()
    mesh_device = open_mesh_device((SP, TP), _ProbeCfg)
    logger.info(
        f"[probe rank {rank}] MESH OK shape={mesh_device.shape} devices={mesh_device.get_num_devices()} "
        f"({(time.perf_counter() - t0):.1f}s)"
    )

    ttnn.distributed_context_barrier()
    if rank == 0:
        logger.info(f"[probe] ALL {num_ranks} MESHES OPENED — single-galaxy carve works")

    if STAGE == "d2d" and num_ranks > 1:
        inbound, outbound = _build_endpoints(mesh_device, rank, num_ranks)
        # The chained rendezvous finishes at staggered times, so without this every rank would race
        # ahead to its first fabric-link lease while a neighbour is still building its socket.
        ttnn.distributed_context_barrier()
        logger.info(f"[probe rank {rank}] D2D ENDPOINTS OK")

        # Fabric links are leased, and a receiver only drains once it has been GRANTED them. This
        # mirrors the runner's per-chunk _lease_reclaim: reclaim both endpoints, then grant the
        # inbound. Skipping the inbound grant hangs the receive indefinitely — the sender ships
        # happily and the receiver waits forever for links it was never given.
        if inbound is not None:
            inbound.wait_for_fabric_links()
        if outbound is not None:
            outbound.wait_for_fabric_links()
        if inbound is not None:
            inbound.release_fabric_links()

        # One all-ones payload walks the whole pipeline unchanged. The hop count rides in the metadata
        # (bumped per forward) rather than in the tensor: the outbound op hard-fails on any spec
        # mismatch against the sender backing, so re-deriving the activation through an op would risk
        # failing for reasons that have nothing to do with the transport we are testing.
        if rank == 0:
            act = ttnn.from_torch(
                torch.full((1, 1, CHUNK_SIZE, HIDDEN_SIZE), 1.0, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.create_mesh_mapper(mesh_device, D2D_MAPPER_CONFIG),
            )
            _send(outbound, act, {"slot_id": 0, "actual_start": 0, "actual_end": CHUNK_SIZE})
            outbound.release_fabric_links()
            logger.info("[probe rank 0] sent payload (all ones, hop=0)")
        else:
            act, meta = _recv(inbound)
            got = ttnn.to_torch(ttnn.get_device_tensors(act)[0]).float()
            payload_ok = torch.allclose(got, torch.ones_like(got))
            hop_ok = meta["slot_id"] == rank - 1
            logger.info(
                f"[probe rank {rank}] RECV meta={meta} shard={tuple(got.shape)} "
                f"mean={got.mean().item():.4f} payload_ok={payload_ok} hop_ok={hop_ok}"
            )
            if outbound is not None:
                _send(outbound, act, {**meta, "slot_id": meta["slot_id"] + 1})
                outbound.release_fabric_links()
                logger.info(f"[probe rank {rank}] forwarded to rank {rank + 1} (hop={meta['slot_id'] + 1})")
            else:
                verdict = "PASSED" if (payload_ok and hop_ok) else "FAILED"
                logger.info(f"[probe rank {rank}] PIPELINE {verdict} end to end across {num_ranks} stages")

    ttnn.distributed_context_barrier()
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[probe rank {rank}] done")


if __name__ == "__main__":
    main()
