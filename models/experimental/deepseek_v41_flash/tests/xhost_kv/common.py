# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the cross-host KV write test (prefill Galaxy on host 40 writes decode's DRAM on host 41 over
TT-Fabric, with no device-side synchronization; the hand-off is a TCP message).

The three parts:

* ``receiver_init.py`` (rank 1, host 41, the decode side): opens mesh 1 with fabric, allocates decode-shaped paged
  pools filled with a sentinel, sends their addresses (and every chip's DRAM bank table) to the sender over TCP, then
  per round asks for a write into a random set of blocks and checks them the moment the sender's "done" arrives.
* ``sender_write.py`` (rank 0, host 40, the prefill side): opens mesh 0 with fabric, and per round writes seeded
  random blocks from its own DRAM straight into the receiver's pages with fabric unicast writes, then replies "done".
* ``receiver_verify.py`` (host 41, after the job, no device opened): reads every page of both pools back over UMD and
  checks them byte for byte: each block the last round that wrote it, untouched blocks the sentinel.

Parts 1 and 2 are one MPI job: the control plane of a two-mesh fabric all-gathers over every rank when it starts, so
both sides must come up together under ``tt-run``. ``tt-run`` hands every rank the launch host's checkout paths, so
when host 41 runs from a different checkout (``/big_home`` is not shared) rank 1 points them back at its own. Launch
from the repo root on host 40 (this directory, the mesh graph descriptor and the rank bindings copied to ``H`` on 41)::

    tt-run --rank-binding tests/tt_metal/distributed/config/dual_bh_galaxy_8x4_hosts_40_41_two_mesh_rank_bindings.yaml \\
      --mpi-args "--host DC16-2-BG-0202-u20-40,DC16-2-BG-0202-u13-41 \\
        --map-by rankfile:file=$PWD/tests/tt_metal/distributed/config/dual_8x4_hosts_40_41.rankfile \\
        --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0.1 --bind-to none --tag-output" \\
      bash -c 'D=models/experimental/deepseek_v41_flash/tests/xhost_kv
        if [ "$OMPI_COMM_WORLD_RANK" = 0 ]; then exec python $D/sender_write.py; fi
        H=/big_home/ttuser/smanoj/metal_1; cd $H
        export TT_METAL_HOME=$H TT_METAL_RUNTIME_ROOT=$H PYTHONPATH=$H LD_LIBRARY_PATH=$H/build/lib \\
          VIRTUAL_ENV=$H/python_env TT_MESH_GRAPH_DESC_PATH=$H/tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_bh_galaxy_8x4_hosts_40_41_two_mesh.textproto
        exec $H/python_env/bin/python $D/receiver_init.py --sender 10.82.97.40'

then on host 41: ``python models/experimental/deepseek_v41_flash/tests/xhost_kv/receiver_verify.py``.

The pools are decode's own formats (``tt/model.py``, ``prepare_static_decode``): compressed KV ``comp_kv`` is bf16
ROW_MAJOR ``[blocks, 1, 64, 512]`` (one 1 KiB entry a page, 64 pages a block) and index keys ``idx_key_cache`` bf16
TILE ``[blocks, 1, 64, 128]`` (eight 2 KiB tiles a block). Both are DRAM interleaved: page ``p`` lives in bank
``p % banks`` at ``base + (p // banks) * page_bytes`` (+ the bank's offset), the same on every chip of the mesh.
"""

from __future__ import annotations

import socket
import sys
import time

import torch

import ttnn

MESH_SHAPE = (8, 4)
SENDER_RANK, RECEIVER_RANK = 0, 1
PORT = 9471
BLOCK = 64  # CSA_INDEX_BLOCK_SIZE: entries a paged block
TILE = 32
SENTINEL = -7.0  # every pool row starts as this; a block the sender must not touch keeps it
# name -> (row width, layout, bytes a page, pages a block)
POOLS = {
    "kv": (512, ttnn.ROW_MAJOR_LAYOUT, 512 * 2, BLOCK),
    "keys": (128, ttnn.TILE_LAYOUT, TILE * TILE * 2, BLOCK * 128 // (TILE * TILE)),
}
DESCRIPTOR = "/tmp/xhost_kv_descriptor.json"  # written by receiver_init.py on host 41, read by receiver_verify.py


def log(role: str, message: str) -> None:
    print(f"[{role} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def init_rank(expected: int, role: str) -> None:
    """Initialize the MPI context (opening a mesh would, but the rank is needed before) and check this is ``role``."""
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    size, rank = int(ttnn.distributed_context_get_size()), int(ttnn.distributed_context_get_rank())
    if size != 2 or rank != expected:
        sys.exit(f"{role}: needs rank {expected} of a 2-rank tt-run job (got rank {rank} of {size})")


def open_mesh(fabric: str, payload: int | None) -> ttnn.MeshDevice:
    """This rank's 8x4 Galaxy with the two-mesh fabric up. Both ranks must pass the same ``fabric`` / ``payload``:
    the control plane refuses ranks whose router settings (packet payload above all) differ."""
    router = ttnn.FabricRouterConfig()
    if payload:
        router.max_packet_payload_size_bytes = payload
    ttnn.set_fabric_config(
        getattr(ttnn.FabricConfig, fabric),
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router,
    )
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL, None),
    )


def close_mesh(mesh: ttnn.MeshDevice) -> None:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def block_data(seed: int, round_: int, chip: int, n_blocks: int, width: int) -> torch.Tensor:
    """The bf16 ``[n_blocks, 1, 64, width]`` a sender chip writes in a round (``chip``: its row-major mesh index, the
    same as its receiver's). Both sides regenerate it, so nothing but addresses crosses TCP."""
    g = torch.Generator().manual_seed(((seed * 1009 + round_) * 1013 + chip) * 31 + width)
    return torch.randn(n_blocks, 1, BLOCK, width, generator=g).to(torch.bfloat16)


def expected_pool(seed: int, chip: int, pool_blocks: int, width: int, writes: list[list[int]]) -> torch.Tensor:
    """A receiver chip's whole pool after the rounds ``writes`` (round r wrote ``writes[r]``): ``SENTINEL`` where no
    round wrote, else the last writer's data."""
    pool = torch.full((pool_blocks, 1, BLOCK, width), SENTINEL, dtype=torch.bfloat16)
    for round_, blocks in enumerate(writes):
        pool[blocks] = block_data(seed, round_, chip, len(blocks), width)
    return pool


def page_bytes_of(blocks: torch.Tensor, layout) -> torch.Tensor:
    """The DRAM pages of bf16 ``[n, 1, 64, w]`` blocks as uint8 ``[n * pages_per_block, page_bytes]``, in page
    order: a ROW_MAJOR page is one row; a TILE page is one 32x32 tile (tiles row-major over the block, each tile its
    four 16x16 faces in order, each face row-major)."""
    n, _, rows, w = blocks.shape
    x = blocks.contiguous().view(torch.int16)
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return x.reshape(n * rows, w).view(torch.uint8)
    f = TILE // 2
    # [n, tile row, face row, row in face, tile col, face col, col in face]
    t = x.reshape(n, rows // TILE, 2, f, w // TILE, 2, f).permute(0, 1, 4, 2, 5, 3, 6)
    return t.reshape(n * (rows // TILE) * (w // TILE), TILE * TILE).contiguous().view(torch.uint8)


def page_location(page: int, banks: int, base: int, page_bytes: int) -> tuple[int, int]:
    """(bank, address in the bank) of page ``page`` of an interleaved DRAM buffer at ``base``."""
    return page % banks, base + (page // banks) * page_bytes


def connect(host: str, port: int, timeout: float = 600.0) -> socket.socket:
    """Connect to the sender's listener, retrying while it is still opening its mesh."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)
