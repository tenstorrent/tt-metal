# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""H2D/D2H socket *server* (device side) for the abstract matmul op test.

This is the device-owning half of a two-process test that exercises the
host <-> device PCIe socket ops (``recv_async_h2d`` / ``send_async_d2h``) end to
end, mirroring the topology of the inworld_tts socket demo
(``socket_decoder.py`` / ``socket_speechlm.py``).

What it does:

  1. Asserts ``TT_VISIBLE_DEVICES`` pins this process to *exactly one* chip (each
     server must own its own device; the connector/driver owns none).
  2. Opens a 1x1 ``MeshDevice`` on that chip.
  3. Creates an ``H2DSocket`` (host -> device input) and a ``D2HSocket``
     (device -> host output) on disjoint cores.
  4. Generates a random ``(K, N)`` weight, **saves it to a .pth file** so the
     driver can compute the matmul reference, and uploads it to device DRAM.
  5. Exports both socket descriptors to ``/dev/shm`` so the driver can attach.
  6. Loops ``num_iterations`` times:
        recv (M, K) input (H2D) -> tilize -> matmul(weight) -> untilize
        -> send (M, N) result (D2H)

Start this server (one per chip) before the driver:

    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    TT_VISIBLE_DEVICES=0 python tests/ttnn/unit_tests/operations/ccl/h2d_d2h_matmul_server.py

The H2D/D2H socket connector path is Blackhole-only.
"""

import argparse
import os
import sys

import torch
from loguru import logger

import ttnn

# Defaults shared with the driver. Override on *both* processes if you change them.
DEFAULT_RUN_ID = "h2d_d2h_matmul"
DEFAULT_WEIGHT_PATH = "/dev/shm/tt_h2d_d2h_matmul_weight.pt"
DEFAULT_M, DEFAULT_K, DEFAULT_N = 32, 64, 64
DEFAULT_NUM_ITERATIONS = 8

_BYTES_PER_ELEMENT = 2  # bfloat16


def _socket_ids(run_id: str) -> tuple[str, str]:
    """Descriptor ids shared between the server and the driver."""
    return f"{run_id}_h2d", f"{run_id}_d2h"


def _require_single_visible_device() -> str:
    """Fail fast unless ``TT_VISIBLE_DEVICES`` names exactly one chip.

    Each socket server must own a single, distinct device. Counting on the user
    to set this correctly avoids accidentally opening (and locking) every chip,
    which would deadlock the other processes in the pipeline.
    """
    raw = os.environ.get("TT_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        raise RuntimeError(
            "TT_VISIBLE_DEVICES must be set to a single device id before running this "
            "server, e.g. `TT_VISIBLE_DEVICES=0 python .../h2d_d2h_matmul_server.py`."
        )
    ids = [tok for tok in raw.replace(",", " ").split() if tok != ""]
    if len(ids) != 1:
        raise RuntimeError(
            f"TT_VISIBLE_DEVICES must specify exactly one device, got {raw!r} "
            f"({len(ids)} devices). Pin this server to a single chip."
        )
    return ids[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="H2D/D2H socket matmul server (device side)")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID, help="Shared descriptor id prefix")
    parser.add_argument("--weight-path", type=str, default=DEFAULT_WEIGHT_PATH, help="Where to save the matmul weight")
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="Input rows")
    parser.add_argument("--K", type=int, default=DEFAULT_K, help="Contraction dim")
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="Output cols")
    parser.add_argument("--num-iterations", type=int, default=DEFAULT_NUM_ITERATIONS, help="Matmul iterations to serve")
    parser.add_argument("--seed", type=int, default=0xB1A5E, help="Weight RNG seed")
    args = parser.parse_args()

    visible_device = _require_single_visible_device()
    logger.info(f"Server pinned to TT_VISIBLE_DEVICES={visible_device}")

    M, K, N = args.M, args.K, args.N
    input_page_size_bytes = K * _BYTES_PER_ELEMENT
    output_page_size_bytes = N * _BYTES_PER_ELEMENT
    h2d_fifo_size_bytes = max(2048, input_page_size_bytes * 4)
    d2h_fifo_size_bytes = max(2048, output_page_size_bytes * 4)

    h2d_id, d2h_id = _socket_ids(args.run_id)

    mesh_device = None
    h2d_socket = None
    d2h_socket = None
    try:
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

        device_coord = ttnn.MeshCoordinate(0, 0)
        h2d_core = ttnn.MeshCoreCoord(device_coord, ttnn.CoreCoord(0, 0))
        d2h_core = ttnn.MeshCoreCoord(device_coord, ttnn.CoreCoord(1, 0))

        h2d_socket = ttnn.H2DSocket(
            mesh_device, h2d_core, ttnn.BufferType.L1, h2d_fifo_size_bytes, ttnn.H2DMode.HOST_PUSH
        )
        h2d_socket.set_page_size(input_page_size_bytes)

        d2h_socket = ttnn.D2HSocket(mesh_device, d2h_core, d2h_fifo_size_bytes)
        d2h_socket.set_page_size(output_page_size_bytes)

        # Random weight saved to disk so the driver can reproduce the reference
        # without any extra IPC, then uploaded once and reused every iteration.
        generator = torch.Generator().manual_seed(args.seed)
        torch_weight = torch.randn(K, N, generator=generator, dtype=torch.float32)
        torch.save(torch_weight, args.weight_path)
        logger.info(f"Saved {tuple(torch_weight.shape)} weight to {args.weight_path}")

        weight_tensor = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Publish the sockets only after the page size is set so the driver
        # inherits it through the descriptor.
        h2d_socket.export_descriptor(h2d_id)
        d2h_socket.export_descriptor(d2h_id)
        logger.info(
            f"Server ready: H2D={h2d_id} D2H={d2h_id}, M={M} K={K} N={N}, " f"serving {args.num_iterations} iterations"
        )

        for iteration in range(args.num_iterations):
            # Pre-allocate the row-major L1 tensor that recv_async_h2d fills; the
            # kernel parks at socket_wait_for_pages until the driver pushes pages.
            input_tensor_rm = ttnn.from_torch(
                torch.zeros(M, K, dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.experimental.recv_async_h2d(input_tensor_rm, h2d_socket)

            input_tensor_tile = ttnn.to_layout(input_tensor_rm, ttnn.TILE_LAYOUT)
            matmul_output_tile = ttnn.matmul(input_tensor_tile, weight_tensor)
            matmul_output_rm = ttnn.to_layout(matmul_output_tile, ttnn.ROW_MAJOR_LAYOUT)

            ttnn.experimental.send_async_d2h(matmul_output_rm, d2h_socket)

            # Drain this iteration's programs before reusing the sockets.
            ttnn.synchronize_device(mesh_device)
            logger.info(f"Served iteration {iteration + 1}/{args.num_iterations}")

        logger.info("Server done serving all iterations.")
        return 0
    finally:
        # Destroy sockets before closing the device so no MeshBuffer outlives its L1.
        del h2d_socket
        del d2h_socket
        if mesh_device is not None:
            try:
                ttnn.close_mesh_device(mesh_device)
            except Exception:
                logger.exception("close_mesh_device failed")


if __name__ == "__main__":
    sys.exit(main())
