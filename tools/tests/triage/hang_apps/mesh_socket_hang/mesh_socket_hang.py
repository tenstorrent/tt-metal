#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Triage hang app: wedges MeshSockets so dump_mesh_sockets has live flow-control state to read.

Needs two chips (n300). Builds a socket pair in each direction and drives exactly one half of each,
so a single triage run sees four endpoints in two distinct flow-control states:

    dev0 -> dev1   send_async with no matching recv_async. The sender pushes until the fifo is full,
                   then wedges in socket_reserve_pages waiting for acks that never come, so this
                   receiver's bytes_sent sits at fifo_size (backpressure).

    dev1 -> dev0   recv_async with no matching send_async. Nothing is ever sent, so this receiver
                   stays at zero (starvation).

Note both dispatches target dev0, and dispatch is in-order per device, so the recv program actually
queues behind the wedged sender and never launches. That does not change what triage sees: a starved
receiver and an untouched one have identical config buffers, since bytes_sent, bytes_acked and
read_ptr all still hold their host-init values.

Holds the device open once wedged, so triage can run against it. The caller decides how long to wait
before triaging.
"""

import time

import ttnn

FIFO_SIZE = 8192
TENSOR_SHAPE = (1, 1, 256, 128)  # bfloat16 -> 64 KB, 8x the fifo, so the sender cannot drain it
# Distinct cores per device so the two sockets never share one.
FWD_SENDER_CORE = ttnn.CoreCoord(0, 0)  # on dev0
FWD_RECEIVER_CORE = ttnn.CoreCoord(1, 1)  # on dev1
REV_SENDER_CORE = ttnn.CoreCoord(2, 2)  # on dev1
REV_RECEIVER_CORE = ttnn.CoreCoord(3, 3)  # on dev0
HOLD_SECONDS = 1800


def make_socket(sender, receiver, sender_core, receiver_core):
    connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(coord, sender_core),
            ttnn.MeshCoreCoord(coord, receiver_core),
        )
        for coord in ttnn.MeshCoordinateRange(sender.shape)
    ]
    config = ttnn.SocketConfig(connections, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, FIFO_SIZE))
    return ttnn.create_socket_pair(sender, receiver, config)


def main() -> int:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    dev0 = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
    dev1 = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))

    fwd_send, _fwd_recv = make_socket(dev0, dev1, FWD_SENDER_CORE, FWD_RECEIVER_CORE)
    _rev_send, rev_recv = make_socket(dev1, dev0, REV_SENDER_CORE, REV_RECEIVER_CORE)

    def tensor_on(device):
        return ttnn.zeros(ttnn.Shape(TENSOR_SHAPE), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Backpressure: only the sender half of dev0 -> dev1 runs, so the fifo fills and stays full.
    ttnn.experimental.send_async(tensor_on(dev0), fwd_send)
    # Starvation: only the receiver half of dev1 -> dev0 runs, so it waits on a page that never comes.
    ttnn.experimental.recv_async(tensor_on(dev0), rev_recv)

    time.sleep(HOLD_SECONDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
