#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Triage hang app: puts MeshSockets on one chip in known states so dump_mesh_sockets has live state to read.

The D2D ops (send_async, recv_async) route through the fabric, which has no route from a chip to itself, so
the D2D sockets here carry no traffic: a 1:1 pair, and a socket whose single sender core feeds two receiver
cores so the dump has a multi-downstream core to report. They exist only to be inspected; the Inspector
records a socket when it is created, not when it is used. Nothing reaches their receivers, so those stay
at zero.

The traffic comes from the host sockets instead, one in each direction:

    H2D   The host writes half a fifo and nothing on the chip reads it, so this receiver's
          bytes_sent sits at fifo_size / 2.

    D2H   send_async_d2h sends more than the fifo holds and the host never reads, so the sender kernel
          fills the fifo, then wedges in socket_reserve_pages waiting for acks that never come
          (backpressure).

Takes the socket fifo size in bytes, so the caller asserting on it owns the value. Holds the device
open once wedged, so triage can run against it; the caller decides how long to wait before triaging.
"""

import os
import sys
import time

import torch
import ttnn

PAIR_SENDER_CORE = ttnn.CoreCoord(1, 0)
PAIR_RECEIVER_CORE = ttnn.CoreCoord(2, 1)
# One sender core feeding two receivers.
FANOUT_SENDER_CORE = ttnn.CoreCoord(3, 0)
FANOUT_RECEIVER_CORES = (ttnn.CoreCoord(3, 2), ttnn.CoreCoord(4, 1))
H2D_CORE = ttnn.CoreCoord(4, 2)
D2H_CORE = ttnn.CoreCoord(5, 1)
PAGE_SIZE = 256  # PCIe-aligned; the fifo holds a whole number of pages
HOLD_SECONDS = 1800


def on_chip(core):
    return ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), core)


def make_socket(mesh, fifo_size, sender_core, *receiver_cores):
    connections = [ttnn.SocketConnection(on_chip(sender_core), on_chip(core)) for core in receiver_cores]
    config = ttnn.SocketConfig(connections, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, fifo_size))
    return ttnn.create_socket_pair(mesh, mesh, config)


def main(fifo_size: int) -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    # _pair, _fanout held only so the sockets stay alive for the triage run.
    _pair = make_socket(mesh, fifo_size, PAIR_SENDER_CORE, PAIR_RECEIVER_CORE)
    _fanout = make_socket(mesh, fifo_size, FANOUT_SENDER_CORE, *FANOUT_RECEIVER_CORES)
    h2d = ttnn.H2DSocket(mesh, on_chip(H2D_CORE), ttnn.BufferType.L1, fifo_size, ttnn.H2DMode.HOST_PUSH)
    d2h = ttnn.D2HSocket(mesh, on_chip(D2H_CORE), fifo_size)

    h2d.set_page_size(PAGE_SIZE)
    h2d.write_tensor(torch.zeros(fifo_size // 2, dtype=torch.uint8))

    # Full D2H fifo: two fifos' worth of pages, so the kernel wedges halfway. The only op, and it goes
    # last: the sockets write their config through the command queue, and nothing queued behind this
    # kernel ever runs.
    d2h.set_page_size(PAGE_SIZE)
    data = ttnn.from_torch(
        torch.zeros((2 * fifo_size // PAGE_SIZE, PAGE_SIZE), dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.experimental.send_async_d2h(data, d2h)

    # The test stops this app with SIGINT (KeyboardInterrupt). A normal exit would wait forever on the wedged
    # kernel while closing the device, so the test would kill it only after 20 s; exit at once instead.
    try:
        time.sleep(HOLD_SECONDS)
    except KeyboardInterrupt:
        os._exit(0)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Expected exactly one argument, got more.", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(int(sys.argv[1])))
