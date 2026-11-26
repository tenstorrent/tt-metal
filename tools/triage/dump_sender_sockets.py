#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_sender_sockets

Description:
    Socket sender flow-control state from the L1 config buffers, one row per sender core owned by
    this rank. Cores come from the Inspector getSockets RPC. Coordinates are logical, and a core
    whose config buffer could not be read gets no row.

    A sender core feeding several downstreams gets one row each, numbered in Downstream #, with Peer
    naming the core that row describes. Match Downstream Addr to a Config Addr in
    dump_receiver_sockets to pair the two halves of a socket (sockets, not cores). Node is a fabric
    node id, the one chip name shared across ranks; a peer on this host shows its device id instead.
    A D2H socket has only its device side: its Peer reads host, and its Downstream Addr reads N/A.

    Counter columns, by source:

      sent     bytes_sent                                    kernel frame, or L1 for D2H
      acked    sender_socket_md.bytes_acked_array[i]         L1, written by the receiver (the host for D2H)
      wr_off   write_ptr, offset from downstream_fifo_addr   kernel frame

    socket_provider describes how kernel frame columns are read.

Owner:
    onenezicTT
"""

from run_checks import run as get_run_checks
from socket_provider import run as get_socket_provider
from triage import ScriptConfig, run_script
from ttexalens.context import Context

script_config = ScriptConfig(
    depends=["run_checks", "socket_provider"],
)


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    provider = get_socket_provider(args, context)
    return run_checks.run_per_block_check(provider.get_cached_senders, block_filter="tensix")


if __name__ == "__main__":
    run_script()
