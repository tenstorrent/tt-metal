#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_receiver_sockets

Description:
    Socket receiver flow-control state from the L1 config buffers, one row per receiver core owned
    by this rank. Cores come from the Inspector getSockets RPC. Coordinates are logical, and a core
    whose config buffer could not be read gets no row.

    Match Config Addr to a Downstream Addr in dump_sender_sockets to pair the two halves of a socket
    (sockets, not cores). Node is a fabric node id, the one chip name shared across ranks; a peer on
    this host shows its device id instead. An H2D socket has only its device side, and its Peer
    reads host.

    Counter columns, by source:

      sent     receiver_socket_md.bytes_sent   L1, written by the sender (the host for H2D)
      acked    bytes_acked                     kernel frame, or L1 for H2D
      rd_addr  read_ptr, absolute L1 address   kernel frame

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
    return run_checks.run_per_block_check(provider.get_cached_receivers, block_filter="tensix")


if __name__ == "__main__":
    run_script()
