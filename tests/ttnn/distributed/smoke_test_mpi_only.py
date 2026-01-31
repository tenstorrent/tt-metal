#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Basic MPI connectivity test - no ttnn, just MPI.
Tests if MPI processes can communicate across Galaxy hosts.
"""

import os
import socket
import sys


def main():
    hostname = socket.gethostname()
    pid = os.getpid()

    print(f"[{hostname}] PID {pid}: Starting MPI test", flush=True)

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"[{hostname}] Rank {rank}/{size}: MPI initialized successfully", flush=True)

        # Simple barrier test
        print(f"[{hostname}] Rank {rank}: Entering barrier...", flush=True)
        comm.Barrier()
        print(f"[{hostname}] Rank {rank}: Passed barrier!", flush=True)

        # Simple send/recv test
        if rank == 0:
            msg = f"Hello from {hostname} (rank 0)"
            print(f"[{hostname}] Rank 0: Sending message to rank 1...", flush=True)
            comm.send(msg, dest=1, tag=42)
            print(f"[{hostname}] Rank 0: Message sent!", flush=True)
        else:
            print(f"[{hostname}] Rank 1: Waiting for message from rank 0...", flush=True)
            msg = comm.recv(source=0, tag=42)
            print(f"[{hostname}] Rank 1: Received: '{msg}'", flush=True)

        comm.Barrier()
        print(f"[{hostname}] Rank {rank}: MPI TEST PASSED!", flush=True)

    except Exception as e:
        print(f"[{hostname}] ERROR: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
