#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal ttnn distributed context test.
Tests ttnn's MPI-based distributed context without full device init.
"""

import os
import socket
import sys


def main():
    hostname = socket.gethostname()
    pid = os.getpid()

    print(f"[{hostname}] PID {pid}: Starting ttnn distributed test", flush=True)

    # Print relevant environment variables set by tt-run
    mesh_id = os.environ.get("TT_MESH_ID", "NOT_SET")
    mesh_graph_desc = os.environ.get("TT_MESH_GRAPH_DESC_PATH", "NOT_SET")
    mesh_host_rank = os.environ.get("TT_MESH_HOST_RANK", "NOT_SET")

    print(f"[{hostname}] TT_MESH_ID={mesh_id}", flush=True)
    print(f"[{hostname}] TT_MESH_HOST_RANK={mesh_host_rank}", flush=True)
    print(f"[{hostname}] TT_MESH_GRAPH_DESC_PATH={mesh_graph_desc}", flush=True)

    try:
        print(f"[{hostname}] Importing ttnn...", flush=True)
        import ttnn

        print(f"[{hostname}] ttnn imported successfully", flush=True)

        # Initialize distributed context manually (without opening device)
        print(f"[{hostname}] Initializing distributed context...", flush=True)
        ttnn.init_distributed_context()
        print(f"[{hostname}] Distributed context initialized!", flush=True)

        if ttnn.distributed_context_is_initialized():
            rank = int(ttnn.distributed_context_get_rank())
            size = int(ttnn.distributed_context_get_size())
            print(f"[{hostname}] Rank {rank}/{size}: Context is ready!", flush=True)

            # Test barrier
            print(f"[{hostname}] Rank {rank}: Entering barrier...", flush=True)
            ttnn.distributed_context_barrier()
            print(f"[{hostname}] Rank {rank}: Passed barrier!", flush=True)

            print(f"[{hostname}] Rank {rank}: TTNN DISTRIBUTED TEST PASSED!", flush=True)
        else:
            print(f"[{hostname}] ERROR: Distributed context not initialized", flush=True)
            sys.exit(1)

    except Exception as e:
        import traceback

        print(f"[{hostname}] ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
