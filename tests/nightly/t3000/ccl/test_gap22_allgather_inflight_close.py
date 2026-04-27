# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-22: AllGather interrupted mid-flight by mesh close (FIX AO + AP + AD)
#
# Strategy: subprocess dispatches AllGather (non-blocking), immediately closes
# mesh, then reopens. The close while AllGather is in-flight exercises
# FIX AO/AP (relay-broken termination skip) and FIX AD (heartbeat poll on
# dirty ERISC state). Repeated 5 times to widen the race window.
#
# Pass  = all 5 close+reopen cycles complete within budget.
# Fail  = SIGABRT (FIX AO/AP regression), hang (FIX AD regression), or
#         reopen failure (FIX AP stale ERISC not cleaned up).

import sys
import subprocess
import textwrap
import time

import pytest
from loguru import logger

_HELPER_SCRIPT = textwrap.dedent("""
    import ttnn
    import torch
    import sys
    import time
    from ttnn import ShardTensorToMesh

    NUM_CYCLES = 5
    REOPEN_DEADLINE_S = 45

    for cycle in range(NUM_CYCLES):
        try:
            mesh = ttnn.open_mesh_device(
                ttnn.MeshShape(1, 8),
                mesh_type=ttnn.MeshType.Ring,
            )
        except Exception as e:
            print(f"Cycle {cycle}: open_mesh_device failed: {e}", file=sys.stderr)
            sys.exit(2)

        num_devices = mesh.get_num_devices()
        if num_devices < 4:
            print(f"Only {num_devices} devices, need >=4", file=sys.stderr)
            sys.exit(77)  # skip sentinel

        # Dispatch AllGather non-blocking — creates in-flight ERISC traffic.
        t = torch.randn(1, 1, 32, 32 * num_devices, dtype=torch.bfloat16)
        tt_in = ttnn.from_torch(
            t,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ShardTensorToMesh(mesh, dim=3),
        )
        # Do NOT synchronize — leave AllGather potentially in-flight.
        tt_out = ttnn.all_gather(tt_in, dim=3, num_links=1)

        # Immediately close — exercises FIX AO/AP/AD on dirty ERISC state.
        close_start = time.time()
        ttnn.close_mesh_device(mesh)
        close_elapsed = time.time() - close_start

        print(
            f"Cycle {cycle}: close_elapsed={close_elapsed:.2f}s",
            file=sys.stderr,
        )
        if close_elapsed > REOPEN_DEADLINE_S:
            print(f"Cycle {cycle}: close too slow ({close_elapsed:.1f}s)", file=sys.stderr)
            sys.exit(3)

    sys.exit(0)
""")

_SUBPROCESS_TIMEOUT_S = 300  # 5 cycles * 45s max + overhead


@pytest.mark.parametrize("iteration", range(3))
def test_allgather_inflight_close(iteration):
    """GAP-22: AllGather in-flight at mesh close must not hang or SIGABRT."""
    result = subprocess.run(
        [sys.executable, "-c", _HELPER_SCRIPT],
        timeout=_SUBPROCESS_TIMEOUT_S,
        capture_output=True,
    )

    stderr_tail = result.stderr.decode(errors="replace")[-1000:]

    if result.returncode == 77:
        pytest.skip("Not enough devices for multi-chip test")

    # SIGABRT = -6: FIX AO/AP regression (termination writes to broken relay).
    assert result.returncode != -6, (
        f"Iteration {iteration}: SIGABRT — FIX AO/AP regression: "
        f"termination writes sent to relay-broken device.\n"
        f"stderr: {stderr_tail}"
    )

    assert result.returncode == 0, (
        f"Iteration {iteration}: subprocess exited {result.returncode}.\n"
        f"stderr: {stderr_tail}"
    )
