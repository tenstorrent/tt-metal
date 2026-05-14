# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_20: Teardown parallel heartbeat poll converges within budget (FIX AD)
#
# Strategy: Run mesh open + AllGather + close as a subprocess, measuring
# total elapsed time. The subprocess must complete within 30s. If FIX AD is
# reverted (reverting to sequential heartbeat polling), teardown with N MMIO
# ETH cores times out at N * per_core_timeout, which can exceed 30s on T3K.
#
# Pass = subprocess exits 0 within 30s AND teardown elapsed < 15s.
# Fail = subprocess times out or teardown elapsed > 15s (sequential poll regression).
#
# Background — FIX AD:
#   In risc_firmware_initializer.cpp teardown, FIX AD replaces the blind
#   sleep(500ms) with a 2-phase parallel heartbeat poll across all MMIO ETH
#   cores simultaneously (kBulkPollMs=5000 budget). Pre-FIX AD the poll was
#   sequential: each core waited up to 1s individually, so N slow ERISCs
#   could take N seconds total. Post-FIX AD all cores are polled in parallel
#   within the same 5s window, so one slow ERISC does not block the others.

import sys
import subprocess
import textwrap
import time

import pytest


_HELPER_SCRIPT = textwrap.dedent("""
    import ttnn
    import torch
    import sys
    import time
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
        num_devices = mesh.get_num_devices()

        # Activate all ERISC channels with one AllGather
        t = torch.randn(1, 1, 32, 32 * num_devices)
        tt_in = ttnn.from_torch(
            t,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ShardTensorToMesh(mesh, dim=3),
        )
        tt_out = ttnn.all_gather(tt_in, dim=3, num_links=1)
        ttnn.deallocate(tt_out)
        ttnn.deallocate(tt_in)

        # Close triggers quiesce + teardown -> heartbeat poll.
        # If FIX AD is reverted (sequential poll), teardown takes N * 1s per
        # MMIO ETH core. With 8 devices * 2 ETH channels = ~16s, well above 15s.
        # With parallel poll + 5s budget, total should be < 10s.
        teardown_start = time.time()
        ttnn.close_mesh_device(mesh)
        elapsed = time.time() - teardown_start

        print(f"TEARDOWN_ELAPSED={elapsed:.2f}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Caught exception: {e}", file=sys.stderr)
        sys.exit(1)
""")

_TEARDOWN_DEADLINE_S = 30
_TEARDOWN_SEQUENTIAL_REGRESSION_S = 15


@pytest.mark.parametrize("iteration", range(3))
def test_teardown_heartbeat_poll_parallel(iteration):
    """Teardown parallel heartbeat poll must not regress to sequential (FIX AD)."""
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", _HELPER_SCRIPT],
        timeout=_TEARDOWN_DEADLINE_S + 10,
        capture_output=True,
    )
    total_elapsed = time.time() - start

    stderr = result.stderr.decode(errors="replace")

    assert result.returncode == 0, (
        f"Iteration {iteration}: subprocess failed (rc={result.returncode}).\n"
        f"stderr: {stderr[-500:]}"
    )

    assert total_elapsed < _TEARDOWN_DEADLINE_S, (
        f"Iteration {iteration}: total elapsed {total_elapsed:.1f}s exceeded "
        f"{_TEARDOWN_DEADLINE_S}s deadline — probable teardown hang (FIX AD regression)."
    )

    # Extract teardown-only elapsed time from subprocess stderr
    for line in stderr.splitlines():
        if "TEARDOWN_ELAPSED=" in line:
            teardown_elapsed = float(line.split("=")[1])
            assert teardown_elapsed < _TEARDOWN_SEQUENTIAL_REGRESSION_S, (
                f"Iteration {iteration}: teardown took {teardown_elapsed:.1f}s > "
                f"{_TEARDOWN_SEQUENTIAL_REGRESSION_S}s — probable sequential heartbeat "
                f"poll regression (FIX AD reverted)."
            )
            break
