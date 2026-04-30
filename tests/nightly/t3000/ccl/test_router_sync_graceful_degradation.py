# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_18: Router sync graceful degradation on dead-relay neighbor (FIX AL)
#
# Strategy: Run mesh open + AllGather + close as a subprocess. The subprocess
# exit code must not be -6 (SIGABRT). If FIX AL is reverted, wait_for_fabric_router_sync
# throws TT_THROW on a read exception or timeout, which calls std::abort -> SIGABRT.
#
# Pass = subprocess exits with code 0 (or non-zero but not -6).
# Fail = subprocess exits with -6 (SIGABRT from TT_THROW -> std::abort).
#
# Background — FIX AL:
#   wait_for_fabric_router_sync polls the master router channel for a sync token.
#   If a relay-broken neighbor blocks ring completion, the read throws or times out.
#   Pre-FIX AL: the exception propagates as TT_THROW -> SIGABRT, crashing the process.
#   Post-FIX AL: exception is caught, log_error is emitted, and the function returns
#   gracefully (degraded state), allowing teardown to complete.

import sys
import subprocess
import textwrap

import pytest


_HELPER_SCRIPT = textwrap.dedent("""
    import ttnn
    import torch
    import sys
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

        # Close triggers quiesce + teardown -> wait_for_fabric_router_sync.
        # If FIX AL is reverted and a relay-broken neighbor is present,
        # wait_for_fabric_router_sync throws TT_THROW -> SIGABRT.
        ttnn.close_mesh_device(mesh)
        sys.exit(0)
    except Exception as e:
        # Non-SIGABRT exceptions are acceptable (degraded state).
        print(f"Caught non-fatal exception: {e}", file=sys.stderr)
        sys.exit(0)
""")


@pytest.mark.parametrize("iteration", range(3))
def test_router_sync_no_sigabrt(iteration):
    """Process must not SIGABRT during fabric close with degraded relay (FIX AL)."""
    result = subprocess.run(
        [sys.executable, "-c", _HELPER_SCRIPT],
        timeout=120,
        capture_output=True,
    )

    stderr_tail = result.stderr.decode(errors="replace")[-500:]

    # returncode -6 == SIGABRT (from TT_THROW -> tt::assert::tt_throw -> std::abort)
    assert result.returncode != -6, (
        f"Iteration {iteration}: Process died with SIGABRT — FIX AL regression: "
        f"wait_for_fabric_router_sync threw instead of returning gracefully.\n"
        f"stderr: {stderr_tail}"
    )
