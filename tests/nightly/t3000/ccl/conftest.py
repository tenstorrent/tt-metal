"""
Conftest for tests/nightly/t3000/ccl — session-level cluster health check.

Before any test opens a device, verify the cluster is healthy.  If the fabric
is degraded (stale base-UMD channels from a prior session that was killed
without quiesce teardown), run ``tt-smi -r`` to reset all chips back to a
clean state, then wait for the driver to re-enumerate them.

This means tests never skip due to pre-existing hardware stale state — they
get a clean cluster and run properly.  Mid-session stale state (if a test
itself SIGKILL'd a subprocess that left stale ERISC firmware) is still caught
by the per-test ``is_fabric_degraded()`` guards in each GAP test, which skip
rather than hang (resetting with an open ``mesh_device`` in scope is not safe).
"""

import subprocess
import time

import pytest

# Mesh shape used by all t3000 CCL GAP tests.
_HEALTH_CHECK_MESH_SHAPE = (1, 8)
# Seconds to wait after tt-smi -r before trying to open a device again.
_POST_RESET_WAIT_S = 15


@pytest.fixture(scope="session", autouse=True)
def ensure_cluster_healthy():
    """
    Session-scoped autouse fixture: detect and fix degraded fabric *before*
    any test opens a device.

    Sequence:
      1. Open a mesh with all 8 devices.
      2. Call ``mesh_device.is_fabric_degraded()``.
      3. Close the mesh (regardless of outcome).
      4. If degraded → ``tt-smi -r`` → wait ``_POST_RESET_WAIT_S`` seconds.
      5. Yield (all tests run after this point on clean hardware).
    """
    import ttnn

    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_HEALTH_CHECK_MESH_SHAPE))
        degraded = mesh.is_fabric_degraded()
        ttnn.close_mesh_device(mesh)
    except Exception as exc:
        # If we can't even open the device the test-level fixtures will fail
        # with a clearer error; just warn and continue.
        print(f"\n[conftest] WARNING: could not open mesh for health check: {exc}")
        yield
        return

    if not degraded:
        yield
        return

    # ── Cluster is degraded — reset hardware before running any test ─────────
    print(
        "\n[conftest] Cluster degraded (stale base-UMD channels detected). "
        "Running tt-smi -r to restore clean state…"
    )
    try:
        result = subprocess.run(
            ["tt-smi", "-r"],
            timeout=120,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "[conftest] WARNING: tt-smi not found — cannot reset hardware. "
            "Tests may skip due to degraded fabric."
        )
        yield
        return
    except subprocess.TimeoutExpired:
        pytest.fail("[conftest] tt-smi -r timed out after 120 s — hardware unresponsive.")

    if result.returncode != 0:
        pytest.fail(
            f"[conftest] tt-smi -r failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    print(f"[conftest] tt-smi -r succeeded. Waiting {_POST_RESET_WAIT_S}s for re-enumeration…")
    time.sleep(_POST_RESET_WAIT_S)
    print("[conftest] Cluster reset complete — proceeding with tests on clean hardware.")
    yield
