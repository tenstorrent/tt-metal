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

import signal
import subprocess
import time

import pytest

# Mesh shape used by all t3000 CCL GAP tests.
_HEALTH_CHECK_MESH_SHAPE = (1, 8)
# Seconds to wait after tt-smi -r before trying to open a device again.
_POST_RESET_WAIT_S = 15
# FIX GS (#42429): Maximum seconds allowed for the health-check open_mesh_device().
# On a healthy T3K cluster, Metal fabric init completes in ~5-10 s.  If it exceeds
# this threshold the cluster is in a dirty post-crash state where UMD's ETH relay
# read is blocking without throwing (so FIX AL in wait_for_fabric_router_sync can't
# help).  Treat any hang longer than this as "degraded" and go straight to tt-smi -r.
_HEALTH_CHECK_OPEN_TIMEOUT_S = 30


class _OpenMeshTimeout(Exception):
    """Raised by SIGALRM when open_mesh_device() in ensure_cluster_healthy hangs."""


class _OpenMeshBusError(Exception):
    """Raised by SIGBUS handler when open_mesh_device() hits a hardware bus fault."""


def _sigalrm_handler(signum, frame):  # noqa: ARG001
    raise _OpenMeshTimeout()


def _sigbus_handler(signum, frame):  # noqa: ARG001
    # FIX GS-3c (#42429): SIGBUS during warm-up open_mesh_device after tt-smi -r.
    # Hardware PCIe BAR mapping is invalid (device still in reset / re-enumerating).
    # Safe to raise Python exception here because SIGBUS came from UMD's MMIO access
    # (not from Python's own memory), so the Python interpreter itself is intact.
    raise _OpenMeshBusError()


def _run_smi_reset():
    """Run ``tt-smi -r`` and wait for re-enumeration.  Calls pytest.fail on error."""
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


@pytest.fixture(scope="session", autouse=True)
def ensure_cluster_healthy():
    """
    Session-scoped autouse fixture: detect and fix degraded fabric *before*
    any test opens a device.

    Sequence:
      1. Open a mesh with all 8 devices (with a 30 s SIGALRM timeout guard).
         FIX GS (#42429): if open_mesh_device() hangs > 30 s, hardware is in a
         dirty post-crash state where UMD relay reads block without throwing.
         Treat this as degraded rather than waiting the full ~300 s UMD timeout.
      2. Call ``mesh_device.is_fabric_degraded()``.
      3. Close the mesh (regardless of outcome).
      4. If degraded (or hung) → ``tt-smi -r`` → wait ``_POST_RESET_WAIT_S`` seconds.
      5. Yield (all tests run after this point on clean hardware).
    """
    import ttnn

    degraded = False
    prev_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
    signal.alarm(_HEALTH_CHECK_OPEN_TIMEOUT_S)
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_HEALTH_CHECK_MESH_SHAPE))
        signal.alarm(0)  # cancel alarm — open succeeded in time
        degraded = mesh.is_fabric_degraded()
        ttnn.close_mesh_device(mesh)
    except _OpenMeshTimeout:
        # FIX GS: open_mesh_device() hung for > _HEALTH_CHECK_OPEN_TIMEOUT_S seconds.
        # The mesh object was never fully constructed so there is nothing to close.
        # Hardware is in dirty post-crash state — go straight to tt-smi -r.
        signal.alarm(0)
        print(
            f"\n[conftest] FIX GS (#42429): open_mesh_device() did not complete within "
            f"{_HEALTH_CHECK_OPEN_TIMEOUT_S} s — hardware in dirty post-crash state "
            "(UMD ETH relay blocking without exception). Treating as degraded; "
            "running tt-smi -r before any test opens a device."
        )
        degraded = True
    except Exception as exc:
        signal.alarm(0)
        # If we can't even open the device the test-level fixtures will fail
        # with a clearer error; just warn and continue.
        print(f"\n[conftest] WARNING: could not open mesh for health check: {exc}")
        yield
        return
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

    if not degraded:
        yield
        return

    # ── Cluster is degraded (or hung) — reset hardware before running any test ─
    print(
        "\n[conftest] Cluster degraded (stale channels or mesh-open timeout detected). "
        "Running tt-smi -r to restore clean state…"
    )
    _run_smi_reset()

    # ── FIX GS-2b (#42429): warm-up cycle after tt-smi reset ─────────────────
    # tt-smi -r reloads base-UMD firmware on ALL ETH channels, including non-MMIO.
    # If the next mesh_device fixture opens with FABRIC_2D, Metal's ControlPlane
    # constructor crashes with SIGBUS because non-MMIO channels are still in
    # base-UMD state during FABRIC_2D init (metal_env.cpp:295 reinit path).
    #
    # Solution: open/close FABRIC_1D once — FIX M (device.cpp:664) transitions
    # all non-MMIO base-UMD channels via launch_msg, loading proper Metal firmware.
    # Close cleanly → firmware properly terminated.  After this warm-up, no
    # channels are in base-UMD state and the test's FABRIC_2D init succeeds.
    print(
        "\n[conftest] FIX GS-2b (#42429): warm-up open/close cycle (FABRIC_1D) "
        "to clear base-UMD state from tt-smi before FABRIC_2D tests run…"
    )
    warmup_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
    # FIX GS-3c (#42429): also catch SIGBUS — after tt-smi -r the PCIe BAR mapping may
    # be transiently invalid; UMD MMIO access during open_mesh_device triggers SIGBUS.
    # Python's signal handler delivers it as a Python exception at the next bytecode
    # boundary, allowing us to exit cleanly instead of crashing the process.
    prev_sigbus_handler = signal.signal(signal.SIGBUS, _sigbus_handler)
    signal.alarm(_HEALTH_CHECK_OPEN_TIMEOUT_S)
    try:
        warmup = ttnn.open_mesh_device(ttnn.MeshShape(*_HEALTH_CHECK_MESH_SHAPE))
        signal.alarm(0)
        # Audit: check if warm-up mesh is still degraded after tt-smi -r.
        # If so, the reset did not fully recover the hardware.
        if warmup.is_fabric_degraded():
            print(
                "[conftest] FIX GS-2b: WARNING: warm-up mesh still reports degraded "
                "fabric AFTER tt-smi -r — hardware may not fully recover for FABRIC_2D tests."
            )
        ttnn.close_mesh_device(warmup)
        print("[conftest] FIX GS-2b: warm-up complete — channels clean for FABRIC_2D.")
    except _OpenMeshTimeout:
        signal.alarm(0)
        print(
            "[conftest] FIX GS-2b: WARNING: warm-up timed out — "
            "FABRIC_2D tests may still hit Bus error."
        )
    except _OpenMeshBusError:
        signal.alarm(0)
        # FIX GS-3c (#42429): SIGBUS during warm-up means PCIe MMIO access to a device
        # in post-reset transition failed at hardware level.  tt-smi -r did not fully
        # restore the bus by the time we tried to open.  Abort the session cleanly so
        # CI reports a hardware failure rather than a core-dump timeout (exit 124).
        print(
            "\n[conftest] FIX GS-3c (#42429): warm-up crashed with SIGBUS — "
            "PCIe BAR access failed after tt-smi -r (hardware not yet re-enumerated).\n"
            "[conftest] Aborting session — board needs physical reset or longer re-enum wait."
        )
        signal.signal(signal.SIGBUS, prev_sigbus_handler)
        signal.signal(signal.SIGALRM, warmup_handler)
        pytest.exit(
            "FIX GS-3c: SIGBUS during warm-up after tt-smi -r — board needs physical reset",
            returncode=1,
        )
        return
    except Exception as exc:
        signal.alarm(0)
        exc_msg = str(exc)
        # FIX GS-3b (#42429): When warm-up fails with "failed to initialize FW", the hardware
        # is in a state where tt-smi -r did not recover it and no test can open a device.
        # Rather than yielding and letting each test fail individually (then getting killed by
        # the outer 300s bash timeout → exit code 124), abort the session cleanly so CI gets
        # an explicit failure message instead of a timeout kill.
        if "failed to initialize FW" in exc_msg or "Try resetting the board" in exc_msg:
            print(
                f"\n[conftest] FIX GS-3b (#42429): warm-up failed with fatal FW init error — "
                "hardware cannot be recovered by tt-smi -r alone (board needs physical reset).\n"
                f"Error: {exc_msg[:300]}\n"
                "[conftest] Aborting session to avoid outer bash timeout (exit code 124)."
            )
            signal.signal(signal.SIGBUS, prev_sigbus_handler)
            signal.signal(signal.SIGALRM, warmup_handler)
            pytest.exit(
                "FIX GS-3b: hardware FW init failed after tt-smi -r — board needs physical reset",
                returncode=1,
            )
            return
        print(
            f"[conftest] FIX GS-2b: WARNING: warm-up failed ({exc}) — "
            "FABRIC_2D tests may fail due to residual base-UMD channels."
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGBUS, prev_sigbus_handler)
        signal.signal(signal.SIGALRM, warmup_handler)

    yield
