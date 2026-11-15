"""FIX.32 (2026-11-15): an ended agent's detached gate run is waited for, never run beside the orchestrator's gate."""
import subprocess
import sys
import time

from models.demos.common.bringup import orchestrator as O


def test_waits_for_detached_agent_gate():
    tid = "FIX.9999"
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(2)",
            "-m",
            "models.demos.common.bringup",
            "gate",
            tid,
            "--check",
        ],
        start_new_session=True,
    )
    try:
        time.sleep(0.3)
        assert p.pid in O.agent_gates(tid)
        assert O.agent_gates("FIX.9998") == []
        msgs = []
        t0 = time.time()
        assert O.wait_agent_gates(tid, msgs.append, poll_s=0.2) == 1
        assert p.poll() is not None and time.time() - t0 < 10
        assert msgs and "waiting for the agent's own gate run" in msgs[0]
    finally:
        p.kill()


def test_no_agent_gate_no_wait():
    assert O.wait_agent_gates("FIX.9997", lambda m: None, poll_s=0.1) == 0
