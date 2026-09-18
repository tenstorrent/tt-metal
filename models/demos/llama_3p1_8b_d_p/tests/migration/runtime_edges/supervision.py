"""Bounded control phases; ambiguity transitions to retention, never forced owner termination."""
import hashlib
import json
import time
from pathlib import Path


def root_release_valid(receipt, identity, nonce, owner_exit, endpoint):
    if not isinstance(receipt, dict):
        return False
    if not (
        owner_exit is not None
        and receipt.get("run_nonce") == nonce
        and receipt.get("owner") == identity
        and receipt.get("native_io_stopped_or_reset") is True
    ):
        return False
    try:
        evidence = receipt["evidence"]
        if not isinstance(evidence, dict):
            return False
        path = Path(evidence["path"])
        if not path.is_absolute() or path.stat().st_size > 1048576:
            return False
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
            return False
        proof = json.loads(raw)
        if not isinstance(proof, dict):
            return False
        return (
            proof.get("run_nonce") == nonce
            and proof.get("owner") == identity
            and str(proof.get("job_id")) == str(endpoint["job_id"])
            and proof.get("node") == endpoint["host"]
            and proof.get("node_lock") == endpoint["node_lock"]
            and proof.get("native_io_stopped_or_reset") is True
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def supervise(
    process,
    read_state,
    request_stop,
    *,
    normal_seconds,
    cancel_seconds,
    lease_seconds,
    recovery_reserve,
    clock=time.monotonic,
    sleep=time.sleep,
    external_stop=lambda: False,
):
    if min(normal_seconds, cancel_seconds, recovery_reserve) <= 0:
        raise ValueError("Positive phase budgets required")
    if lease_seconds < normal_seconds + cancel_seconds + recovery_reserve:
        raise ValueError("Lease cannot cover work, cancellation and root recovery reserve")
    start = clock()
    stop_at = min(start + normal_seconds, start + lease_seconds - cancel_seconds - recovery_reserve)
    cancel_at = None
    reason = None
    while True:
        state = read_state()
        rc = process.poll()
        if rc is not None:
            final = state.get("result")
            if final and final.get("owner_cleanup_complete") is True and not final.get("cleanup_errors"):
                return dict(phase="finished", owner_exit=rc, release_lock=True, result=final)
            return dict(
                phase="recovery_hold",
                reason="owner_exited_without_clean_release",
                owner_exit=rc,
                release_lock=False,
                buffers_retained=False,
                evidence=state,
            )
        if state.get("recovery"):
            return dict(
                phase="recovery_hold",
                reason="owner_reports_ambiguous_native_shutdown",
                release_lock=False,
                buffers_retained=True,
                evidence=state,
            )
        now = clock()
        if cancel_at is None and (state.get("failure") or now >= stop_at or external_stop()):
            reason = "endpoint_failure" if state.get("failure") else "deadline_or_external_stop"
            request_stop(reason)
            cancel_at = now + cancel_seconds
        if cancel_at is not None and now >= cancel_at:
            return dict(
                phase="recovery_hold",
                reason="cooperative_cleanup_deadline",
                release_lock=False,
                buffers_retained=True,
                stop_reason=reason,
                evidence=state,
            )
        sleep(0.1)


def root_cleanup_valid(receipt, identity, nonce, manager_pid, manager_exit, endpoint):
    """Allow a retained live owner to close its mesh only after bound root proof.

    This does not release the supervisor lock or turn a failed attempt into success.
    A reset invalidates the live mesh, so this cooperative path requires no reset.
    """
    if not isinstance(receipt, dict):
        return False
    if manager_exit is None or receipt.get("action") != "close_owner_mesh":
        return False
    expected_manager = dict(pid=manager_pid, exit_code=manager_exit)
    try:
        evidence = receipt["evidence"]
        if not isinstance(evidence, dict):
            return False
        path = Path(evidence["path"])
        if not path.is_absolute() or path.stat().st_size > 1048576:
            return False
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
            return False
        proof = json.loads(raw)
        if not isinstance(proof, dict):
            return False
        for value in (receipt, proof):
            if not (
                value.get("run_nonce") == nonce
                and value.get("owner") == identity
                and value.get("manager") == expected_manager
                and value.get("native_io_stopped_or_reset") is True
                and value.get("device_reset_performed") is False
                and value.get("action") == "close_owner_mesh"
            ):
                return False
        return (
            str(proof.get("job_id")) == str(endpoint["job_id"])
            and proof.get("node") == endpoint["host"]
            and proof.get("node_lock") == endpoint["node_lock"]
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def await_root_cleanup(path, identity, nonce, manager, endpoint, sleep=time.sleep):
    """Preserve buffers on absent/stale proof or a live manager; root decides cleanup."""
    while True:
        try:
            receipt = json.loads(Path(path).read_bytes())
            if root_cleanup_valid(receipt, identity, nonce, manager.pid, manager.poll(), endpoint):
                return receipt
        except (OSError, TypeError, ValueError, AttributeError):
            pass
        sleep(1)
