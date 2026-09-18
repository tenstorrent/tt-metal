"""Nonce-bound two-endpoint stop observation; no native imports or process signals."""

import json
from pathlib import Path

from runner_support import require, sha256, write_json


class PeerEndpointStopped(RuntimeError):
    pass


def _receipt(path, nonce, peer_role, kind):
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_bytes())
    require(value.get("run_nonce") == nonce, "Foreign/stale peer stop receipt: " + str(path))
    require(value.get("role") == peer_role, "Peer stop receipt role differs: " + str(path))
    if kind == "peer_recovery_hold" and value.get("phase") != "recovery_hold":
        return None
    return dict(kind=kind, peer_role=peer_role, path=str(path), sha256=sha256(path), receipt=value)


def read_peer_event(run, role, nonce):
    """Return the first fail-closed peer stop event in deterministic priority order."""
    require(role in ("source", "passive"), "Invalid endpoint role")
    peer = "passive" if role == "source" else "source"
    run = Path(run)
    candidates = (
        (run / peer / "failure.json", "peer_failure"),
        (run / peer / "recovery-required.json", "peer_recovery"),
        (run / (peer + "-supervisor") / "decision.json", "peer_recovery_hold"),
        (run / (peer + "-stop-requested.json"), "peer_stop"),
    )
    for path, kind in candidates:
        event = _receipt(path, nonce, peer, kind)
        if event is not None:
            return event
    return None


class PeerStopGuard:
    """Latch one peer stop observation and reject all later guarded work."""

    def __init__(self, run, role, nonce, output):
        self.run, self.role, self.nonce = Path(run), role, nonce
        self.output = Path(output)
        self.observed = None

    def check(self):
        if self.observed is None:
            event = read_peer_event(self.run, self.role, self.nonce)
            if event is not None:
                self.observed = dict(run_nonce=self.nonce, role=self.role, **event)
                write_json(self.output / "peer-stop-observed.json", self.observed)
        if self.observed is not None:
            raise PeerEndpointStopped(self.observed["kind"] + " observed from " + self.observed["peer_role"])


def guarded(check, operation, *args, **kwargs):
    """Check before issuing work and again after a potentially blocking operation."""
    check()
    result = operation(*args, **kwargs)
    check()
    return result
