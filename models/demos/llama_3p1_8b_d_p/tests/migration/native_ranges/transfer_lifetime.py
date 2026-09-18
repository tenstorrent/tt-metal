"""Both native endpoints must stop before either owner releases transport buffers."""
import hashlib
import json
import time
from pathlib import Path

from runner_support import require, require_clean_manager_exit, sha256, write_json
from supervision import root_cleanup_valid


def same_owner(a, b):
    return (
        isinstance(a, dict)
        and isinstance(b, dict)
        and all(a.get(k) == b.get(k) and a.get(k) is not None for k in ("pid", "start_ticks"))
    )


def started_manager(plan, role, owner, output):
    value = json.loads((Path(output) / "manager-started.json").read_bytes())
    require(
        isinstance(value, dict)
        and value.get("run_nonce") == plan["run_nonce"]
        and value.get("role") == role
        and value.get("endpoint") == plan[role]
        and same_owner(value.get("owner"), owner),
        "Foreign manager-started receipt",
    )
    identity = value.get("manager")
    require(
        isinstance(identity, dict)
        and all(type(identity.get(k)) is int and identity[k] > 0 for k in ("pid", "start_ticks")),
        "Malformed manager-started identity",
    )
    return identity


def stopped_receipt(plan, role, owner, manager, output, *, manager_safe):
    require(manager_safe and (manager is None or manager.poll() == 0), "Native stop remains ambiguous")
    log = Path(output) / "manager.log"
    if manager is not None:
        identity = started_manager(plan, role, owner, output)
        require(identity["pid"] == manager.pid, "Stopped manager differs from launched process")
        require_clean_manager_exit(manager.poll(), log.read_text(errors="replace"))
    else:
        require(not (Path(output) / "manager-started.json").exists(), "Cannot claim manager never started")
    return dict(
        run_nonce=plan["run_nonce"],
        role=role,
        ok=True,
        owner=owner,
        endpoint=plan[role],
        manager=None if manager is None else dict(pid=manager.pid, start_ticks=identity["start_ticks"], exit_code=0),
        manager_log=None if manager is None else dict(path=str(log), sha256=sha256(log)),
        manager_never_started=manager is None,
        native_io_stopped=True,
        device_reset_performed=False,
    )


def peer_stopped(plan, role):
    peer = "passive" if role == "source" else "source"
    run = Path(plan["run_dir"])
    try:
        value = json.loads((run / peer / "native-stopped.json").read_bytes())
        started = json.loads((run / (peer + "-supervisor") / "started.json").read_bytes())
        require(isinstance(value, dict) and isinstance(started, dict), "Malformed peer cleanup receipt")
        require(
            value.get("run_nonce") == started.get("run_nonce") == plan["run_nonce"]
            and value.get("role") == started.get("role") == peer,
            "Foreign peer cleanup receipt",
        )
        require(same_owner(value.get("owner"), started.get("owner")), "Peer owner identity differs")
        require(
            value.get("endpoint") == plan[peer]
            and value.get("ok") is True
            and value.get("native_io_stopped") is True
            and value.get("device_reset_performed") is False,
            "Peer native stop not proved",
        )
        manager = value.get("manager")
        require(
            manager is None
            or (
                isinstance(manager, dict)
                and type(manager.get("pid")) is int
                and manager["pid"] > 0
                and manager.get("exit_code") == 0
            ),
            "Peer manager exit not proved",
        )
        if manager is None:
            require(
                value.get("manager_never_started") is True and not (run / peer / "manager-started.json").exists(),
                "False peer never-started receipt",
            )
        else:
            identity = started_manager(plan, peer, value["owner"], run / peer)
            require(
                same_owner(manager, identity) and value.get("manager_never_started") is False,
                "Peer stopped a different manager generation",
            )
            log = run / peer / "manager.log"
            require(value.get("manager_log") == dict(path=str(log), sha256=sha256(log)), "Peer native log differs")
            require_clean_manager_exit(manager["exit_code"], log.read_text(errors="replace"))
        return value
    except FileNotFoundError:
        return None


def wait_peer_stopped(plan, role, timeout=180, clock=time.monotonic, sleep=time.sleep):
    # Failure/stop receipts intentionally do not interrupt this cleanup barrier.
    deadline = clock() + timeout
    while clock() < deadline:
        value = peer_stopped(plan, role)
        if value is not None:
            return value
        sleep(0.1)
    raise TimeoutError("Peer native stop not proved; both cache owners must remain retained")


def pair_cleanup_valid(receipt, owner, manager, plan, role):
    """Root cooperative recovery binds both owners and forbids closing a reset mesh."""
    pid, rc = (None, 0) if manager is None else (manager.pid, manager.poll())
    if not root_cleanup_valid(receipt, owner, plan["run_nonce"], pid, rc, plan[role]):
        return False
    try:
        raw = Path(receipt["evidence"]["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["evidence"]["sha256"]:
            return False
        proof = json.loads(raw)
        endpoints = proof.get("endpoints")
        if not isinstance(endpoints, dict) or set(endpoints) != {"source", "passive"}:
            return False
        for name in ("source", "passive"):
            row = endpoints[name]
            started = json.loads((Path(plan["run_dir"]) / (name + "-supervisor") / "started.json").read_bytes())
            if not (
                isinstance(row, dict)
                and isinstance(started, dict)
                and started.get("run_nonce") == plan["run_nonce"]
                and started.get("role") == name
                and same_owner(row.get("owner"), started.get("owner"))
                and row.get("endpoint") == plan[name]
                and row.get("native_io_stopped") is True
                and row.get("device_reset_performed") is False
            ):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return False


def await_pair_cleanup(path, owner, manager, plan, role, sleep=time.sleep):
    while True:
        try:
            receipt = json.loads(Path(path).read_bytes())
            if pair_cleanup_valid(receipt, owner, manager, plan, role):
                return receipt
        except (OSError, KeyError, TypeError, ValueError, AttributeError):
            pass
        sleep(1)


def publish_native_stopped(plan, role, owner, manager, output):
    path = Path(output) / "native-stopped.json"
    value = stopped_receipt(plan, role, owner, manager, output, manager_safe=True)
    if not path.exists():
        write_json(path, value)
    else:
        require(json.loads(path.read_bytes()) == value, "Native stop receipt changed")
    return value


def finish_native_pair(plan, role, owner, manager, output):
    publish_native_stopped(plan, role, owner, manager, output)
    return wait_peer_stopped(plan, role)


def check_unallocated_bootstrap_release(plan, role, output):
    """Each manager waits for both allocation receipts; absence of ours prevents peer startup."""
    peer = "passive" if role == "source" else "source"
    require(not (Path(output) / "allocation.json").exists(), "Allocation was published; peer stop proof required")
    require(
        not (Path(plan["run_dir"]) / peer / "manager-started.json").exists(),
        "Peer manager unexpectedly started before our allocation; retain bootstrap",
    )
