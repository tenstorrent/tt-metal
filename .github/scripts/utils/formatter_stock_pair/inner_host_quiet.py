"""Five stock-leg checkpoints through the existing owned host observer only."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import time
import traceback

import owned_seed_copy as owned

POINTS = ("initial-quiet", "pre-topology-quiet", "post-topology-quiet", "preflight", "post-cleanup")
PHASES = ("producer", "baseline", "candidate")
CHECKS_SHA = "59a6e7bc3eb779129244cf1d8b0c6925315dd6c322777417f5049d36e48825ba"
HELPER_SHA = "1f2676ea91090990a71244b1a19c1a56d93b600ae1e8401dc6ebabfa78e169b4"


def read(path, uid):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode) and before.st_uid == uid and before.st_size < 64 * 1024
        with os.fdopen(fd, "r", closefd=False) as stream:
            value = json.load(stream)
        assert owned.stamp(os.fstat(fd)) == owned.stamp(before), "Checkpoint receipt changed"
        return value
    finally:
        os.close(fd)


def publish(path, value):
    """Atomic visibility and exclusive final names; never overwrite cached proof."""
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as out:
            out.write(data)
            out.flush()
        identity = os.fstat(fd)
    finally:
        os.close(fd)
    os.link(temporary, path, follow_symlinks=False)
    final = temporary.lstat()
    assert (final.st_dev, final.st_ino, final.st_uid) == (identity.st_dev, identity.st_ino, identity.st_uid)
    temporary.unlink()  # Only this completed, positively identified own temp.


def binding(assignment, phase, point):
    assert phase in PHASES and point in POINTS
    assert assignment["driver_indices"] == [0, 1, 2, 3]
    return {
        "job_id": assignment["job_id"],
        "nonce": assignment["nonce"],
        "phase": phase,
        "checkpoint": point,
        "driver_indices": assignment["driver_indices"],
        "driver_rdevs": [row["rdev"] for row in assignment["drivers"]],
    }


def paths(scope, phase, point):
    assert phase in PHASES and point in POINTS and scope.is_dir() and not scope.is_symlink()
    return (
        scope / ("inner-" + phase + "-" + point + "-request.json"),
        scope / ("inner-" + phase + "-" + point + "-response.json"),
    )


def request(path, assignment, phase, seconds=105):
    """Called by the fixed root leg, while all model/native entry waits."""
    assert os.getuid() == 0 and seconds == 105
    assert path.suffix == ".json" and path.parent.name == ("baseline" if phase == "producer" else phase)
    point = path.stem
    request_path, response_path = paths(path.parent.parent, phase, point)
    assert not response_path.exists() and not response_path.is_symlink(), "Stale host response"
    client = owned.identity(os.getpid())
    assert client
    value = {**binding(assignment, phase, point), "client": client, "challenge": secrets.token_hex(16)}
    publish(request_path, value)
    end = time.monotonic() + seconds
    while not response_path.exists():
        assert not response_path.is_symlink(), "Host response link"
        assert time.monotonic() < end, "Fresh owned host checkpoint deadline"
        time.sleep(0.05)
    response = read(response_path, path.parent.parent.stat().st_uid)
    assert (
        response["request"] == value
        and response["request_sha256"] == hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    ), "Wrong or stale host result"
    ownership = response["ownership"]
    assert ownership["checks_sha256"] == CHECKS_SHA and ownership["helper_sha256"] == HELPER_SHA
    assert ownership["role"] == "inner-" + phase + "-" + point
    assert Path(ownership["operation_dir"]).parts[-2:] == (
        str(assignment["job_id"]) + "-" + assignment["nonce"],
        "evidence",
    )
    assert ownership["internal_deadline_seconds"] == 75
    assert ownership["closed"] and ownership["close"]["reaped"] and not ownership["close"]["escalated"]
    assert ownership["close"]["privileged_worker_absent"] and ownership["released"]
    receipt = response["receipt"]
    assert receipt["driver_rdevs"] == sorted(value["driver_rdevs"]) and not receipt["telemetry_exclusions"]
    owned.save(path, receipt)
    assert (
        response["code"] == 0
        and receipt["quiet"]
        and not receipt["holders"]
        and not receipt["incomplete_pid_visibility"]
    ), "Busy driver or incomplete host PID/FD visibility"
    return receipt


def serve(controller, scope, phase, item, handled, deadline):
    """One fixed container's five checkpoints; no arbitrary host operation."""
    assert phase in PHASES and scope == (
        controller.evidence / "producer" if phase == "producer" else controller.evidence
    )
    assert scope.is_dir() and not scope.is_symlink() and scope.stat().st_uid == os.getuid()
    for point in POINTS:
        request_path, response_path = paths(scope, phase, point)
        if not request_path.exists():
            assert not request_path.is_symlink(), "Inner request link"
            continue
        if point in handled:
            continue
        # Only original stock order, one fresh request per point.
        assert point == POINTS[len(handled)] and not response_path.exists() and not response_path.is_symlink()
        value = read(request_path, 0)
        expected = binding(controller.assignment, phase, point)
        assert all(value.get(k) == v for k, v in expected.items())
        assert set(value) == set(expected) | {"client", "challenge"}
        assert (
            isinstance(value["challenge"], str)
            and len(value["challenge"]) == 32
            and all(c in "0123456789abcdef" for c in value["challenge"])
        )
        row = controller.inspect(item)
        assert row["State"]["Running"] and row["State"]["Pid"] > 0
        client = value["client"]
        assert set(client) == {"pid", "birth_ticks", "state"}
        assert owned.same_process(client), "Checkpoint client birth changed"
        proc = Path("/proc") / str(client["pid"])
        assert proc.stat().st_uid == 0
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        assert int(fields[1]) == row["State"]["Pid"], "Checkpoint client is not this container init child"
        role = "inner-" + phase + "-" + point
        # This is the SAME225c/0433 host-root observation and ordinary-close proof.
        assert deadline - time.monotonic() > 105, "Original container/controller checkpoint budget exhausted"
        receipt = controller.probe(None, role)
        ownership = read(controller.evidence / (role + "-ownership.json"), os.getuid())
        response = {
            "request": value,
            "request_sha256": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
            "receipt": receipt,
            "ownership": ownership,
            "code": 0 if receipt["quiet"] else 2,
        }
        publish(response_path, response)
        handled.add(point)
        # Actual holders/gaps close the pair; this response cannot admit the leg.
        assert (
            receipt["quiet"] and not receipt["holders"] and not receipt["incomplete_pid_visibility"]
        ), "Busy driver or incomplete host PID/FD visibility"


def source_frames(error):
    return [
        {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
        for frame in traceback.extract_tb(error.__traceback__)
    ]


def report_error(base, phase, error):
    """Source frames only: no locals, exception arguments, environment or credentials."""
    assert phase in PHASES
    output = base / "evidence" / ("baseline" if phase == "producer" else phase)
    output.mkdir(parents=True, exist_ok=True)
    frames = source_frames(error)
    record = {
        "complete": False,
        "phase": phase,
        "error_type": type(error).__name__,
        "source_frames": frames,
        "candidate_prohibited": True,
    }
    owned.save(output / "leg-incomplete.json", record)
    print("Stopped diagnostic leg: " + record["error_type"] + "; source frames " + json.dumps(frames), flush=True)
    return record
