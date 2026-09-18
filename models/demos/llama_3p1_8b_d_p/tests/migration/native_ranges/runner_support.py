"""Small stdlib process/receipt helpers; importing this module opens no native runtime."""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_jit_cache(output, environ):
    # prefill_env.sh assigns the historical shared cache. Override it after sourcing, before native imports.
    path = Path(output) / "jit-cache"
    path.mkdir(exist_ok=False)
    require(not any(path.iterdir()), "Attempt-local JIT cache was not empty")
    environ["TT_METAL_CACHE"] = str(path.resolve())
    return dict(path=str(path.resolve()), empty_before=True)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    require(not path.exists(), "Refusing to replace a preserved receipt")
    temp.rename(path)


def wait_receipt(path, nonce, timeout=300, check=lambda: None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check()
        if Path(path).exists():
            value = json.loads(Path(path).read_bytes())
            require(value.get("run_nonce") == nonce, "Stale or foreign receipt")
            require(value.get("ok") is True, "Peer phase failed")
            return value
        time.sleep(0.05)
    raise TimeoutError("Receipt not produced: " + str(path))


def require_clean_manager_exit(returncode, log):
    # stop() joins the control thread, which clears isRunning before the optional
    # "KV manager stopped" log. Require actual exit and native destruction instead.
    # DmkDeviceIO destroys every DmkLink before its Cluster; failed drain/disconnect
    # is logged and swallowed by DmkLink, so exit0 alone is insufficient.
    require(returncode == 0, "Manager did not exit successfully; allocations must remain retained")
    failures = (
        "drain failed",
        "disconnect failed",
        "service thread join failed",
        "[LinkWatchdog] stop failed",
        "[FATAL]",
        "terminating (fatal)",
        "KV manager failed:",
    )
    require(not any(word.lower() in log.lower() for word in failures), "Manager shutdown diagnostics reject release")
    events = re.findall(r"Cluster (constructor|destructor) (started|completed)\.", log)
    cycle = [
        ("constructor", "started"),
        ("constructor", "completed"),
        ("destructor", "started"),
        ("destructor", "completed"),
    ]
    require(events == cycle, "Native Cluster completion missing, unordered, or followed by reopen")


def mapped_libraries(pid, expected):
    """Require exact resolved file hashes, not just RUNPATH or link-time flags."""
    maps = Path(f"/proc/{pid}/maps").read_text()
    entries = [
        line.split() for line in maps.splitlines() if len(line.split()) >= 6 and line.split()[-1].startswith("/")
    ]
    paths = {row[-1] for row in entries}
    result = {}
    for soname, pin in expected.items():
        candidates = {
            str(Path(p).resolve()) for p in paths if Path(p).name == soname or Path(p).name.startswith(soname + ".")
        }
        target = str(Path(pin["path"]).resolve())
        require(candidates == {target}, "Unexpected loaded library for " + soname)
        require(sha256(target) == pin["sha256"], "Loaded library bytes changed: " + target)
        stat = Path(target).stat()
        require(
            {int(row[4]) for row in entries if str(Path(row[-1]).resolve()) == target} == {stat.st_ino},
            "Loaded inode differs from current library: " + target,
        )
        result[soname] = dict(path=target, sha256=pin["sha256"], inode=stat.st_ino, device=stat.st_dev)
    return dict(pid=pid, maps=maps, libraries=result)


class Bridge:
    def __init__(self, executable, config, output, env, check=lambda: None):
        self.check = check
        self.check()
        self.output = Path(output)
        self.journal = self.output / "bridge.jsonl"
        config = dict(config, journal=str(self.journal))
        path = self.output / "bridge-config.json"
        write_json(path, config)
        self.log = (self.output / "bridge.log").open("xb")
        self.process = subprocess.Popen(
            [str(executable), str(path)], stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT, env=env
        )
        self.sequence = 0
        self.terminal_exit_code = None
        try:
            self.wait(lambda rows: any(r.get("event") == "bridge_ready" for r in rows), 150)
        except BaseException:
            # This child owns only client sockets/counter SHM, never device allocations or DMK kernels.
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process.stdin.close()
            self.log.close()
            raise

    def rows(self):
        if not self.journal.exists():
            return []
        # Only complete newline-terminated records are visible; an in-progress write is not malformed evidence.
        raw = self.journal.read_bytes()
        return [json.loads(row) for row in raw.split(b"\n")[:-1] if row]

    def _record_exit(self, returncode, rows):
        self.terminal_exit_code = returncode
        path = self.output / "bridge-exit.json"
        if not path.exists():
            write_json(
                path,
                dict(
                    exit_code=returncode,
                    pid=self.process.pid,
                    complete_rows=len(rows),
                    error_rows=sum(r.get("event") == "error" for r in rows),
                    journal_bytes=self.journal.stat().st_size if self.journal.exists() else 0,
                ),
            )

    def wait(self, predicate, timeout=150, terminal=False):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check()
            rows = self.rows()
            returncode = self.process.poll()
            if returncode is not None:
                # A terminal child can append its reply and exit between the first read and poll.
                # Re-read only complete records, record the actual status, and never let a reply
                # conceal a nonzero exit or explicit error record.
                rows = self.rows()
                self._record_exit(returncode, rows)
                require(
                    not any(r.get("event") == "error" for r in rows),
                    f"Bridge reported an error; exit_code={returncode}",
                )
                require(returncode == 0, f"Bridge exited unsuccessfully; exit_code={returncode}")
                require(terminal and predicate(rows), f"Bridge exited before requested phase; exit_code={returncode}")
                return rows
            if any(r.get("event") == "error" for r in rows):
                try:
                    returncode = self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("Bridge reported an error; exit_code=running")
                rows = self.rows()
                self._record_exit(returncode, rows)
                raise RuntimeError(f"Bridge reported an error; exit_code={returncode}")
            if predicate(rows):
                return rows
            time.sleep(0.005)
        raise TimeoutError("Bridge response timed out")

    def rpc(self, op, **fields):
        self.check()
        self.sequence += 1
        request = dict(op=op, id=self.sequence, **fields)
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        self.process.stdin.flush()
        rows = self.wait(
            lambda rows: any(r.get("reply") == self.sequence for r in rows), terminal=op in ("drain", "drain_cancelled")
        )
        return next(r for r in rows if r.get("reply") == self.sequence)

    def snapshot_until(self, predicate, timeout=300):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.rpc("snapshot")
            if predicate(value):
                return value
            time.sleep(0.01)
        raise TimeoutError("Bridge progress timed out")

    def drain(self):
        result = self.rpc("drain")
        returncode = self.process.wait(timeout=150)
        self._record_exit(returncode, self.rows())
        require(returncode == 0, f"Bridge drain failed; exit_code={returncode}")
        self.process.stdin.close()
        self.log.close()
        result = dict(result)
        result["bridge_exit_code"] = returncode
        return result


def validate_plan(plan):
    from transfer_contract import validate_plan as check

    return check(plan)


class CapturedCompletionSink:
    """Capture one synchronized chunk before publishing any of its real layer callbacks."""

    def __init__(self, recorder, capture, push, clock=time.monotonic_ns, check=lambda: None):
        self.recorder, self.capture, self.push, self.clock, self.check = recorder, capture, push, clock, check
        self.snapshots, self.published = [], []

    def __call__(self, layer, request):
        self.check()
        require(self.recorder.completed_ns is not None, "Snapshot/readiness preceded model synchronization")
        require(
            self.recorder.active is not None
            and request == self.recorder.active[0]
            and type(layer) is int
            and 0 <= layer < 32
            and layer == len(self.recorder.rows) - request * 32,
            "Wrong or duplicate layer callback before snapshot",
        )
        if layer == 0:
            row = self.capture(request)
            row["snapshot_complete_ns"] = self.clock()
            self.snapshots.append(row)
        self.check()
        self.recorder.ack(layer, request, self.clock())
        self.check()
        self.push(layer, request)
        self.published.append(dict(request_id=request, layer=layer, published_ns=self.clock()))


def verify_source_terminal(value, uuids, slot_map=(1, 0)):
    require(
        value.get("successful") is True and value.get("acks") == 128 and value.get("retired") is True,
        "Source did not retire all 128 real routed acknowledgments",
    )
    slots = value["slots"]
    require(len(slots) == 2 and {r["slot"] for r in slots} == {0, 1}, "Wrong source slot inventory")
    require(
        all(r["position"] == 2048 and r["pins"] == r["in_flight"] == 0 for r in slots), "Source lifetime not drained"
    )
    calls = value["calls"]
    registers = [c for c in calls if c["op"] == "register"]
    require(
        len(registers) == 2
        and {(r["src"], r["uuid"], r["reused"]) for r in registers} == {(s, uuids[s], 0) for s in (0, 1)},
        "Native registration identity differs",
    )
    require(len({r["transfer"] for r in registers}) == 2, "Transfer IDs collide")
    for registration in registers:
        slot, transfer = registration["src"], registration["transfer"]
        own = [c for c in calls if c.get("transfer") == transfer]
        ready = [c for c in own if c["op"] == "peer_ready"]
        require(
            len(ready) == 1
            and (ready[0]["src"], ready[0]["dst"], ready[0]["from"], ready[0]["to"]) == (slot, slot_map[slot], 0, 2048),
            "Destination announce differs",
        )
        actual = [(c["src"], c["dst"], c["layer"], c["from"], c["to"]) for c in own if c["op"] == "layer"]
        expected = [(slot, slot_map[slot], layer, begin, begin + 1024) for begin in (0, 1024) for layer in range(32)]
        require(actual == expected, "Missing, duplicated or reordered production layer command")
        require(sum(c["op"] == "seal" for c in own) == 1, "Burst was not sealed exactly once")
        completed = [c for c in own if c["op"] == "completion"]
        require(
            len(completed) == 1
            and (completed[0]["slot"], completed[0]["status"], completed[0]["tokens"]) == (slot, 0, 2048),
            "Native source terminal differs",
        )
        order = [c["op"] for c in own]
        require(
            order.index("register") < order.index("peer_ready") < order.index("layer")
            and len(order) - 1 == order.index("completion")
            and order[-2] == "seal",
            "Terminal preceded seal or registration",
        )
    require(sum(c["op"] == "layer" for c in calls) == 128 and len(calls) == 136, "Unexpected native client calls")
