import hashlib
import json
import os
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
