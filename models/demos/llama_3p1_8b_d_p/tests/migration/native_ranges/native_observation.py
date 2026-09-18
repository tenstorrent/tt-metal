"""Read existing Linux process/SeatLock evidence; never open a TT device or a seat lock."""
import hashlib
import os
import re
from pathlib import Path

SEAT = re.compile(r"dmk-seat-(\d+)-(\d+)-(\d+)\.lock")


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def process_identity(pid, proc=Path("/proc")):
    text = (Path(proc) / str(pid) / "stat").read_text()
    end = text.rfind(")")
    fields = text[end + 2 :].split()
    require(end >= 0 and len(fields) > 19, "Malformed process identity")
    return dict(pid=int(pid), start_ticks=int(fields[19]), state=fields[0])


def inspect_seat_locks(pid, asic_ids, proc=Path("/proc"), lock_dir=Path("/dev/shm/tt-kv")):
    """Observe locks production DmkLink already holds, including PID/inode identity."""
    proc, lock_dir = Path(proc), Path(lock_dir)
    before = process_identity(pid, proc)
    expected = set(asic_ids)
    require(len(expected) == 32, "Full Galaxy ASIC inventory required")
    seats = []
    for fd in sorted((proc / str(pid) / "fd").iterdir(), key=lambda p: int(p.name)):
        try:
            target = Path(os.readlink(fd))
        except FileNotFoundError:
            continue
        match = SEAT.fullmatch(target.name)
        if not match:
            require("dmk-seat-" not in str(target), "Deleted or malformed seat descriptor")
            continue
        require(target.parent == lock_dir, "Seat uses a private/non-shared lock namespace")
        asic, x, y = map(int, match.groups())
        require(asic in expected, "Seat belongs to an unexpected ASIC")
        stat, path_stat = fd.stat(), target.stat()
        require((stat.st_dev, stat.st_ino) == (path_stat.st_dev, path_stat.st_ino), "Seat inode changed")
        info = (proc / str(pid) / "fdinfo" / fd.name).read_text()
        locks = []
        for line in info.splitlines():
            if not line.startswith("lock:"):
                continue
            fields = line.split()
            if len(fields) == 9 and fields[2:5] == ["FLOCK", "ADVISORY", "WRITE"]:
                major, minor, inode = fields[6].split(":")
                if (int(fields[5]), int(major, 16), int(minor, 16), int(inode), fields[7:]) == (
                    pid,
                    os.major(stat.st_dev),
                    os.minor(stat.st_dev),
                    stat.st_ino,
                    ["0", "EOF"],
                ):
                    locks.append(line)
        require(len(locks) == 1, "Seat FD lacks its own exclusive whole-file flock")
        seats.append(
            dict(
                asic_id=asic,
                x=x,
                y=y,
                fd=int(fd.name),
                path=str(target),
                device=stat.st_dev,
                inode=stat.st_ino,
                lock=locks[0],
            )
        )
    require(len(seats) == 64, "Expected exactly64 live data-plane seats")
    require(len({(s["asic_id"], s["x"], s["y"]) for s in seats}) == 64, "Duplicate seat descriptors")
    require(all(sum(s["asic_id"] == a for s in seats) == 2 for a in expected), "Expected two seats per ASIC")
    after = process_identity(pid, proc)
    require(
        before["start_ticks"] == after["start_ticks"] and after["state"] != "Z",
        "Manager changed/exited during observation",
    )
    return dict(
        process=after,
        lock_directory=str(lock_dir),
        seats=seats,
        scope="Actual held data-plane coordinates; coownership correctness still requires live transfer",
    )


def device_fds(pid, proc=Path("/proc")):
    result = []
    for fd in (Path(proc) / str(pid) / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("/dev/tenstorrent/"):
            result.append(dict(fd=int(fd.name), target=target))
    return sorted(result, key=lambda item: item["fd"])


def observe_manager(
    process,
    owner_pid,
    asic_ids,
    libraries,
    expected_env,
    *,
    identity_reader=process_identity,
    library_reader=None,
    seat_reader=inspect_seat_locks,
    fd_reader=device_fds,
    environ_reader=None,
):
    """One live process identity brackets every library/seat/FD/environment observation."""
    if library_reader is None:
        from runner_support import mapped_libraries

        library_reader = mapped_libraries
    if environ_reader is None:
        environ_reader = lambda pid: Path(f"/proc/{pid}/environ").read_bytes()
    require(process.poll() is None, "Manager exited before ownership observation")
    before = identity_reader(process.pid)
    require(before["state"] not in ("Z", "X"), "Manager is not live")
    mapped = library_reader(process.pid, libraries)
    seats = seat_reader(process.pid, asic_ids)
    owner_fds, manager_fds = fd_reader(owner_pid), fd_reader(process.pid)
    require(
        owner_fds and manager_fds and {r["target"] for r in owner_fds} == {r["target"] for r in manager_fds},
        "Manager and owner physical device FD sets differ",
    )
    raw = environ_reader(process.pid)
    actual = dict(x.decode(errors="surrogateescape").split("=", 1) for x in raw.split(b"\0") if b"=" in x)
    require(
        all(actual.get(k) == v for k, v in expected_env.items()), "Manager environment differs from reviewed startup"
    )
    after = identity_reader(process.pid)
    require(
        (before["pid"], before["start_ticks"]) == (after["pid"], after["start_ticks"])
        and after["state"] not in ("Z", "X")
        and process.poll() is None,
        "Manager exited or changed identity during complete ownership observation",
    )
    return dict(
        process=after,
        libraries=mapped,
        seats=seats,
        owner_device_fds=owner_fds,
        manager_device_fds=manager_fds,
        environment=expected_env,
        environment_sha256=hashlib.sha256(raw).hexdigest(),
    )
