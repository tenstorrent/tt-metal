#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warm fork server for run_safe_pytest.sh (opt-in: SAFE_PYTEST_WARM=1).

A cold run_safe_pytest.sh call imports pytest, torch, numpy, pandas and scipy while it holds the device lock.
This module keeps one server per (repo, interpreter, import-relevant environment) that has already imported
them, and serves each run by forking a child from it. To a test, the child is a fresh process:

  * the server never imports ttnn or opens a device, so the child initializes all of that itself;
  * the child takes the caller's environment, cwd, umask, rlimits, CPU affinity, argv and every inheritable
    fd at the same number (incl. the device-lock fd 9, so the flock is held until the child exits);
  * signal dispositions and the numpy/torch global RNGs are reset; the child exits through normal finalization.

Only third-party modules are preloaded, so repo code is always imported fresh. If a preloaded file changes on
disk, the server retires and the client starts a new one.

Failure model (the device must never see two runs at once, and a killed run must never keep running):
  * Until the server answers "started", any problem makes the client run the command cold. The server answers
    only after the child has finished its setup; if that fails, it kills and reaps the child first.
  * After "started" the client never falls back: if it loses the server it kills the child's process group.
  * The child dies with the server (PR_SET_PDEATHSIG). The server kills the child's process group, and every
    nested job (tracy's inner workload), when the client's socket closes.
  * One server per socket: an exclusive flock on <sock>.lock, held until the server retires.

Usage (from run_safe_pytest.sh):

  safe_pytest_warm.py run --pytest <pytest args...>       # like `pytest <args>`
  safe_pytest_warm.py run -m tracy <args...>              # like `python -m tracy <args>`
  safe_pytest_warm.py stop                                # retire this repo's servers
  safe_pytest_warm.py serve --socket PATH --key KEY       # internal: the server itself

The client's exit code is the child's exit code (128+N if the child died on signal N).
"""

from __future__ import annotations

import array
import ctypes
import errno
import fcntl
import hashlib
import importlib
import io
import json
import os
import resource
import select
import selectors
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import time

# Preloaded by the server. Third-party only: repo code must be imported fresh by the child. Never ttnn (loading
# _ttnn.so starts Tracy threads a forked child would not have), and never a module of a distribution that ships
# a pytest plugin, or anything that imports one (pytest warns "Module already imported so cannot be
# rewritten"); _preload() reports any.
PRELOAD = (
    # pytest itself.
    "pytest",
    "_pytest.python",
    "_pytest.fixtures",
    "_pytest.junitxml",
    "_pytest.capture",
    "_pytest.terminal",
    # What tests and conftest import. torch, numpy and loguru are also the heavy part of what ttnn imports.
    "numpy",
    "torch",
    "loguru",
    "pandas",
    "scipy",
    # The other third-party modules `import ttnn` pulls in. ttnn itself is still imported by the child.
    "click",
    "yaml",
    "networkx",
    "graphviz",
    # What the dash pytest plugin imports (it loads on every pytest start). Not `dash`: it ships the plugin.
    "IPython",
    "jupyter_client",
    "werkzeug",
    "flask",
    "requests",
    "psutil",
)

# Environment the preloaded imports or the interpreter read at startup. Part of the server key; everything else
# is applied per request in the child.
KEY_ENV = ("LD_LIBRARY_PATH", "LD_PRELOAD", "TZ", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE")
KEY_ENV_PREFIXES = (
    "PYTHON",
    "LOGURU_",
    "VIRTUAL_ENV",
    "OMP_",
    "GOMP_",
    "KMP_",
    "MKL_",
    "OPENBLAS_",
    "NPY_",
    "NUMPY_",
    "TORCH_",
    "PYTORCH_",
)

# Resource limits the child takes from the client (UMD cares about MEMLOCK and NOFILE).
RLIMITS = ("RLIMIT_NOFILE", "RLIMIT_MEMLOCK", "RLIMIT_CORE", "RLIMIT_STACK", "RLIMIT_AS", "RLIMIT_NPROC")

IDLE_TIMEOUT_S = int(os.environ.get("SAFE_PYTEST_WARM_IDLE_S", "3600"))
START_TIMEOUT_S = 120
CHILD_SETUP_TIMEOUT_S = 30
SELF_PATH = os.path.realpath(__file__)
# The only modules `run -m` accepts.
RUN_MODULES = ("tracy",)
REPO_DIR = os.path.dirname(os.path.dirname(SELF_PATH))
MAX_MSG = 16 * 1024 * 1024
MAX_FDS = 250  # SCM_RIGHTS limit is 253 per message
PR_SET_PDEATHSIG = 1


def _log(msg: str) -> None:
    print(f"SAFE_PYTEST_WARM: {msg}", file=sys.stderr, flush=True)


# --- Wire format: one length-prefixed JSON message each way, plus the client's fds as SCM_RIGHTS ---
def _send_msg(sock: socket.socket, obj: dict, fds: list[int] | None = None) -> None:
    data = json.dumps(obj).encode()
    payload = struct.pack("!I", len(data)) + data
    sent = 0
    if fds:
        sent = sock.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))])
    if sent < len(payload):
        sock.sendall(payload[sent:])


def _recv_exact(sock: socket.socket, n: int, fds_out: list[int] | None = None) -> bytes:
    buf = b""
    while len(buf) < n:
        if fds_out is not None:
            msg, ancdata, _flags, _addr = sock.recvmsg(n - len(buf), socket.CMSG_SPACE(MAX_FDS * 4))
            for level, ctype, cdata in ancdata:
                if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                    arr = array.array("i")
                    arr.frombytes(cdata[: len(cdata) - (len(cdata) % arr.itemsize)])
                    fds_out.extend(arr)
        else:
            msg = sock.recv(n - len(buf))
        if not msg:
            raise ConnectionError("peer closed")
        buf += msg
    return buf


def _recv_msg(sock: socket.socket, fds_out: list[int] | None = None) -> dict:
    (length,) = struct.unpack("!I", _recv_exact(sock, 4, fds_out))
    if length > MAX_MSG:
        raise ValueError(f"message too large: {length}")
    return json.loads(_recv_exact(sock, length, fds_out))


def _close_fds(fds) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


# --- Server identity and the private socket directory ---
def _server_key(env: dict) -> str:
    h = hashlib.sha256()
    for part in (os.path.realpath(sys.executable), sys.prefix, sys.version, REPO_DIR):
        h.update(part.encode() + b"\0")
    st = os.stat(os.path.realpath(__file__))
    h.update(f"{st.st_mtime_ns}:{st.st_size}\0".encode())
    for k in sorted(env):
        if k in KEY_ENV or k.startswith(KEY_ENV_PREFIXES):
            h.update(f"{k}={env[k]}\0".encode())
    return h.hexdigest()[:16]


def _socket_dir() -> str:
    """A directory only this user can write to; raises if an existing one is not safe to trust."""
    base = os.environ.get("SAFE_PYTEST_WARM_DIR") or f"/tmp/safe-pytest-warm-{os.getuid()}"
    try:
        os.mkdir(base, 0o700)
    except FileExistsError:
        pass
    st = os.lstat(base)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or (st.st_mode & 0o077):
        raise PermissionError(f"{base} is not a private directory owned by uid {os.getuid()}")
    return base


def _socket_path(key: str) -> str:
    repo_tag = hashlib.sha256(REPO_DIR.encode()).hexdigest()[:10]
    return os.path.join(_socket_dir(), f"{repo_tag}-{key}.sock")


# --- Server ---
def _preload() -> dict[str, tuple[int, int]]:
    """Import PRELOAD, then fingerprint every loaded module file so the server can tell when one changes."""
    t0 = time.monotonic()
    for name in PRELOAD:
        try:
            importlib.import_module(name)
        except Exception:  # optional dependency missing or broken: the child imports it (or fails) itself
            pass
    try:
        from importlib.metadata import entry_points

        plugins = {ep.value.split(":")[0] for ep in entry_points(group="pytest11")}
        # A plugin's top-level package counts too: pytest wants to rewrite its whole distribution.
        leaked = sorted({m for p in plugins for m in (p, p.split(".")[0]) if m in sys.modules})
        if leaked:
            _log(f"warning: preload imported pytest plugin module(s) {leaked}; pytest will not rewrite them")
    except Exception:
        pass
    fingerprints = {}
    for mod in list(sys.modules.values()):
        path = getattr(mod, "__file__", None)
        if not path:
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        fingerprints[path] = (st.st_mtime_ns, st.st_size)
    # The venv usually lives inside the repo (python_env/): exclude every interpreter prefix.
    repo_prefix = REPO_DIR.rstrip("/") + "/"
    env_prefixes = tuple(os.path.realpath(p).rstrip("/") + "/" for p in {sys.prefix, sys.base_prefix, sys.exec_prefix})
    in_repo = sorted(
        p
        for p in fingerprints
        if os.path.realpath(p).startswith(repo_prefix) and not os.path.realpath(p).startswith(env_prefixes)
    )
    if in_repo:
        _log(f"note: preload imported {len(in_repo)} repo module(s), e.g. {in_repo[0]}")
    _log(f"server preloaded {len(sys.modules)} modules in {time.monotonic() - t0:.1f}s")
    return fingerprints


def _fingerprints_changed(fingerprints: dict[str, tuple[int, int]]) -> str | None:
    for path, fp in fingerprints.items():
        try:
            st = os.stat(path)
        except OSError:
            return path
        if (st.st_mtime_ns, st.st_size) != fp:
            return path
    return None


class _Job:
    __slots__ = ("conn", "pid", "pidfd", "parent")

    def __init__(self, conn, pid, pidfd, parent):
        self.conn = conn
        self.pid = pid
        self.pidfd = pidfd
        self.parent = parent


def _kill_group(pid: int) -> None:
    # The child calls setsid() first thing; until it has, its pgid is still the server's, so also hit the pid.
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _ours(sock_path: str, ino: int) -> bool:
    try:
        return os.lstat(sock_path).st_ino == ino
    except FileNotFoundError:
        return False


def _await_child_ready(ack_r: int) -> str | None:
    """Wait for the forked child to finish its setup. None = it is about to run the command; else why not."""
    data = b""
    deadline = time.monotonic() + CHILD_SETUP_TIMEOUT_S
    try:
        while not data.startswith(b"ok"):
            left = deadline - time.monotonic()
            if left <= 0:
                return f"child setup did not finish within {CHILD_SETUP_TIMEOUT_S}s"
            if not select.select([ack_r], [], [], left)[0]:
                continue
            chunk = os.read(ack_r, 4096)
            if not chunk:  # EOF without "ok": it reported an error, or died
                return f"child setup failed: {data.decode(errors='replace') or 'child died'}"
            data += chunk
    except OSError as e:
        return f"child setup unknown: {e}"
    return None


def serve(sock_path: str, key: str):
    """Server loop. Returns a request dict in a forked child (the caller runs it); never returns otherwise."""
    # One server per socket path.
    lock_fd = os.open(sock_path + ".lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _log("another server already owns this socket; exiting")
        os._exit(0)

    fingerprints = _preload()
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    sys.stdout.flush()
    sys.stderr.flush()

    try:
        os.unlink(sock_path)  # stale file of a dead server
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    os.chmod(sock_path, 0o600)
    my_ino = os.lstat(sock_path).st_ino
    listener.listen(16)

    sel = selectors.DefaultSelector()
    sel.register(listener, selectors.EVENT_READ, "accept")
    listening = True
    jobs: dict[int, _Job] = {}
    last_activity = time.monotonic()
    retiring = False

    def _retire() -> None:
        nonlocal retiring, listening
        retiring = True
        if listening:
            sel.unregister(listener)
            listener.close()  # late clients get "refused" and start a new server
            listening = False
        if _ours(sock_path, my_ino):
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass
        # Let a successor take the socket path now, not when our last live run has ended.
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass

    def _drop(job: _Job) -> None:
        for obj in (job.conn, job.pidfd):
            try:
                sel.unregister(obj)
            except (KeyError, ValueError):
                pass
        try:
            job.conn.close()
        except OSError:
            pass
        try:
            os.close(job.pidfd)
        except OSError:
            pass
        jobs.pop(job.pid, None)

    def _kill_tree(pid: int) -> None:
        """Kill a job and every job started from inside it (its pid recorded as their parent)."""
        _kill_group(pid)
        for child in [j.pid for j in jobs.values() if j.parent == pid]:
            _kill_tree(child)

    def _reply_and_close(conn, fds, msg) -> None:
        _close_fds(fds)
        del fds[:]
        try:
            _send_msg(conn, msg)
        except OSError:
            pass
        conn.close()

    def _accept() -> dict | None:
        nonlocal last_activity
        conn, _ = listener.accept()
        last_activity = time.monotonic()
        fds: list[int] = []
        try:
            conn.settimeout(10)
            req = _recv_msg(conn, fds)
            conn.settimeout(None)
        except Exception as e:  # malformed or abandoned request: drop it, keep serving
            _close_fds(fds)
            conn.close()
            _log(f"dropped request: {e}")
            return None
        try:
            return _handle(conn, req, fds)
        except BaseException:
            # Never keep the caller's fds (incl. the device-lock fd 9) open in this long-lived process.
            _close_fds(fds)
            del fds[:]
            conn.close()
            raise

    def _handle(conn, req: dict, fds: list[int]) -> dict | None:
        op = req.get("op")
        if op == "ping":
            _reply_and_close(conn, fds, {"op": "pong", "key": key, "pid": os.getpid()})
            return None
        if op == "stop":
            _retire()
            _reply_and_close(conn, fds, {"op": "retired"})
            return None
        if retiring or req.get("key") != key:
            _reply_and_close(conn, fds, {"op": "stale", "reason": "server retiring" if retiring else "key mismatch"})
            return None
        changed = _fingerprints_changed(fingerprints)
        if changed:
            # Retire first: the client retries as soon as it reads the reply and must not find our socket.
            _retire()
            _reply_and_close(conn, fds, {"op": "stale", "reason": f"preloaded file changed: {changed}"})
            return None
        fd_map = req.get("fd_numbers") or []
        if len(fd_map) != len(fds) or 0 not in fd_map or 1 not in fd_map or 2 not in fd_map:
            _reply_and_close(conn, fds, {"op": "stale", "reason": "bad fd list"})
            return None

        sys.stdout.flush()
        sys.stderr.flush()
        # The child reports over this pipe once its setup is done, and only then is the client told "started".
        ack_r, ack_w = os.pipe()
        try:
            pid = os.fork()
        except OSError as e:
            _close_fds((ack_r, ack_w))
            _log(f"fork failed: {e}")
            _reply_and_close(conn, fds, {"op": "failed", "reason": f"fork failed: {e}"})
            return None
        if pid == 0:
            # Child: drop every server resource, then hand the request back to main().
            try:
                try:
                    libc = ctypes.CDLL(None, use_errno=True)
                    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
                except Exception:
                    pass
                if os.getppid() == 1:  # server died between fork and prctl
                    os._exit(1)
                sel.close()
                listener.close()
                os.close(lock_fd)
                os.close(ack_r)
                for job in jobs.values():
                    job.conn.close()
                    os.close(job.pidfd)
                req["_fds"] = list(zip(fds, fd_map))
                req["_ack"] = ack_w
                conn.close()  # the server tracks this run by pid; the child needs no channel back
                return req
            except BaseException:
                os._exit(1)  # never fall through into the server loop as a second server
        _close_fds(fds)
        del fds[:]
        os.close(ack_w)
        failure = _await_child_ready(ack_r)
        os.close(ack_r)
        pidfd = None
        if failure is None:
            try:
                pidfd = os.pidfd_open(pid)
            except OSError as e:
                failure = f"pidfd_open failed: {e}"
        if failure is not None:
            # Nothing of the caller's has run. Make sure nothing will, then let the client run cold.
            _kill_group(pid)
            os.waitpid(pid, 0)
            _log(f"run not started: {failure}")
            _reply_and_close(conn, [], {"op": "failed", "reason": failure})
            return None
        job = _Job(conn, pid, pidfd, req.get("parent_job"))
        jobs[pid] = job
        sel.register(pidfd, selectors.EVENT_READ, ("exit", pid))
        try:
            _send_msg(conn, {"op": "started", "pid": pid})
            sel.register(conn, selectors.EVENT_READ, ("conn", pid))
        except OSError:  # client vanished between request and fork: nobody owns this run
            _kill_tree(pid)
        return None

    while True:
        if not jobs and time.monotonic() - last_activity > IDLE_TIMEOUT_S:
            _retire()
        if listening and not _ours(sock_path, my_ino):  # removed by hand or replaced: stop accepting
            _retire()
        if listening and not os.path.exists(SELF_PATH):  # the checkout was deleted: nobody can ask for us again
            _retire()
        if retiring and not jobs:
            break
        for skey, _ in sel.select(timeout=5.0):
            tag = skey.data
            try:
                if tag == "accept":
                    req = _accept()
                    if req is not None:
                        return req  # forked child
                    continue
                kind, pid = tag
                job = jobs.get(pid)
                if job is None:
                    continue
                if kind == "conn":
                    # Readable means EOF (the client went away): kill the run and everything nested in it.
                    try:
                        data = job.conn.recv(64, socket.MSG_DONTWAIT)
                    except BlockingIOError:
                        continue
                    except OSError:
                        data = b""
                    if not data:
                        _kill_tree(pid)
                        sel.unregister(job.conn)
                else:
                    _, status = os.waitpid(pid, 0)
                    code = 128 + os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status)
                    last_activity = time.monotonic()
                    for nested in [j.pid for j in jobs.values() if j.parent == pid]:
                        _kill_tree(nested)  # a nested job must not outlive the job that started it
                    try:
                        _send_msg(job.conn, {"op": "exit", "code": code})
                    except OSError:
                        pass
                    _drop(job)
            except Exception as e:  # never let one event take the server (and every live run) down
                _log(f"error handling {tag}: {type(e).__name__}: {e}")
                if isinstance(tag, tuple) and tag[1] in jobs:
                    _kill_tree(tag[1])
    _retire()
    os._exit(0)


def run_child(req: dict) -> int:
    """In the forked child: become the requested process, run it, and return its exit code."""
    ack = [req.pop("_ack")]
    try:
        _child_setup(req, ack)
        os.write(ack[0], b"ok")
        os.close(ack[0])
    except BaseException as e:
        # Nothing has run yet: tell the server why, so the client runs the command cold.
        try:
            os.write(ack[0], f"{type(e).__name__}: {e}".encode()[:2000])
        except OSError:
            pass
        os._exit(1)

    argv = req["argv"]
    if req["mode"] == "pytest":
        import pytest

        exe = req.get("pytest_exe") or "pytest"
        sys.argv = [exe, *argv]
        sys.path.insert(0, os.path.dirname(os.path.realpath(exe)))
        return pytest.console_main()

    import runpy

    module = req["module"]
    if module not in RUN_MODULES:
        raise ValueError(f"module {module!r} is not one of {RUN_MODULES}")
    sys.argv = [module, *argv]
    sys.path.insert(0, os.getcwd())
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


def _child_setup(req: dict, ack: list[int]) -> None:
    """Make the forked child the process the caller asked for. `ack[0]` is the readiness pipe (it may move)."""
    os.setsid()  # own process group, so the server can kill the whole run (incl. tracy/triage subprocesses)

    # File descriptors: exactly the client's inheritable set, at the same numbers (fd 9 = the device lock).
    received = req["_fds"]
    wanted = {dst for _, dst in received}
    # Park received fds above every target first, so dup2 never clobbers one not yet placed. F_DUPFD needs its
    # floor below RLIMIT_NOFILE, so lift the soft limit for the shuffle (the caller's limits are applied below).
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ValueError, OSError):
        pass
    floor = max(wanted) + 1
    moved = fcntl.fcntl(ack[0], fcntl.F_DUPFD_CLOEXEC, floor)  # the ack pipe may sit on a target number too
    os.close(ack[0])
    ack[0] = moved
    high = []
    for src, dst in received:
        h = fcntl.fcntl(src, fcntl.F_DUPFD_CLOEXEC, floor)
        os.close(src)
        high.append((h, dst))
    keep = set()
    for h, dst in high:
        os.dup2(h, dst, inheritable=True)
        os.close(h)
        keep.add(dst)
    for fd_name in os.listdir("/proc/self/fd"):
        fd = int(fd_name)
        if fd not in keep and fd > 2:
            try:
                if not (fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC):
                    os.close(fd)  # anything else inheritable came from the server, not the caller
            except OSError:
                pass
    del wanted

    env = req["env"]
    os.environ.clear()
    os.environ.update(env)
    os.chdir(req["cwd"])
    os.umask(req["umask"])
    for name, (soft, hard) in (req.get("rlimits") or {}).items():
        try:
            resource.setrlimit(getattr(resource, name), (soft, hard))
        except (ValueError, OSError, AttributeError):
            pass
    if req.get("affinity"):
        try:
            os.sched_setaffinity(0, req["affinity"])
        except OSError:
            pass

    # Signal dispositions of a fresh CPython: SIGPIPE/SIGXFSZ ignored, SIGINT -> KeyboardInterrupt, rest default.
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGCHLD, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(sig, signal.SIG_DFL)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    # A signal the caller ignores stays ignored across exec in a cold run (nohup's SIGHUP, for one).
    for num in req.get("ignored_signals") or []:
        try:
            signal.signal(num, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

    # Unseeded draws must differ between runs, as across cold processes (`random` reseeds itself after fork).
    np = sys.modules.get("numpy")
    if np is not None:
        np.random.seed()
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.seed()
    # tempfile caches its directory on first use; the caller's TMPDIR may differ from the server's.
    if "tempfile" in sys.modules:
        sys.modules["tempfile"].tempdir = None

    # stdio: rebuild the text wrappers with CPython's startup rules. PYTHONUNBUFFERED puts the text layer on the
    # raw fd, as `python -u` does (write_through over a BufferedWriter would lose buffered bytes on a crash).
    unbuffered = bool(env.get("PYTHONUNBUFFERED"))
    encoding = env.get("PYTHONIOENCODING", "").split(":")[0] or "utf-8"
    sys.stdin = sys.__stdin__ = open(0, "r", closefd=False, encoding=encoding)
    for fd, name in ((1, "stdout"), (2, "stderr")):
        errors = "backslashreplace" if name == "stderr" else "strict"
        line_buffering = name == "stderr" or os.isatty(fd)
        if unbuffered:
            raw = open(fd, "wb", buffering=0, closefd=False)
            stream = io.TextIOWrapper(
                raw, encoding=encoding, errors=errors, line_buffering=line_buffering, write_through=True
            )
        else:
            stream = open(fd, "w", closefd=False, encoding=encoding, errors=errors)
            stream.reconfigure(line_buffering=line_buffering)
        setattr(sys, name, stream)
        setattr(sys, f"__{name}__", stream)

    # Nested runs (tracy's inner workload) report this job as their parent so the server can kill them with it.
    os.environ["SAFE_PYTEST_WARM_JOB"] = str(os.getpid())

    # sys.path[0] is what the interpreter would have computed for the emulated command line.
    server_dir = os.path.dirname(os.path.realpath(__file__))
    if sys.path and os.path.realpath(sys.path[0] or ".") == server_dir:
        sys.path.pop(0)


# --- Client ---
def _connect(sock_path: str) -> socket.socket | None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
        return s
    except OSError:
        s.close()
        return None


def _ping(sock_path: str) -> bool:
    s = _connect(sock_path)
    if s is None:
        return False
    try:
        s.settimeout(10)
        _send_msg(s, {"op": "ping"})
        return _recv_msg(s).get("op") == "pong"
    except (OSError, ValueError, ConnectionError):
        return False
    finally:
        s.close()


def _start_server(sock_path: str, key: str) -> None:
    log_path = sock_path[: -len(".sock")] + ".log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, os.path.realpath(__file__), "serve", "--socket", sock_path, "--key", key],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd=REPO_DIR,
            start_new_session=True,
            close_fds=True,
        )
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if _ping(sock_path):
            return
        rc = proc.poll()
        # rc 0 = it lost the startup race to a concurrent server for the same socket: wait for that one.
        if rc is not None and rc != 0:
            raise RuntimeError(f"warm server exited with {rc} (log: {log_path})")
        time.sleep(0.05)
    raise TimeoutError(f"warm server did not come up within {START_TIMEOUT_S}s (log: {log_path})")


def _inheritable_fds() -> list[int]:
    fds = []
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        try:
            if not (fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC):
                fds.append(fd)
        except OSError:
            pass  # the fd listdir itself used, already closed
    return sorted(fds)


def _ignored_signals() -> list[int]:
    """Signals this process ignores: inherited that way (nohup, a parent's trap ''), or by CPython itself."""
    out = []
    for sig in signal.valid_signals():
        try:
            if signal.getsignal(sig) is signal.SIG_IGN:
                out.append(int(sig))
        except (ValueError, OSError):
            pass
    return out


def _cold_exec(mode: str, module: str | None, argv: list[str]) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if mode == "pytest":
        exe = shutil.which("pytest") or "pytest"
        os.execvp(exe, [exe, *argv])
    os.execv(sys.executable, [sys.executable, "-m", module, *argv])


def client_run(mode: str, module: str | None, argv: list[str]) -> int:
    env = dict(os.environ)
    env.pop("SAFE_PYTEST_WARM_JOB", None)
    parent_job = os.environ.get("SAFE_PYTEST_WARM_JOB")
    key = _server_key(env)
    cur_umask = os.umask(0o077)
    os.umask(cur_umask)
    rlimits = {}
    for name in RLIMITS:
        if hasattr(resource, name):
            rlimits[name] = list(resource.getrlimit(getattr(resource, name)))
    fds = _inheritable_fds()
    ignored = _ignored_signals()
    req = {
        "op": "run",
        "key": key,
        "mode": mode,
        "module": module,
        "argv": argv,
        "env": env,
        "cwd": os.getcwd(),
        "umask": cur_umask,
        "rlimits": rlimits,
        "affinity": sorted(os.sched_getaffinity(0)),
        "ignored_signals": ignored,
        "pytest_exe": shutil.which("pytest") if mode == "pytest" else None,
        "fd_numbers": fds,
        "parent_job": int(parent_job) if parent_job and parent_job.isdigit() else None,
    }

    reason = None
    try:
        if len(fds) > MAX_FDS or not {0, 1, 2} <= set(fds):
            raise RuntimeError(f"cannot forward fds {fds[:8]}...")
        sock_path = _socket_path(key)
        for _attempt in range(3):
            s = _connect(sock_path)
            started_server = False
            if s is None:
                t0 = time.monotonic()
                _start_server(sock_path, key)
                started_server = True
                s = _connect(sock_path)
                if s is None:
                    raise ConnectionError("server unreachable after start")
                _log(f"started warm server in {time.monotonic() - t0:.1f}s")
            sys.stdout.flush()
            sys.stderr.flush()
            _send_msg(s, req, fds=fds)
            reply = _recv_msg(s)
            if reply.get("op") == "stale":
                reason = reply.get("reason")
                s.close()
                continue  # the server unlinked itself if it is retiring; the next attempt starts a new one
            if reply.get("op") == "failed":  # the server could not start the run, and made sure nothing runs
                raise RuntimeError(reply.get("reason") or "server could not start the run")
            if reply.get("op") != "started":
                raise ConnectionError(f"unexpected reply {reply}")
            break
        else:
            raise RuntimeError(f"no usable server ({reason})")
    except Exception as e:  # nothing has run yet: run cold instead of failing the caller's run
        _log(f"warm path unavailable ({type(e).__name__}: {e}); running cold")
        _cold_exec(mode, module, argv)
        return 127  # not reached

    # From here on the child exists. Never fall back to a cold run: that would put two runs on the device.
    child = reply["pid"]
    if not started_server:
        _log("reusing warm server")

    def _kill_child(signum):
        for target in (lambda: os.killpg(child, signum), lambda: os.kill(child, signum)):
            try:
                target()
                return
            except (ProcessLookupError, PermissionError):
                continue

    def _forward(signum, _frame):
        _kill_child(signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        if int(sig) not in ignored:  # an ignored signal (nohup's SIGHUP) stays ignored, here and in the child
            signal.signal(sig, _forward)
    try:
        reply = _recv_msg(s)
        return int(reply.get("code", 1))
    except Exception as e:
        _log(f"lost the warm server mid-run ({type(e).__name__}: {e}); killing the run")
        _kill_child(signal.SIGKILL)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        return 1


def client_stop() -> int:
    try:
        base = _socket_dir()
    except OSError:
        return 0
    repo_tag = hashlib.sha256(REPO_DIR.encode()).hexdigest()[:10]
    for name in os.listdir(base):
        if name.startswith(repo_tag) and name.endswith(".sock"):
            s = _connect(os.path.join(base, name))
            if s is None:
                continue
            try:
                s.settimeout(10)
                _send_msg(s, {"op": "stop"})
                _recv_msg(s)
            except Exception:
                pass
            s.close()
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, rest = args[0], args[1:]
    if cmd == "serve":
        sock_path = rest[rest.index("--socket") + 1]
        key = rest[rest.index("--key") + 1]
        req = serve(sock_path, key)
        # Only a forked child gets here. Returning from main() makes the interpreter exit normally.
        return run_child(req)
    if cmd == "stop":
        return client_stop()
    if cmd == "run":
        if rest and rest[0] == "--pytest":
            return client_run("pytest", None, rest[1:])
        if len(rest) >= 2 and rest[0] == "-m" and rest[1] in RUN_MODULES:
            return client_run("module", rest[1], rest[2:])
        modules = "|".join(RUN_MODULES)
        print(f"usage: safe_pytest_warm.py run (--pytest ARGS... | -m {modules} ARGS...)", file=sys.stderr)
        return 2
    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
