#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warm test process for scripts/run_safe_pytest.sh.

One long-lived process per checkout keeps torch, ttnn and pytest imported and the Metal runtime
initialized between run_safe_pytest.sh calls, and runs each call's pytest session in-process. The
device itself is opened and closed by each session's fixtures, as in a fresh process; only the
runtime and chip setup underneath survive, so every open after the worker's first one is the cheap
kind (~0.1 s instead of ~0.7 s on wormhole). It saves the per-call interpreter start, imports,
most of the first device open, and interpreter exit.

Subcommands (run_safe_pytest.sh drives them; see its "Warm process" section):
  worker --repo DIR --lock-fd N [--since-ms MS]
                                   the long-lived process. Started by a run_safe_pytest.sh call that
                                   holds the device lock (taken at epoch MS); it inherits that lock
                                   fd and keeps the lock for as long as it lives (the runtime keeps
                                   the chip open even while no device is).
  run    --repo DIR [--wait-start SECONDS PIDFILE] [--pid-out F] -- ARGS...
                                   run one pytest session (ARGS as for `pytest`) in DIR's worker,
                                   with this process's environment, cwd, stdout and stderr; F gets
                                   the pid of the worker that ran it.
                                   Exit code: pytest's, or 75 (not run: no worker, or it refused)
                                   or 70 (the worker died during the run).
  ping   --repo DIR                exit 0 if DIR's worker is up, else 75.
  stop   --repo DIR [--pid P]      ask DIR's worker (only if its pid is P) to exit.

State carried from one call to the next, and how it is handled:
  - kernel binaries: the in-process JIT build cache keys a kernel by its source path, not its
    contents, so ttnn.device.ClearKernelCache() runs before every session; the on-disk cache then
    rebuilds exactly the kernels whose sources changed.
  - programs and device memory: none carry over. The program cache belongs to a device, and each
    session's fixtures close their devices, as in a fresh process.
  - Python modules: every module a session loaded that is not an installed library (the op under
    test, its tests, conftests, plugins) is dropped afterwards, so the next session imports the
    files as they are on disk. Modules the worker loaded at start (ttnn itself) are fingerprinted;
    if any of their files, or the built libraries, change, the worker refuses and exits, and the
    call runs cold.
  - environment: TT_*/TTNN_* variables are read once by the runtime, so a call whose values
    differ from the worker's is refused (runs cold). The JIT-server variables may differ.
  - RNG: torch, numpy and random generators are restored to their start state before each
    session, as a fresh interpreter would have them.
  - failures: after any session that does not pass (exit code != 0) the worker exits, so a hang
    or a broken device never carries over. It also exits after
    SAFE_PYTEST_WARM_MAX_RUNS sessions, SAFE_PYTEST_WARM_MAX_AGE_S seconds, or past
    SAFE_PYTEST_WARM_MAX_RSS_GB of memory.
  - other users of the device: the worker holds the device lock while it is up. It watches the
    lock file with inotify, and its mtime against the time the spawning call took the lock (for
    runners that started waiting before the worker existed); as soon as anyone else opens it
    (every runner does, right before it waits on the lock; run_safe_pytest.sh again on each
    20 s retry), the worker exits, at once when idle or right after the session in progress, and
    leaves a mark that keeps the next lock winner from starting a new worker for 120 s. It also
    exits after SAFE_PYTEST_WARM_IDLE_S seconds without a call.
  - Ethernet cores left outside base firmware at a device close are logged in the session's own
    output, where run_safe_pytest.sh's stuck-Ethernet check sees them, as in a cold run.
  - known gap: a passing session that leaves a device open (a test that opens one and never
    closes it) is not detected; the next session's open of the same device would fail.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import socket
import struct
import sys
import time

EX_NOT_RUN = 75  # the session did not start; run it cold
EX_WORKER_DIED = 70  # the worker died mid-session; the device may be dirty

# Same paths as run_safe_pytest.sh; overridable for tests that must not touch the real ones.
DIRTY_FLAG = os.environ.get("SAFE_PYTEST_DIRTY_FLAG", "/tmp/tt-device.dirty")
YIELD_MARK = os.environ.get("SAFE_PYTEST_WARM_YIELD", "/tmp/tt-device.warm-yield")
# 60 s covers ~94% of the gaps between consecutive calls in a tilize eval run (median 0.05 s).
IDLE_S = float(os.environ.get("SAFE_PYTEST_WARM_IDLE_S", "60"))
MAX_RUNS = int(os.environ.get("SAFE_PYTEST_WARM_MAX_RUNS", "200"))
# Bounds how long any runner can wait behind a warm process it cannot signal (one that queued on
# the lock before this worker existed and never reopens the lock file): under the 900 s the graded
# runner (eval_test_runner.sh) waits for the lock.
MAX_AGE_S = float(os.environ.get("SAFE_PYTEST_WARM_MAX_AGE_S", "600"))
MAX_RSS_GB = float(os.environ.get("SAFE_PYTEST_WARM_MAX_RSS_GB", "16"))

# Environment variables the worker accepts with a different value than it started with: they are
# read on every use (JIT server), or only by the Python side, which is re-imported per session.
_DYNAMIC_ENV_PREFIXES = ("TT_METAL_JIT_SERVER_", "TT_METAL_JIT_PREPROCESS", "TT_DEVICE_TIMING_LOG")


def _key(repo: str) -> str:
    return hashlib.sha1(os.path.realpath(repo).encode()).hexdigest()[:12]


def _address(repo: str) -> str:
    # Abstract socket: no file to clean up; bind fails while another worker for repo is up.
    return f"\0tt-safe-pytest-warm-{os.getuid()}-{_key(repo)}"


def _send(sock: socket.socket, obj, fds=()) -> None:
    data = json.dumps(obj).encode()
    msg = struct.pack("!I", len(data)) + data
    if fds:
        sock.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))])
    else:
        sock.sendall(msg)


def _recv(sock: socket.socket, want_fds: int = 0):
    """One length-prefixed JSON message (and up to want_fds passed fds), or (None, []) on EOF."""
    fds = array.array("i")
    head = b""
    while len(head) < 4:
        msg, anc, _, _ = sock.recvmsg(4 - len(head), socket.CMSG_LEN(want_fds * fds.itemsize) if want_fds else 0)
        if not msg:
            return None, []
        head += msg
        for level, kind, data in anc:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.frombytes(data[: len(data) - (len(data) % fds.itemsize)])
    (size,) = struct.unpack("!I", head)
    body = b""
    while len(body) < size:
        chunk = sock.recv(size - len(body))
        if not chunk:
            return None, list(fds)
        body += chunk
    return json.loads(body), list(fds)


# ---------------------------------------------------------------------------------------------
# Client side (cheap: no torch/ttnn import)


def _pid_alive(pid: int) -> bool:
    """False once pid has exited or is a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rpartition(")")[2].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _connect(repo: str, wait_s: float = 0.0, pid_file: str | None = None):
    """Connect to repo's worker, waiting up to wait_s for one that is starting. With pid_file
    (written by the spawning call just before the worker starts), stop waiting as soon as that
    worker is gone."""
    deadline = time.time() + wait_s
    pid = None
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(_address(repo))
            return sock
        except OSError:
            sock.close()
            if pid is None and pid_file:
                try:
                    with open(pid_file) as f:
                        pid = int(f.read().strip())
                except (OSError, ValueError):
                    pass
            if time.time() >= deadline or (pid is not None and not _pid_alive(pid)):
                return None
            time.sleep(0.1)


def client(
    op: str,
    repo: str,
    args: list[str],
    wait_s: float = 0.0,
    pid_file: str | None = None,
    pid_out: str | None = None,
    stop_pid: int | None = None,
) -> int:
    sock = _connect(repo, wait_s, pid_file)
    if sock is None:
        return EX_NOT_RUN
    started = False
    try:
        with sock:
            if op == "run":
                req = {"op": "run", "args": args, "env": dict(os.environ), "cwd": os.getcwd()}
                _send(sock, req, fds=(1, 2))
            else:
                _send(sock, {"op": op, "pid": stop_pid})
            reply, _ = _recv(sock)
            if reply is not None and reply.get("status") == "started":
                started = True
                if pid_out:
                    # The caller stops exactly this worker later, never a newer one.
                    with open(pid_out, "w") as f:
                        f.write(str(reply.get("pid", "")))
                reply, _ = _recv(sock)
    except OSError:
        # The worker went away (it may exit with this connection still queued).
        reply = None
    if reply is None:
        # Before the session started nothing ran; after, the worker died mid-session.
        return EX_WORKER_DIED if started else EX_NOT_RUN
    if reply.get("status") == "refused":
        print(f"SAFE_PYTEST: warm process refused the run ({reply.get('reason')}); running cold", file=sys.stderr)
        return EX_NOT_RUN
    return int(reply.get("exit", 0))


# ---------------------------------------------------------------------------------------------
# Worker side


class Worker:
    def __init__(self, repo: str, lock_fd: int, since_ms: int | None = None):
        self.repo = os.path.realpath(repo)
        self.lock_fd = lock_fd
        self.since_ms = since_ms
        self.lock_path = os.readlink(f"/proc/self/fd/{lock_fd}")
        self.started = time.time()
        self.runs = 0
        # Children (pytest subprocesses, the dispatch-timeout triage command) must not inherit the
        # lock: one that outlived the worker would keep the device locked.
        os.set_inheritable(lock_fd, False)
        # Watch before the slow imports: a runner that opens the lock file meanwhile is seen too.
        self.lock_watch = self._watch_lock_file()

        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        os.chdir(self.repo)

        import random

        import numpy
        import pytest  # noqa: F401
        import torch
        import ttnn  # noqa: F401

        # Session plugins wrap these (phase_timing); put the originals back after each session so a
        # wrapper from a purged plugin module never outlives its session.
        self.patch_points = {}
        for module_name in ("ttnn", "ttnn.device", "ttnn.distributed.distributed"):
            module = sys.modules.get(module_name)
            for name in (
                "CreateDevice",
                "CreateDevices",
                "open_device",
                "open_mesh_device",
                "close_device",
                "CloseDevice",
                "CloseDevices",
                "close_mesh_device",
            ):
                if module is not None and hasattr(module, name):
                    self.patch_points[(module_name, name)] = getattr(module, name)
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

        import sysconfig

        roots = {sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")}
        # Editable installs (ttnn) live in the checkout, not here, so they are never "library".
        self.library_roots = tuple(os.path.realpath(r) for r in roots if r)

        self.snapshot = set(sys.modules)
        self.fingerprint = self._fingerprint()
        self.env0 = dict(os.environ)
        # pytest's prepend import mode adds each test directory to sys.path; a directory left
        # there from an earlier session can shadow a same-named test module of another op.
        self.sys_path0 = list(sys.path)
        self.rng0 = (torch.get_rng_state(), numpy.random.get_state(), random.getstate())

        # Everything imported so far lives as long as the worker. Freezing it keeps every later
        # gc.collect() (pytest runs five per session in its unraisable-exception cleanup, plus
        # ours) from walking the whole torch/ttnn heap: ~0.7 s per session otherwise.
        import gc

        gc.collect()
        gc.freeze()

    # -- what may not change under the worker --------------------------------------------------

    def _under_repo(self, module) -> str | None:
        path = getattr(module, "__file__", None)
        if not path:
            paths = list(getattr(module, "__path__", []) or [])
            path = paths[0] if paths else None
        if not path:
            return None
        path = os.path.realpath(path)
        return path if path.startswith(self.repo + os.sep) else None

    def _fingerprint(self):
        files = set()
        for name in self.snapshot:
            path = self._under_repo(sys.modules.get(name))
            if path and os.path.isfile(path):
                files.add(path)
        ttnn_lib = os.path.realpath(sys.modules["ttnn"]._ttnn.__file__)
        lib_dir = os.path.dirname(ttnn_lib)
        files.update(os.path.join(lib_dir, f) for f in os.listdir(lib_dir) if f.endswith(".so"))
        files.add(ttnn_lib)
        out = {}
        for path in sorted(files):
            try:
                st = os.stat(path)
                out[path] = (st.st_mtime_ns, st.st_size)
            except OSError:
                out[path] = None
        return out

    def _refuse_reason(self, req) -> str | None:
        if self.runs >= MAX_RUNS:
            return f"served {self.runs} runs"
        if time.time() - self.started > MAX_AGE_S:
            return "worker too old"
        changed = [p for p, v in self._fingerprint().items() if self.fingerprint.get(p) != v]
        if changed:
            return f"{len(changed)} preloaded file(s) changed, e.g. {changed[0]}"
        env = req["env"]
        for k in set(env) | set(self.env0):
            if not k.startswith(("TT_", "TTNN_")) or k.startswith(_DYNAMIC_ENV_PREFIXES):
                continue
            if env.get(k) != self.env0.get(k):
                return f"environment differs: {k}"
        return None

    # -- one session ---------------------------------------------------------------------------

    def _prepare(self, req) -> None:
        import random

        import numpy
        import torch
        import ttnn

        os.environ.clear()
        os.environ.update(req["env"])
        sys.path[:] = self.sys_path0
        for entry in reversed([p for p in req["env"].get("PYTHONPATH", "").split(os.pathsep) if p]):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        os.chdir(req["cwd"])
        torch.set_rng_state(self.rng0[0])
        numpy.random.set_state(self.rng0[1])
        random.setstate(self.rng0[2])
        ttnn.device.ClearKernelCache()

    def _is_library(self, module) -> bool:
        path = getattr(module, "__file__", None)
        if not path:
            paths = list(getattr(module, "__path__", []) or [])
            path = paths[0] if paths else None
        if not path:
            return True  # builtins and frozen modules
        path = os.path.realpath(path)
        return any(path.startswith(root + os.sep) for root in self.library_roots)

    def _purge(self) -> None:
        """Drop every module the session loaded that is not an installed library (the op under
        test, tests, conftests, plugins, wherever they live), and undo the session plugins'
        wrappers on the device entry points."""
        for (module_name, name), fn in self.patch_points.items():
            setattr(sys.modules[module_name], name, fn)
        sys.path[:] = self.sys_path0
        for name in list(sys.modules):
            if name in self.snapshot:
                continue
            module = sys.modules.get(name)
            if module is None or self._is_library(module):
                continue
            del sys.modules[name]
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name) if parent_name else None
            if parent is not None and getattr(parent, child, None) is module:
                try:
                    delattr(parent, child)
                except AttributeError:
                    pass

    def _run_session(self, conn, req, fds) -> int:
        import threading

        import pytest

        t0 = time.time()
        self._prepare(req)
        t1 = time.time()
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        os.dup2(fds[0], 1)
        os.dup2(fds[1], 2)
        for fd in fds:
            os.close(fd)

        # If the caller goes away mid-session (wrapper killed), stop at once: it already marked the
        # device dirty, and dying releases the lock so the next runner can reset the device.
        done = threading.Event()

        def watch_caller():
            import select

            poller = select.poll()
            poller.register(conn.fileno(), select.POLLRDHUP | select.POLLHUP | select.POLLERR)
            while not done.is_set():
                if poller.poll(200):
                    os._exit(EX_WORKER_DIED)

        watcher = threading.Thread(target=watch_caller, daemon=True)
        watcher.start()
        sys.argv = ["pytest", *req["args"]]
        try:
            code = int(pytest.main(list(req["args"])))
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        except BaseException:  # noqa: BLE001 - report, then exit the worker below
            import traceback

            traceback.print_exc()
            code = 3
        finally:
            done.set()
            watcher.join()
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        t2 = time.time()
        self._purge()
        import gc

        gc.collect()
        print(
            f"warm worker: session {self.runs + 1} exit {code}: prepare {t1 - t0:.2f}s, "
            f"pytest {t2 - t1:.2f}s, cleanup {time.time() - t2:.2f}s",
            flush=True,
        )
        return code

    def _rss_gb(self) -> float:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30

    # -- lock and lifetime ---------------------------------------------------------------------

    def _watch_lock_file(self) -> int:
        """inotify fd that turns readable when anyone opens the lock file. Everything that wants
        the device opens it before it calls flock (run_safe_pytest.sh, tt-probe.sh,
        eval_test_runner.sh, flock(1)); calls that run in this worker never do. /proc/locks would
        name the waiters directly, but it hides processes in other PID namespaces (sandboxes)."""
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        in_nonblock, in_cloexec = 0o4000, 0o2000000
        in_open, in_delete_self, in_move_self = 0x20, 0x400, 0x800
        fd = libc.inotify_init1(in_nonblock | in_cloexec)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        if libc.inotify_add_watch(fd, self.lock_path.encode(), in_open | in_delete_self | in_move_self) < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        return fd

    def _lock_file_touched(self) -> bool:
        """Did anyone open the lock file for writing since the spawning call took the lock? That
        covers runners that started waiting before this process (and its inotify watch) existed:
        run_safe_pytest.sh, tt-probe.sh and eval_test_runner.sh all open it with `exec 9>`, and
        the truncation updates its mtime."""
        try:
            st = os.stat(self.lock_path)
        except OSError:
            return True
        if self.since_ms is not None and st.st_mtime_ns // 1_000_000 > self.since_ms:
            return True
        # Blocked flock waiters, where /proc/locks shows them (same PID namespace): catches a
        # runner that queued before this worker existed and never reopens the lock file.
        want = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        try:
            with open("/proc/locks") as f:
                return any("->" in line and want in line.split() for line in f)
        except OSError:
            return False

    def _yield_mark(self) -> None:
        """Keep the next lock winner from starting a new warm process before a waiter got its
        turn (run_safe_pytest.sh reads this mark)."""
        try:
            with open(YIELD_MARK, "a"):
                os.utime(YIELD_MARK)
        except OSError:
            pass

    def serve(self) -> None:
        import select

        lock_watch = self.lock_watch
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(_address(self.repo))
        except OSError:
            print("warm worker: another worker serves this checkout; exiting", flush=True)
            return
        listener.listen(8)
        print(f"warm worker: up for {self.repo} (pid {os.getpid()})", flush=True)
        last = time.time()
        try:
            while True:
                ready, _, _ = select.select([lock_watch, listener], [], [], 0.5)
                if lock_watch in ready or self._lock_file_touched():
                    # Checked first: a waiter that queued during the last session goes ahead of
                    # this checkout's next call.
                    print("warm worker: another process wants the device, exiting", flush=True)
                    self._yield_mark()
                    return
                if time.time() - self.started > MAX_AGE_S:
                    print(f"warm worker: up {MAX_AGE_S:.0f} s, making way for any waiter, exiting", flush=True)
                    self._yield_mark()
                    return
                if not ready:
                    if time.time() - last > IDLE_S:
                        print("warm worker: idle, exiting", flush=True)
                        return
                    continue
                conn, _ = listener.accept()
                with conn:
                    if not self._handle(conn):
                        return
                last = time.time()
        finally:
            listener.close()
            os.close(lock_watch)

    def _handle(self, conn) -> bool:
        """Serve one connection. False = the worker must exit now."""
        req, fds = _recv(conn, want_fds=2)
        if req is None:
            for fd in fds:
                os.close(fd)
            return True
        op = req.get("op")
        if op == "ping":
            _send(conn, {"status": "ok"})
            return True
        if op == "stop":
            # A stop names the worker it means: an older worker's caller must not stop this one.
            mine = req.get("pid") in (None, os.getpid())
            _send(conn, {"status": "ok" if mine else "not-me"})
            return not mine
        if op != "run" or len(fds) != 2:
            for fd in fds:
                os.close(fd)
            _send(conn, {"status": "refused", "reason": "bad request"})
            return True
        reason = self._refuse_reason(req)
        if reason:
            for fd in fds:
                os.close(fd)
            _send(conn, {"status": "refused", "reason": reason})
            print(f"warm worker: refused ({reason}), exiting", flush=True)
            return False
        _send(conn, {"status": "started", "pid": os.getpid()})
        # This process holds the lock, so it owns the dirty flag: set while a session runs, so a
        # runner that comes after a crash resets the device; cleared after a session that passed.
        # After any other result the caller takes the device over and decides (reset or clear).
        with open(DIRTY_FLAG, "a"):
            pass
        code = self._run_session(conn, req, fds)
        self.runs += 1
        if code == 0:
            try:
                os.unlink(DIRTY_FLAG)
            except FileNotFoundError:
                pass
        keep = code == 0 and self._rss_gb() < MAX_RSS_GB
        _send(conn, {"status": "done", "exit": code, "exiting": not keep})
        if not keep:
            print(f"warm worker: session exit {code}, exiting", flush=True)
        return keep

    def shutdown(self) -> None:
        import gc

        # The lock fd stays open on purpose: the runtime closes the chip during interpreter exit,
        # after this, and the kernel drops the lock only once the process is gone, so no other
        # runner can take the device while that teardown runs.
        # Let interpreter shutdown free what gc.freeze() pinned, or nanobind reports it as leaked.
        gc.unfreeze()


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--repo":
        print(__doc__, file=sys.stderr)
        return 2
    op, repo, rest = argv[0], argv[2], argv[3:]
    # Options up to `--` (run passes everything after it to pytest).
    opts = {}
    while rest and rest[0] != "--":
        name = rest.pop(0)
        nargs = 2 if name == "--wait-start" else 1
        if not name.startswith("--") or len(rest) < nargs:
            print(__doc__, file=sys.stderr)
            return 2
        opts[name] = rest[:nargs] if nargs > 1 else rest[0]
        del rest[:nargs]
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if op == "worker":
        if "--lock-fd" not in opts:
            print("worker needs --lock-fd N", file=sys.stderr)
            return 2
        since = opts.get("--since-ms")
        worker = Worker(repo, int(opts["--lock-fd"]), int(since) if since else None)
        try:
            worker.serve()
        finally:
            worker.shutdown()
        return 0
    if op == "ping":
        return client("ping", repo, [])
    if op == "stop":
        pid = opts.get("--pid")
        return client("stop", repo, [], stop_pid=int(pid) if pid else None)
    if op == "run":
        wait_s, pid_file = 0.0, None
        if "--wait-start" in opts:
            wait_s, pid_file = float(opts["--wait-start"][0]), opts["--wait-start"][1]
        return client("run", repo, rest, wait_s, pid_file, pid_out=opts.get("--pid-out"))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
