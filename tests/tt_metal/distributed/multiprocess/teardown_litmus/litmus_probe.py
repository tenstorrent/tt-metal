# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The rank program of the teardown litmus: one tt-metal rank on a mock cluster, with one injected fault.

Walks what a real rank walks after tt-run starts it: import ttnn, initialize the distributed context,
open a MeshDevice on the mock cluster (the control plane's distributed init: discovery barrier, the
topology mapper's exchange, init_control_plane), one section of collectives on the distributed context,
close, exit. Exactly one thing goes wrong, chosen by the environment, on the victim rank or in the
collectives section. Timestamped markers on stdout let the test see how far each rank got and what each
call returned.

Env (all optional):
  PROBE_SCENARIO   clean | raise | exit_nofinalize | sigkill | abort | use_before_init
  PROBE_PHASE      before_init | after_init | in_open | after_open   (where the scenario lands; default after_open)
  PROBE_VICTIM     the rank that misbehaves (default 1)
  PROBE_DELAY_MS   delay before the fault (default 300; in_open: after open_enter)
  PROBE_COLL       barrier | barrier_skip | barrier_slow | allgather_skip | send_no_recv | recv_no_send | recv_dead_peer | barrier_dead_peer | recv_bad_rank
  PROBE_SLOW_S     seconds a slow-but-arriving peer waits before joining (barrier_slow); must be < the operation timeout
  PROBE_ON_ERROR   exit (default: a failed collective propagates and the rank exits non-zero, as an
                   application would) | continue (record and go on)
  PROBE_FABRIC_RANK   rank that sets FABRIC_2D before open (-1 none): a fabric-config conflict across ranks
  PROBE_MESH_ROWS / PROBE_MESH_COLS   the mesh shape (default 8x4); PROBE_MESH_SHAPE_VICTIM "RxC" on the victim
  PROBE_START_DELAY_S / PROBE_START_DELAY_RANK   sleep before importing ttnn on that rank (a late rank)
  PROBE_PIDDIR     directory for rank_<r>.pid files, so the test can signal one rank from outside
  PROBE_CORE_UNLIMITED=1   raise the core limit before an abort, so the kernel writes a core (P8)
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading
import time

T0 = time.time()
RANK = int(os.environ.get("TT_RUN_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "-1")))


def mark(msg: str) -> None:
    print(f"[probe r={RANK} pid={os.getpid()} t={time.time() - T0:.3f}] {msg}", flush=True)


def env(k: str, d: str) -> str:
    v = os.environ.get(k)
    return v if v else d


SCENARIO = env("PROBE_SCENARIO", "clean")
VICTIM = int(env("PROBE_VICTIM", "1"))
PHASE = env("PROBE_PHASE", "after_open")
COLL = env("PROBE_COLL", "barrier")
DELAY_MS = int(env("PROBE_DELAY_MS", "300"))
ON_ERROR = env("PROBE_ON_ERROR", "exit")
FABRIC_RANK = env("PROBE_FABRIC_RANK", "-1")

if os.environ.get("PROBE_PIDDIR"):
    os.makedirs(os.environ["PROBE_PIDDIR"], exist_ok=True)
    with open(os.path.join(os.environ["PROBE_PIDDIR"], f"rank_{RANK}.pid"), "w") as f:
        f.write(str(os.getpid()))


def do_fault() -> None:
    time.sleep(DELAY_MS / 1000.0)
    mark(f"victim acting: {SCENARIO} at {PHASE}")
    if SCENARIO == "raise":
        raise RuntimeError(f"injected failure at {PHASE}")  # the Python exception path: unwinds, atexit runs
    if SCENARIO == "abort":
        if env("PROBE_CORE_UNLIMITED", "0") == "1":
            import resource

            try:
                resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
                mark("core limit raised")
            except (ValueError, OSError) as e:
                mark(f"core limit not raised: {e}")
        os.abort()  # the TT_FATAL / std::terminate shape: SIGABRT, no atexit
    if SCENARIO == "exit_nofinalize":
        os._exit(1)
    if SCENARIO == "sigkill":
        os.kill(os.getpid(), signal.SIGKILL)


def is_victim() -> bool:
    return VICTIM == -1 or RANK == VICTIM


def maybe_fault(phase: str) -> None:
    if SCENARIO in ("raise", "abort", "exit_nofinalize", "sigkill") and is_victim() and PHASE == phase:
        do_fault()


def timed(label: str, fn) -> None:
    """Run one call and print what it returned; on error either propagate (exit non-zero) or record and go on."""
    t = time.time()
    mark(f"{label}_enter")
    try:
        r = fn()
        mark(f"{label}_rc=SUCCESS dt={time.time() - t:.3f}" + (f" value={r!r}"[:80] if r is not None else ""))
    except BaseException as e:  # noqa: BLE001 - the error shape is the measurement
        mark(f"{label}_rc={type(e).__name__}: {str(e).splitlines()[0][:120]} dt={time.time() - t:.3f}")
        if ON_ERROR == "exit":
            raise


atexit.register(lambda: mark("py_atexit"))

mark(f"start scenario={SCENARIO} victim={VICTIM} phase={PHASE} coll={COLL} on_error={ON_ERROR} ppid={os.getppid()}")
_sd = float(env("PROBE_START_DELAY_S", "0"))
if _sd > 0 and RANK == int(env("PROBE_START_DELAY_RANK", "1")):
    mark(f"start_delay {_sd}s")
    time.sleep(_sd)
import ttnn  # noqa: E402

mark("import_done")
maybe_fault("before_init")
if SCENARIO == "use_before_init" and is_victim():
    timed("barrier_before_init", ttnn.distributed_context_barrier)
ttnn.init_distributed_context()
RANK = int(ttnn.distributed_context_get_rank())
SIZE = int(ttnn.distributed_context_get_size())
mark(f"ctx_init rank={RANK} size={SIZE}")
timed("barrier_init", ttnn.distributed_context_barrier)
maybe_fault("after_init")

# ---- open: the control plane's distributed init ----
fabric_on = FABRIC_RANK != "-1" and RANK == int(FABRIC_RANK)
if fabric_on:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mark("fabric_config_set FABRIC_2D")
rows, cols = int(env("PROBE_MESH_ROWS", "8")), int(env("PROBE_MESH_COLS", "4"))
if is_victim() and env("PROBE_MESH_SHAPE_VICTIM", ""):
    rows, cols = (int(x) for x in env("PROBE_MESH_SHAPE_VICTIM", "").split("x"))
if SCENARIO in ("raise", "abort", "exit_nofinalize", "sigkill") and is_victim() and PHASE == "in_open":
    threading.Timer(0.0, do_fault).start()  # DELAY_MS after open_enter, inside the control plane's exchange
t = time.time()
mark(f"open_enter shape={rows}x{cols}")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
mark(f"open_done dt={time.time() - t:.3f}")
timed("barrier_open", ttnn.distributed_context_barrier)
maybe_fault("after_open")

# ---- the collectives section ----
if COLL == "barrier":
    timed("coll_barrier", ttnn.distributed_context_barrier)
elif COLL == "barrier_skip":
    if is_victim():
        mark("coll_skip: victim skips the barrier, sleeps 3 s")
        time.sleep(3)
    else:
        timed("coll_barrier", ttnn.distributed_context_barrier)
elif COLL == "send_no_recv":
    if is_victim():
        timed("coll_send_1MB_to_0", lambda: ttnn.distributed_context_send_bytes(b"x" * 1_000_000, 0, 7))
    else:
        timed("coll_barrier", ttnn.distributed_context_barrier)
elif COLL == "recv_dead_peer":
    if is_victim():
        mark("victim acting: sigkill before the peer's recv")
        os.kill(os.getpid(), signal.SIGKILL)
    elif RANK == 0:
        time.sleep(0.5)
        timed(f"coll_recv_16B_from_{VICTIM}", lambda: ttnn.distributed_context_recv_bytes(16, VICTIM, 7))
elif COLL == "barrier_dead_peer":
    if is_victim():
        mark("victim acting: sigkill before the peers' barrier")
        os.kill(os.getpid(), signal.SIGKILL)
    else:
        time.sleep(0.5)
        timed("coll_barrier", ttnn.distributed_context_barrier)
elif COLL == "recv_bad_rank":
    if RANK == 0:
        timed("coll_recv_8B_from_rank5_nonexistent", lambda: ttnn.distributed_context_recv_bytes(8, 5, 7))
elif COLL == "allgather_skip":
    if is_victim():
        mark("coll_skip: victim skips the all_gather, sleeps 30 s")
        time.sleep(30)
    else:
        timed("coll_allgather_int", lambda: ttnn.distributed_context_allgather_int(RANK))
elif COLL == "barrier_slow":
    if is_victim():
        slow = float(env("PROBE_SLOW_S", "4"))
        mark(f"coll_slow: victim joins the barrier {slow} s late (a legitimately slow peer, not a hang)")
        time.sleep(slow)
    timed("coll_barrier", ttnn.distributed_context_barrier)
elif COLL == "recv_no_send":
    if RANK == 0:
        timed(f"coll_recv_16B_from_{VICTIM}", lambda: ttnn.distributed_context_recv_bytes(16, VICTIM, 7))
    elif is_victim():
        mark("coll_recv_no_send: victim never sends, goes to close")

t = time.time()
mark("close_enter")
ttnn.close_mesh_device(mesh)
mark(f"close_done dt={time.time() - t:.3f}")
if fabric_on:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
timed("barrier_close", ttnn.distributed_context_barrier)
mark("main_exit")
sys.exit(0)
