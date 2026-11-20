# Rank-side teardown: what a rank does when its job stops

This directory's `DistributedContext` owns how a **rank** leaves a job that is stopping: leave the MPI call it is in,
close its devices, wait for the other ranks, and exit with a truthful code. Getting the stop to every rank is the
launcher's half (tt-run, `tt-run-signal`, `tt-run-rank`; tt-metal #65121).

Target stack: Open MPI 5.0.7 (ULFM build), PRRTE 3.0.8, PMIx 5.0.5.

## How a stop reaches a rank

Every stop is SIGTERM to every rank on every host through PRRTE job control (PMIx `PMIX_JOB_CTRL_SIGNAL`): each host's
daemon signals its own ranks, the job is not aborted and no SIGKILL clock starts. It is sent by:

| Who | When |
|---|---|
| tt-run (`tt-run-signal`) | Ctrl-C or SIGTERM at tt-run, or the job's `--timeout` |
| a rank (`DistributedContext::request_job_stop`) | the rank fails: a non-zero exit, an uncaught C++ or Python exception, `sys.exit(n)`, `abort(n)`, an operation timeout, a mesh close while an exception unwinds |
| a rank's watcher thread | the rank got SIGTERM from outside the launcher (`kill`, `scancel` of one task) |
| `tt-run-rank`, the dead rank's wrapper | the rank died (SIGKILL, segfault) or exited non-zero without stopping the job |

Only a force kills without a close: a second Ctrl-C, or ranks still running when tt-run's stop grace
(`TT_RUN_STOP_GRACE_S`, 120 s) runs out. tt-run then warns that the devices may need `tt-smi -r`.

## What a rank does (`distributed_termination.cpp`, `mpi_distributed_context.cpp`)

- The SIGTERM handler only sets a flag and wakes a watcher thread; it chains to Python's handler, which raises
  `SystemExit(143)` so `with`/`finally` blocks close the mesh.
- Every MPI wait is polled (`MPI_I*` + `MPI_Test`, a 0.2 ms spin then sleeps up to 1 ms) and throws on the flag, so a
  rank inside a collective leaves it at once.
- A stopping rank closes its mesh without the pre-close barrier (`mesh_device.cpp`).
- `leave_stopped_job()` waits for the other ranks at a PMIx fence (bounded by the grace + 10 s, 5 s once a peer is
  known dead), tells the wrapper, and exits with the stop's code without `MPI_Finalize`. A rank that exits 0 finalizes
  normally and starts nothing.
- A watchdog calls `MPI_Abort` if the rank is still running 10 s after its exit wait's bound.
- `TT_METAL_STOP_TIMEOUT_S` overrides the exit wait's bound.

## Tests

`tests/ttnn/distributed/teardown_sanity/test_teardown_sanity.py` (two ttnn ranks on the mock galaxy, one fault per row,
every ending), `tests/scripts/multihost/run_teardown_multihost_context_tests.sh` (the same endings with the failing
rank on a remote host, SC4), and the raw-MPI gate `test_mpi_delivers_proc_failed_ulfm`. The table of endings is in
`tests/tt_metal/distributed/multiprocess/teardown_sanity/README.md` ("How a job stops").
