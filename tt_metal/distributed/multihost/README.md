# Rank-side teardown: what a rank does when it fails or its job stops

This directory's `DistributedContext` owns how a **rank** leaves a job: leave the MPI call it is in and exit with a
truthful code, through one place (`teardown`). A rank never signals its peers and never talks to the launcher. Ending the job is
`tt-run-rank`'s, the wrapper every rank runs in under tt-run or a plain `mpirun ... tt-run-rank -- <cmd>` line
(tt-metal #65121, `tools/scaleout/README_tt_run_rank.md`).

Target stack: Open MPI 5.0.7 (ULFM build), PRRTE 3.0.8, PMIx 5.0.5.

## Who ends the job

Every stop is SIGTERM to every rank on every host through PRRTE job control (PMIx `PMIX_JOB_CTRL_SIGNAL`): each host's
daemon signals its own ranks, the job is not aborted and no SIGKILL clock starts. It is sent by:

| Who | When |
|---|---|
| tt-run | Ctrl-C or SIGTERM at tt-run, or the job's `--job-timeout` |
| `tt-run-rank`, the rank's wrapper | its rank failed (wrote `failed <code> <reason>` on the pipe, or exited non-zero without a stop having reached it) or died (SIGKILL, segfault) |

The wrappers of the failed and the stopped ranks hold their exits until every rank has exited, so the launcher never
kills a rank that is still closing. A rank still running `--stop-grace` seconds (120) after the stop is killed by its
wrapper. Under tt-run a second Ctrl-C forces the job at once.

## What a rank does (`distributed_termination.cpp`, `mpi_distributed_context.cpp`)

- When it fails (`DistributedContext::fail_rank`, called by `std::terminate`, `on_exit` with a non-zero status, a wait
  past `TT_METAL_OPERATION_TIMEOUT_SECONDS`, and from Python by a mesh close with an exception in flight or the
  excepthook) it marks itself terminating and writes one `failed <code> <reason>` line to its wrapper
  (`TT_RUN_RANK_FD`); every later wait throws. The wrapper prints the `[rank R] <reason> (on HOST, pid P); stopping
  the job (exit CODE)` line tt-run reads, and sends the SIGTERM.
- Every way out then goes through `DistributedContext::teardown(reason)`: the failure paths above, the stop (the
  SIGTERM handler sets a flag; a Python rank's handler, installed by `ttnn.distributed` next to the context, calls it
  at the next bytecode), a Ctrl-C to one rank (`KeyboardInterrupt` unwinds into the mesh close, which calls it). For
  now `teardown` prints `[rank R] teardown initiated: <reason>; exiting with code CODE` and exits without
  `MPI_Finalize`; closing the devices in an orderly way is the next step and lands there.
- A peer's death reaches a wait as ULFM's `MPI_ERR_PROC_FAILED`, usually a few ms before the wrapper's SIGTERM; the
  wait gives the signal up to 2 s to arrive, so the survivor leaves as a stopped rank, not as a second failure.
- Every MPI wait is polled (`MPI_I*` + `MPI_Test`, a 0.2 ms spin then sleeps up to 1 ms) and throws on the flag, so a
  rank inside a collective leaves it at once. `split` and `create_sub_context`, which have no non-blocking form, run on a
  helper thread and are polled the same way.
- On the launcher's SIGTERM the rank exits 143 (`leave_job`), without `MPI_Finalize`. A SIGTERM that did not come
  through the wrapper (`kill`, `scancel` of one task) is reported as this rank's failure first, so the wrapper stops the
  job for it. A rank that exits 0 finalizes normally and starts nothing.
- No PMIx in the rank: it sees a pipe and a signal.

## Tests

`tests/ttnn/distributed/teardown_sanity/test_teardown_sanity.py` (two ttnn ranks on the mock galaxy, one fault per row,
every ending), `tests/scripts/multihost/run_teardown_multihost_context_tests.sh` (the same endings with the failing
rank on a remote host, SC4), and the raw-MPI gate `test_mpi_delivers_proc_failed_ulfm`. The table of endings is in
`tests/tt_metal/distributed/multiprocess/teardown_sanity/README.md` ("How a job stops").
