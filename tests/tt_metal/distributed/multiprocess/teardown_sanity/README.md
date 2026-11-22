# Teardown sanity — structure and groupings

A dedicated test suite for the distributed-teardown plan (tt-metal #56902, tt-blaze `tools/teardown_probe`). It
is **not a fabric test**: it does not exercise the fabric or a device. It asks one question — *after a single
failure, does a distributed tt-metal / tt-run job tear down correctly?* — and answers it one fault at a time:
one fault, one assertion per row.

A row that passes today is a **gate**: the behavior already holds and the row guards it. A row that fails today
is marked `@known("P..")` with the plan's error id and fails on purpose, so the lane stays red until that fix
lands; the marker is dropped in the PR that fixes it.

## How tt-run stops a job

What tt-run does for every way a job ends. A stop is SIGTERM to every rank on every host through PRRTE job control
(PMIx), sent by tt-run itself (`tt-run-rank --uri … --signal N`) or, for a rank that died or failed, by that rank's wrapper
(`tt-run-rank`, which tt-run starts every rank in); never the launcher's own kill path (SIGTERM, SIGKILL 250 ms later).
This suite checks that the stop reaches the ranks and that each rank detects the SIGTERM (`caught signal 15` from every
rank); how a tt-metal rank then closes its devices is the rank side, in [#64546](https://github.com/tenstorrent/tt-metal/pull/64546).

| ending | what tt-run does | tt-run returns | rows |
|---|---|---|---|
| Ctrl-C / SIGTERM at tt-run | SIGTERM to every rank through job control; every rank detects it | 130 / 143 | `test_ttrun_stop[ctrl_c, sigterm]`, `test_ttrun_stop_on_each_mpi_install` |
| `tt-run-rank` built without PMIx (no MPI install with it was found) | it takes PMIx from the ULFM install, else from the MPI tt-metal falls back to (every Open MPI 5 has it), else the distribution's libpmix. Without any: the wrapper and the guard still run, a stop is forced through the launcher with the device warning, and SIGUSR1/2, SIGALRM, SIGTSTP and SIGCONT still reach the ranks through prterun (SIGQUIT and SIGWINCH cannot) | 130 | `test_ttrun_stop[ctrl_c_no_helper]`, `test_ttrun.py::test_run_launcher_passes_signals_through_the_launcher_without_a_signal_tool` |
| a rank killed, aborting or exiting non-zero | its wrapper stops the job; tt-run names the rank and returns its code | 128 + signal / the code | `test_ttrun_dead_rank`, `test_ttrun_abort_ends`, `test_ttrun_dead_peer_survivors_abort` |
| the daemon on a host dies (a killed `prted`, a lost node) | each rank's wrapper there sees its daemon gone and stops its rank (SIGTERM, SIGKILL 10 s later); prterun ends the job but exits 0, so tt-run fails it, naming the host | 1 | `multihost_remote_daemon_death` (SC4), `test_ttrun.py::test_rank_wrapper_stops_its_rank_when_its_daemon_dies` |
| the job reaches tt-run's `--job-timeout N` (off by default) | SIGTERM to every rank at N s, as for Ctrl-C | 241 | `test_ttrun_hang_timeout` |
| `--mpi-args "--timeout N"` | PRRTE's own clock, passed to it unchanged: tt-run warns at the start; at N s PRRTE kills every rank at once, tt-run says the job was forced | 241 | `test_ttrun_mpi_args_timeout_is_prrtes_own` |
| Ctrl-C (or SIGTERM) a second time | forces at once: SIGTERM to the launcher, which kills every rank still running (SIGTERM, then SIGKILL 250 ms later), with a warning to run `tt-smi -r` | 128 + the first signal | `test_ttrun.py::test_run_launcher_stops_the_ranks_with_sigterm[second_signal]` |
| ranks still running when the stop grace runs out | forced the same way, with the same warning | 128 + the signal | `test_ttrun.py::test_run_launcher_stops_the_ranks_with_sigterm[grace_expired]` |
| tt-run itself killed (SIGKILL, an abort, a crash) | tt-run cannot act; its guard (`tt-run-rank --guard`, started next to the launcher in a session of its own) sees tt-run gone and stops the job as tt-run would: SIGTERM to every rank, the grace, then the force; with tt-metal ranks, their mesh close is #64546 | (tt-run is gone) | `test_ttrun_killed_stops_the_ranks`, `test_ttrun.py::test_rank_wrapper_guard_stops_the_job_when_tt_run_dies`, `test_teardown_sanity.py::test_ttrun_killed_ranks_close_their_mesh` (strict xfail until #64546) |
| SIGQUIT, SIGUSR1, SIGUSR2, SIGALRM, SIGWINCH, SIGCONT at tt-run | passed on to every rank unchanged (job control, then each rank's wrapper); the job goes on. A rank with no handler for it dies of it, and its wrapper stops the job as for any killed rank | the job's own | `test_ttrun_passes_signals_on`, `test_ttrun.py::test_run_launcher_passes_other_signals_on_to_the_ranks` |
| Ctrl-Z (SIGTSTP) at tt-run | passed on to every rank, which pauses; then tt-run pauses itself. `fg` / `bg` resumes tt-run, which passes the SIGCONT on and the ranks resume | the job's own | `test_ttrun_ctrl_z_pauses_and_resumes_the_ranks`, `test_teardown_sanity.py::test_ctrl_z_pauses_and_resumes_the_job` |
| SIGHUP at tt-run (its terminal closed, an ssh session dropped) | ignored: the job keeps running under tt-run, which logs `SIGHUP received … stop it with kill -INT <pid>`; tmux or screen keep the output too | the job's own | `test_ttrun_hangup_keeps_the_job`, `test_ttrun.py::test_run_launcher_keeps_the_job_on_a_hangup` |
| an uncaught C++ exception in a rank | `std::terminate` aborts the rank (SIGABRT); its wrapper stops the job as for any killed rank and tt-run names it. The other ranks leave without closing their mesh: on the failed collective (MPI_ERR_PROC_FAILED) or on the SIGTERM, which they do not handle yet | 134 | `test_teardown_sanity.py::test_uncaught_cxx_exception_stops_the_job` (strict xfail until #64546) |
| a Python exception or `sys.exit(1)` in a rank | nothing until the rank exits, and it does not: the exception unwinds into `MPI_Finalize`, which waits on the other ranks, still in their collectives; the job hangs until `--timeout` | 241 with `--timeout` | `test_teardown_sanity.py` (`test_raise_after_open`, `test_exit1_stops_the_job`, `test_thread_exception_stops_the_job`, `test_bad_shape`; strict xfails until #64546) |
| a process still exiting after the launcher returned | waited for up to the grace after a clean finish (not at all after a stop: it already had the grace), then SIGTERM, then SIGKILL 10 s later; a zombie counts as ended | 0 if it exits in time | `test_ttrun.py` (`TestJobOwnership`) |
| clean | normal finalize | 0 | `test_ttrun_clean` |

One grace covers every graceful shutdown: `TT_RUN_STOP_GRACE_S`, 120 s by default (tt-run forwards it to the ranks).
A stop logs a warning with the grace; ranks still running when it runs out are forced through the launcher, with a
warning to run `tt-smi -r`. Every other bound is fixed: a process tt-run sends SIGTERM (the launcher on a force, a leftover, a
launcher still up after a rank's `MPI_Abort`) is SIGKILLed 10 s later. The constants are at the top of
tt-run's `ttrun.py`.

Under Slurm, `scancel` and a time limit send SIGTERM to the job step's process group, which is tt-run's: tt-run stops
the job as for a SIGTERM at tt-run. Measured with a real `scancel` of an sbatch job and of an `srun` step on
bh-glx-120-a07u02: every rank got the SIGTERM, took its time to close and exited, and tt-run returned 143. Slurm
SIGKILLs what is left after its `KillWait` (30 s there), so under Slurm the stop grace is at most `KillWait`. tt-run
starts prterun with SIGTERM blocked (and forces it with SIGINT), so a SIGTERM delivered to every process at once (a
Slurm that signals the whole cgroup, `pkill -TERM -u`, a container stop) does not start prterun's kill path either
(`test_ttrun_sigterm_to_every_process_stops_gracefully`; without the block every rank is SIGKILLed during its exit).

A raw MPI program that exits non-zero while still inside `MPI_Finalize` cannot be seen by the wrapper
(`test_ttrun_error_exit_fails_fast_not_hang`, a strict xfail).

TODO([#64546](https://github.com/tenstorrent/tt-metal/pull/64546)): left to the rank side, each a strict xfail in `tests/ttnn/distributed/teardown_sanity/test_teardown_sanity.py`
that turns red (XPASS) when it is fixed:
- a failing rank stops the job itself (a Python exception, `sys.exit(1)`, a worker thread's exception, a TT_FATAL
  raised in Python, an uncaught C++ exception), and every other rank closes its mesh on the SIGTERM;
- the ranks of a job whose tt-run was killed close their mesh on the guard's SIGTERM;
- a pause (Ctrl-Z) longer than a per-collective operation timeout must not fail the collective it paused.

---

## Part 1 — tt-run, the launcher (PR #65121, base)

*Does tt-run address each MPI/launcher failure, and does the MPI install meet what tt-run needs?* The fault rows
launch the raw C++ probe (`sanity_mpi_probe.cpp`) **through tt-run** (no heavy Python import, seconds per row) and assert
tt-run ends the job correctly: the right exit code, every rank gone, nothing left in `/dev/shm`. The multi-host
group (D) runs the same probe across hosts on the production `mpirun` line. The install group (E) queries the
OpenMPI / PRRTE build directly. Files: `test_teardown_sanity_mpi.py`, `sanity_mpi_probe.cpp`,
`sanity_mpi_binding.yaml`, `sanity_common.py`, `run_teardown_multihost_tests.sh`.

### Group A — fault classes
How tt-run ends the job for each class of single-rank failure, each launched through tt-run.
| test | what tt-run must do |
|---|---|
| `test_ttrun_clean` | a healthy job finishes: exit 0, every rank gone, nothing in `/dev/shm` |
| `test_ttrun_dead_rank` | a rank killed by a signal → the job returns a truthful non-zero code (the plain launcher reaps the rest) |
| `test_ttrun_abort_ends` | a rank that aborts (SIGABRT) → the job ends fast, with no stall for a core dump, and every survivor is torn down |
| `test_ttrun_hang_timeout` | a rank that spins forever → the job ends at the opt-in PRRTE `--timeout` (given via `--mpi-args`) with exit 241, nothing left |
| `test_ttrun_no_shm_leak` | a failed job leaves nothing in `/dev/shm` (the `self,tcp` line maps no shared-memory segment) |
| `test_ttrun_stop[ctrl_c, sigterm]` | Ctrl-C / SIGTERM at tt-run: every rank gets SIGTERM through PRRTE job control and exits; 128 + the signal |
| `test_ttrun_stop[ctrl_c_no_helper]` | without `tt-run-rank` tt-run forces through the launcher and says the devices may need a reset |
| `test_ttrun_stop_on_each_mpi_install` | the stop on every Open MPI 5 install on the host |

### Group B — the mpirun transport/config line
tt-run launches every job with a fixed line: the point-to-point layer `pml ob1` over the TCP byte transport
(`--mca btl self,tcp`), with TCP pinned to a chosen NIC (`--mca btl_tcp_if_include <nic>`). These rows change **one
variable** of that line (through `--mpi-args`) and check that tt-run still ends the job with a predictable shape —
never a hang, never a leak — whatever the transport config does.
| test | what it changes | expected result |
|---|---|---|
| `test_ttrun_bad_interface` | a TCP interface that does not exist | the job fails fast with a non-zero code — it does not hang trying to connect |
| `test_ttrun_lo_interface` | forces the loopback interface | a clean job still completes (exit 0) |
| `test_ttrun_shm_layer_still_ends` | adds the shared-memory btl back alongside tcp | the job still ends cleanly and leaves no `/dev/shm` segment |
| `test_ttrun_pml_ucx_known_shape` | UCX instead of `ob1` | either it runs, or tt-run ends the job non-zero — never a hang |
| `test_scripts_tcp_only` | static check, no job launched | every multihost script in the repo carries `--mca btl self,tcp` |

### Group C — tt-run unit tests
`test_ttrun.py` (under `tests/.../distributed/`), class `TestJobOwnership`: the launcher logic in isolation — the clock is
off by default, a user `--timeout` in `--mpi-args` is passed through, the timeout exit code is preserved, signals
are forwarded, a leftover rank makes the job fail, and the pre-launch stale-segment sweep removes an unmapped
`/dev/shm` segment while keeping a mapped one.

### Group D — multi-host (runs on SC4, needs ≥ 2 hosts)
What a single host cannot show: a fault on a **remote** host, and whether tt-run tears the **whole** job down
across every host — the right exit code **and** nothing left alive on **any** host. `run_teardown_multihost_tests.sh`
launches the C probe through tt-run, one rank per host (the SC4 job "SC4 MPI Teardown tests"), and runs:
| case | the additional thing it tests |
|---|---|
| `multihost_clean` | a healthy cross-host job finishes 0 with nothing left on any host (baseline) |
| `multihost_remote_rank_death` | a rank is SIGKILLed on a **remote** host → its wrapper stops the job, tt-run returns 137, nothing left on any host |
| `multihost_remote_hang_timeout` | a rank hangs on a remote host → tt-run stops the job at its `--timeout` (exit 241), nothing left |
| `multihost_remote_daemon_death` | the launch daemon (`prted`) on a remote host is killed while the ranks wait → the wrapper of the rank there stops it (a rank never outlives its daemon), and tt-run fails the job although prterun exits 0 after losing a daemon; nothing left |
| `multihost_interrupt_forwarded_to_ranks` | Ctrl-C at tt-run → every rank, on every host, catches SIGTERM; nothing left |

### Group E — MPI install & configuration requirements
Gates that check the OpenMPI / PRRTE install itself (no job fault injected), so a host with the wrong MPI on
`PATH`, or a build missing something tt-run's launch line relies on, is caught before any teardown row runs.
Honours `TT_SANITY_MPI_PREFIX`, so a second install can be checked with the same rows.
| test | what it requires of the install |
|---|---|
| `test_mpi_build_is_open_mpi_5_plus` | the `mpirun` tt-run would use is Open MPI 5.x / PRRTE |
| `test_required_mca_components_available` | `ompi_info` lists btl `self`, btl `tcp`, and pml `ob1` |
| `test_production_transport_line_runs_clean` | a trivial job under `pml ob1` + `--mca btl self,tcp` actually runs |
| `test_ulfm_launcher_available` | `mpirun-ulfm` is installed and a `--with-ft ulfm` job runs |

---

## Running it

Locally (a built tree and the python environment):
```
pytest tests/tt_metal/distributed/multiprocess/teardown_sanity -v -p no:benchmark
```

In CI the single-host groups run as `teardown-sanity-mpi`, its own matrix entry, so it is reported as teardown, not
folded into the fabric suite. The multi-host group (D) runs on the SC4 lane; `TEARDOWN_REMOTE=srun` runs it inside a
Slurm allocation. Deadlines: `TT_SANITY_DEADLINE` (a row that must end) and
`TT_SANITY_DEADLINE_HANG` (a row that hangs today). `TT_SANITY_MPI_PREFIX=/opt/openmpi-<version>` reruns the MPI
rows, install group E included, against another install.
