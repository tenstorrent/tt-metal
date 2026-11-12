# Teardown sanity — structure and groupings

A dedicated test suite for the distributed-teardown plan (tt-metal #56902, tt-blaze `tools/teardown_probe`). It
is **not a fabric test**: it does not exercise the fabric or a device. It asks one question — *after a single
failure, does a distributed tt-metal / tt-run job tear down correctly?* — and answers it one fault at a time:
one fault, one assertion per row.

A row that passes today is a **gate**: the behavior already holds and the row guards it. A row that fails today
is marked `@known("P..")` with the plan's error id and fails on purpose, so the lane stays red until that fix
lands; the marker is dropped in the PR that fixes it.

This suite is built in two stacked PRs. This document is the map; each PR fills in its part.

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
| `test_ttrun_ctrl_c` | Ctrl-C (SIGINT) at tt-run is forwarded to the launcher, which ends every rank; nothing left |
| `test_ttrun_sigterm` | SIGTERM at tt-run is forwarded the same way; nothing left |

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
launches the C probe one rank per host on the production `mpirun` line and runs:
| case | the additional thing it tests |
|---|---|
| `multihost_clean` | a healthy cross-host job finishes 0 with nothing left on any host (baseline) |
| `multihost_remote_rank_death` | a rank is SIGKILLed on a **remote** host → the launcher ends the whole job non-zero, nothing left on either host |
| `multihost_remote_hang_timeout` | a rank hangs on a remote host → the PRRTE `--timeout` wall clock ends the job (exit 241), nothing left |
| `multihost_remote_daemon_death` | the launch daemon (`prted`) on a remote host is killed while the ranks wait → the job must still end non-zero with nothing left, instead of finishing 0 and orphaning the rank on the launch host |

### Group E — MPI install & configuration requirements
Gates that check the OpenMPI / PRRTE install itself (no job fault injected), so a host with the wrong MPI on
`PATH`, or a build missing something tt-run's launch line relies on, is caught before any teardown row runs.
Honours `TT_SANITY_MPI_PREFIX`, so a second install can be checked with the same rows.
| test | what it requires of the install |
|---|---|
| `test_mpi_build_is_open_mpi_5_plus` | the `mpirun` tt-run would use is Open MPI 5.x / PRRTE |
| `test_launcher_supports_timeout` | the launcher accepts the `--timeout` option (tt-run's opt-in wall clock) |
| `test_required_mca_components_available` | `ompi_info` lists btl `self`, btl `tcp`, and pml `ob1` |
| `test_production_transport_line_runs_clean` | a trivial job under `pml ob1` + `--mca btl self,tcp` actually runs |
| `test_ulfm_launcher_available` | `mpirun-ulfm` is installed and a `--with-ft ulfm` job runs |

---

## Part 2 — tt-metal, the DistributedContext (PR #64546, stacked on Part 1)

*Does tt-metal's own distributed stack tear down?* Added in the stacked PR, continuing the same structure.

### Group F — DistributedContext rows (`test_teardown_sanity.py`, `sanity_probe.py`)
Two tt-metal ranks on the mock Blackhole dual galaxy (CPU only, no device), launched by tt-run; one fault at each
stage of a real distributed job, grouped by the piece that fails:
- **the per-collective timeout** — a barrier / send / all_gather / recv that waits on a peer that is gone or wedged
  gives up at its deadline and tears the whole job down, instead of hanging forever; plus its false-positive guard
  (a slow-but-arriving peer must *not* trip the timeout);
- **the control plane's init** — ranks that disagree on the mesh shape, or on the fabric config, when the device is opened;
- **the solver and descriptor loading** — a rank killed during the solve, an unsatisfiable mesh, a descriptor that
  describes a different machine;
- **the user's signals** — Ctrl-C / SIGTERM at tt-run while the job is in a collective or in init.

### Group G — raw-MPI expectation gates
Two gates that confirm the OpenMPI build we ship provides what the DistributedContext teardown relies on — not
tests of things MPI cannot do: `test_mpi_delivers_proc_failed_ulfm` (under ULFM a dead rank is delivered to the
survivors as a process failure) and `test_mpi_abort_reports_failure` (a survivor's `MPI_Abort` makes the launcher
return a non-zero code).

### Group H — multi-host DistributedContext (runs on SC4, needs ≥ 2 hosts)
The cross-host counterpart of Group D: Group D proves the *launcher* reaps a fault on a remote host; this proves
the DistributedContext's **own** teardown does the same when the fault lands *inside* a collective on a remote host.
`run_teardown_multihost_context_tests.sh` runs the DistributedContext probe (`sanity_probe.py`) across two hosts
through tt-run, one rank per host on the mock Blackhole dual galaxy, and checks:
- `context_multihost_clean` — a clean cross-host job finishes 0 with nothing left;
- `context_multihost_remote_peer_death` — a peer dies inside a collective on the remote host → the `MPI_ERRORS_RETURN`
  errhandler turns the survivor's collective into a returned error and the context calls `MPI_Abort`; the job ends
  non-zero with nothing left on either host;
- `context_multihost_remote_hang_timeout` — a peer hangs inside a collective → the per-collective
  `TT_METAL_OPERATION_TIMEOUT_SECONDS` gives up and tears the job down, well before the launcher wall clock.
It needs the DistributedContext fix (the errhandler and the timeout), so it lives in this PR with that fix.

---

## Running it

Locally (a built tree and the python environment):
```
pytest tests/tt_metal/distributed/multiprocess/teardown_sanity -v -p no:benchmark
```

its own matrix entry, so they are reported as teardown, not folded into the fabric suite. The multi-host launcher
group (Part 1, D) runs on the SC4 lane, added in the base PR; the stacked PR adds the DistributedContext multi-host
group (Part 2, H) on the same lane. Deadlines: `TT_SANITY_DEADLINE` (a row that must end) and
`TT_SANITY_DEADLINE_HANG` (a row that hangs today). `TT_SANITY_MPI_PREFIX=/opt/openmpi-<version>` reruns the MPI
rows, install group E included, against another install.
