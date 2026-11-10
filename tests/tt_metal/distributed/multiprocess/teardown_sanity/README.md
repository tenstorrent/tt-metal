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
launch the raw C probe (`sanity_mpi_probe.c`) **through tt-run** (no ttnn import, seconds per row) and assert
tt-run ends the job correctly: the right exit code, every rank gone, nothing left in `/dev/shm`. The multi-host
group (D) runs the same probe across hosts on the production `mpirun` line. The install group (E) queries the
OpenMPI / PRRTE build directly. Files: `test_teardown_sanity_mpi.py`, `sanity_mpi_probe.c`,
`sanity_mpi_binding.yaml`, `sanity_common.py`, `run_teardown_multihost_tests.sh`.

### Group A — fault classes (does tt-run end the job right on each failure?)
| test | tt-run behavior | plan |
|---|---|---|
| `test_ttrun_clean` | a clean job: exit 0, nothing left | — |
| `test_ttrun_dead_rank` | a dead rank gets a truthful non-zero code (plain launcher) | P4 |
| `test_ttrun_abort_ends` | an aborting rank ends fast, no core-dump stall, survivors gone | P8 |
| `test_ttrun_hang_timeout` | a hang ends at PRRTE `--timeout` passed via `--mpi-args` (241) | P5 |
| `test_ttrun_no_shm_leak` | self,tcp: a failed job leaves nothing in `/dev/shm` | P11 |
| `test_ttrun_ctrl_c` | Ctrl-C at tt-run is forwarded to the launcher; nothing left | P6 |
| `test_ttrun_sigterm` | SIGTERM at tt-run is forwarded; nothing left | P7 |

### Group B — the mpirun line (group E of the plan): one-variable config departures, through tt-run
Production line: `pml ob1`, `--mca btl self,tcp`, `--mca btl_tcp_if_include <nic>`. Each row changes one variable
through `--mpi-args` and asserts a known teardown shape.
| test | departure | shape |
|---|---|---|
| `test_ttrun_bad_interface` | a bogus TCP interface | fails fast, non-zero |
| `test_ttrun_lo_interface` | the loopback interface | clean job completes |
| `test_ttrun_shm_layer_still_ends` | the shared-memory layer put back | still ends cleanly (P15) |
| `test_ttrun_pml_ucx_known_shape` | UCX instead of ob1 | refuses with a non-zero code, never a hang |
| `test_scripts_tcp_only` | static: every multihost script carries `--mca btl self,tcp` | P15 |

### Group C — tt-run unit tests
`tests/ttnn/distributed/test_ttrun.py`, class `TestJobOwnership`: the launcher logic in isolation — the clock is
off by default, a user `--timeout` in `--mpi-args` is passed through, the timeout exit code is preserved, signals
are forwarded, a leftover rank makes the job fail, and the pre-launch stale-segment sweep removes an unmapped
`/dev/shm` segment while keeping a mapped one.

### Group D — multi-host (SC4), lane L1c of the plan
The two-host classes cannot run on one host, but they are still a launcher concern (no DistributedContext).
`run_teardown_multihost_tests.sh`, wired into the SC4 (4-host) `fabric-multihost-exabox` CI, runs the C probe
across the hosts on the production `mpirun` line and asserts the teardown shape for a clean job, a rank death on
a remote host (P20), a remote hang bounded by the wall clock (P22 backstop), and a lost remote daemon — the
prted on a remote host SIGKILLed while the ranks wait, which with the plain launcher (tt-metal #60468) aborts
the job non-zero instead of ending 0 with the launch-host rank orphaned (P19). Asserting the right exit code and
that nothing is left on **any** host is what one host cannot see.

### Group E — MPI install & configuration requirements (does this build meet what tt-run needs?)
Gates that query or exercise the OpenMPI / PRRTE install directly (no fault injected), so a host with the wrong
MPI on `PATH` or a build missing a component tt-run's launch line relies on is caught before any teardown row.
Honours `TT_SANITY_MPI_PREFIX`, so a second install can be checked with the same rows.
| test | requirement | plan |
|---|---|---|
| `test_mpi_build_is_open_mpi_5_plus` | the mpirun tt-run would use is Open MPI 5.x / PRRTE | P5 P15 |
| `test_launcher_supports_timeout` | the launcher accepts PRRTE `--timeout` (the opt-in wall clock) | P5 |
| `test_required_mca_components_available` | `ompi_info` carries btl self, btl tcp, pml ob1 | P15 |
| `test_production_transport_line_runs_clean` | a job under `pml ob1` + `--mca btl self,tcp` runs | P15 |
| `test_ulfm_launcher_available` | `mpirun-ulfm` is installed and a `--with-ft ulfm` job runs | — |

---

## Part 2 — tt-metal, the DistributedContext (PR #64546, stacked on Part 1)

*Does tt-metal's own distributed stack tear down?* Added in the stacked PR, continuing the same structure.

### Group F — DistributedContext rows (`test_teardown_sanity.py`, `sanity_probe.py`)
Two tt-metal ranks on the mock dual galaxy, launched by tt-run; one fault at each stage. Grouped by the piece
that fails: the per-collective timeout (barrier / send / all_gather / recv, P16) and its false-positive guard,
the control plane's init (shape, fabric config), the solver, descriptor loading, and the user's signals.

### Group G — raw-MPI expectation gates (does our OpenMPI build meet our expectations?)
Gates only, not tests of things MPI cannot do: `test_mpi_delivers_proc_failed_ulfm` (ULFM delivers the failure)
and `test_mpi_abort_reports_failure` (MPI_Abort reports a non-zero code) — the two MPI capabilities the
DistributedContext recovery relies on (the counterpart, for the context, of Part 1's install group E).

The stacked PR adds one more group here: **Group H — multi-host DistributedContext (SC4)**, the context-level
counterpart of Group D.

---

## Running it

Locally (a built tree and the python environment):
```
pytest tests/tt_metal/distributed/multiprocess/teardown_sanity -v -p no:benchmark
```

In CI the single-host groups run as `teardown-sanity-mpi` (Part 1) and `teardown-sanity-ttmetal` (Part 2); each is
its own matrix entry, so they are reported as teardown, not folded into the fabric suite. The multi-host launcher
group (Part 1, D) runs on the SC4 lane, added in the base PR; the stacked PR adds the DistributedContext multi-host
group (Part 2, H) on the same lane. Deadlines: `TT_SANITY_DEADLINE` (a row that must end) and
`TT_SANITY_DEADLINE_HANG` (a row that hangs today). `TT_SANITY_MPI_PREFIX=/opt/openmpi-<version>` reruns the MPI
rows, install group E included, against another install.
