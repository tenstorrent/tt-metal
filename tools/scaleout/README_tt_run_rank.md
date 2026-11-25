# tt-run-rank

`tt-run-rank` is the wrapper each MPI rank runs in. When one rank fails, every other rank gets one SIGTERM, closes its
devices and exits; the job returns the failing rank's own code and names it. It ships with the tt-metal build
(`build/tools/scaleout/tt-run-rank`) and needs no configuration. Your program does not change.

`tt-run` adds it to every rank for you. Use it directly only when you launch with `mpirun` yourself.

## Quick start

Put `tt-run-rank --` in front of the program, and add `--mca mpi_ft_enable 1` to the mpirun line:

```bash
# before
mpirun -np 4 python run.py
# after
mpirun --mca mpi_ft_enable 1 -np 4 tt-run-rank -- python run.py
```

Without `mpi_ft_enable`, Open MPI aborts the surviving ranks on a dead peer's socket before the SIGTERM reaches them.

## Usage

| Launch | What to change |
|---|---|
| `tt-run ...` | Nothing: every rank already runs in the wrapper. tt-run is the only launch where Ctrl-C is graceful too. |
| `mpirun ...` (scripts, plm_k8s, the MPI Operator) | `--mca mpi_ft_enable 1` on the mpirun line and `tt-run-rank --` before the program. The image that runs the ranks carries the binary. |
| Kubernetes pods, Slurm `srun --mpi=pmix` | No wrapper: the launcher already sends SIGTERM with a grace. Set `OMPI_MCA_mpi_ft_enable=1` in the rank environment and a termination grace above 0. |

More launch lines:

```bash
# a pytest rank: pytest's exit code becomes the rank's
mpirun --mca mpi_ft_enable 1 ... tt-run-rank -- pytest tests/...

# a C++ rank
mpirun --host "$HOSTS" --mca mpi_ft_enable 1 ... ./build/tools/scaleout/tt-run-rank -- ./build/tools/scaleout/run_cluster_validation ...

# a rank that may take long to close: force it after 300 s instead of 120 s
mpirun --mca mpi_ft_enable 1 ... tt-run-rank --stop-grace 300 -- python run.py

# keep the job running when a rank fails; the others are told which rank (see --on-peer-exit)
mpirun --mca mpi_ft_enable 1 --mca mpi_ft_detector true --mca mpi_ft_detector_period 0.2 \
       --mca mpi_ft_detector_timeout 1 ... tt-run-rank --on-peer-exit notify -- python run.py
```

Under tt-run the same options are `tt-run --on-peer-exit notify` and `TT_RUN_STOP_GRACE_S`.

## Behaviour

When a rank fails, its wrapper prints one line, which tt-run reads:

```
[rank 1] was killed by signal 9 (SIGKILL) (on host1, pid 41212); stopping the job (exit 137)
```

Then every other rank gets SIGTERM, closes and exits; the job returns 137.

- **A failure** is a rank that dies (a signal: 128 + the signal), exits non-zero without a stop having reached it, or
  writes `failed <code> <reason>` to `TT_RUN_RANK_FD` (reported at once, before the rank has closed).
- **A stopped rank** is one that a stop (SIGTERM, SIGINT, SIGHUP) reached first: whatever its code, it is not a failure.
  The other ranks of a failed job exit 143.
- **The exit code** of the job is the failing rank's. A clean run returns 0.
- **No rank is killed mid-close**: the wrappers of the failed and the stopped ranks hold their exits until every rank
  has ended, so the launcher's own kill finds nothing left to kill.
- **A rank that never closes** is killed by its wrapper `--stop-grace` seconds after the stop (exit 137, with a line
  saying the devices may need `tt-smi -r`).
- **A core dump** of a crashing rank is cut short after 10 s, at once when the job is already stopping.
- **A rank never outlives its daemon**: if the daemon that started it dies, the rank gets SIGTERM, then SIGKILL 10 s
  later.
- **Ctrl-C at mpirun** is still prterun's kill (SIGTERM, then SIGKILL 250 ms later). Only tt-run makes it graceful.
- **One more process per rank**: the rank is the wrapper's child. Tools that find a rank by its parent pid see the
  wrapper first.

## Options

```
tt-run-rank [--stop-grace <s>] [--on-peer-exit stop|notify] -- <program> [args...]
tt-run-rank --guard <prterun pid> [<signal tool>] [--stop-grace <s>]
tt-run-rank --uri <prterun PMIx server URI> --signal <N>
```

| Option | Default | Meaning |
|---|---|---|
| `--stop-grace <s>` | 120 | How long the rank has to exit after a stop reached it before the wrapper kills it (137). tt-run passes one 2 s below its own `TT_RUN_STOP_GRACE_S`, so the wrapper forces first. |
| `--on-peer-exit stop` | yes | When another rank fails, write `exit <rank> <code> <signal> <reason>` to `TT_RUN_RANK_EVENTS_FD`, then send this rank SIGTERM: the job stops. |
| `--on-peer-exit notify` | | The same line, then SIGUSR1 and no SIGTERM: the job goes on and the rank decides. MPI calls on the dead rank fail only with ULFM's heartbeat detector on (the `mpi_ft_detector` options above; tt-run adds them). A collective that included the dead rank returns an error, not data: recover with `MPIX_Comm_revoke` / `MPIX_Comm_shrink`. The failure line ends "the job goes on (exit N)" and the job still returns N. |
| `--guard` | | Started by tt-run next to the launcher: stops the job gracefully if tt-run itself is killed. |
| `--uri ... --signal N` | | Send signal N to every rank of a running prterun job through PRRTE job control (tt-run's stop). Exit 0 on success, 1 on a PMIx error, 2 on bad arguments, 3 without PMIx. |

| Variable | Set by | Meaning |
|---|---|---|
| `TT_RUN_RANK_FD` | the wrapper, for the rank | Write `failed <code> <reason>` here to report a failure before the rank closes. |
| `TT_RUN_RANK_EVENTS_FD` | the wrapper, for the rank | Read `exit <rank> <code> <signal> <reason>` lines here when another rank failed. |
| `TT_RUN_RANK_VERBOSE=1` | you | The wrapper says when it connects to and leaves the PMIx server. |

The wrapper talks to its PMIx server as a tool (one connection per rank). Without PMIx in the build, or when it cannot
connect, it still forwards signals and names a failed rank, but the launcher ends the job its own way.
