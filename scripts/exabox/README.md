# Slurm cluster: build on a build node, run on a reserved node

Helper scripts for working on tt-metal from a Slurm login node that has no Tenstorrent devices. Nothing
here names a user or a machine: per-user paths come from `env.sh` (default `/data/$USER`) and every
setting can be overridden from the environment.

| Script | Purpose |
|---|---|
| `xrun build\|test\|run\|mode` | one entry point; decides on every call where the work runs |
| `build-metal [branch]` | sbatch the tt-metal build to a build node (fetch, checkout, submodules, `build_metal.sh`) |
| `build-wait <jobid>` | block until the build job ends; exit 0 only if `BUILD_DONE` |
| `on-node "<cmd>"` | run a command in the repo as a Slurm step inside your reservation job (`srun --jobid --overlap`) |
| `build_tt_metal.sbatch` | the plain-sbatch build job (gcc-12 toolchain by default, ccache, `/tmp` free-space guard), used when `tt-sbatch` is not installed or `XRUN_BUILDER=sbatch` |
| `create_venv.sbatch` | create `python_env` on a build node with `HOME` on the shared filesystem |
| `env.sh` | the settings below, plus the eval DB and run dir on the shared filesystem |

## Settings (`env.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `XRUN_DATA` | `/data/$USER` | per-user dir on the shared filesystem: `logs/`, `tmp/`, `.ccache/`, `evaldb/`, `evalruns/`, `buildhome/` |
| `XRUN_REPO` | the checkout these scripts live in (always; `XRUN_REPO_OVERRIDE=<path>` picks another) | repo to build and run in; must be on the shared filesystem. An inherited value is ignored, so an eval clone builds and tests itself |
| `NODE_JOB` | found by `on-node` | the reservation job to run steps in |
| `XRUN_RESERVATION_NAME` | unset | find the reservation by job name instead (when you hold several jobs) |
| `XRUN_BUILD_PARTITION` | `cpu_only` | partition for builds |
| `XRUN_BUILD_EXCLUDE` | empty | build nodes to avoid, comma-separated (e.g. ones with a full `/tmp`) |
| `XRUN_TOOLCHAIN` | `cmake/x86_64-linux-gcc-12-toolchain.cmake` | toolchain file for `build_tt_metal.sbatch` |
| `XRUN_PRECOMPILE_WORKERS` | `16` | precompile compile workers on the reserved node (exported as `PRECOMPILE_WORKERS`); not nproc, the node's per-user process limit (`ulimit -u` 512) counts every thread of yours there |
| `XRUN_PRECOMPILE` | `1` | `0` runs tests on the reserved node without precompile |

Without `NODE_JOB` or `XRUN_RESERVATION_NAME`, `on-node` uses your only running job outside the build
partition, and stops with a message if there are none or several.

## One entry point: xrun

`scripts/exabox/xrun build|test|run` decides where to run on every call (`xrun mode` prints the decision):

| Where you are | `xrun build [args]` | `xrun test <args>` / `xrun run "<cmd>"` |
|---|---|---|
| Slurm login node: `sbatch` present, no `/dev/tenstorrent`, not inside a Slurm job | sbatch to a build node, waits for it | Slurm step in your reservation (`on-node`) |
| Anywhere else (IRD container, dev box, device node, inside a job) | `./build_metal.sh <args>` | `scripts/run_safe_pytest.sh <args>` / the command |

Overrides: `XRUN_MODE=login|local`, `XRUN_DRY_RUN=1`, `XRUN_BYPASS=1` (skip the guards below). `build_metal.sh`,
`scripts/run_safe_pytest.sh` and the eval harness's `eval/eval_test_runner.sh` carry a guard that calls
`xrun` when `xrun mode` says `login`, so the usual commands also work from the login node.

## Building (host C++ only)
- Never run `./build_metal.sh`, `cmake` or `ninja` on the login node.
- Submit the build: `scripts/exabox/xrun build` (waits), or `scripts/exabox/build-metal` (prints a job id)
  and then `scripts/exabox/build-wait <jobid>` in the background (exit 0 = built, `BUILD_OK`).
- With `tt-sbatch` installed, the build is `tt-sbatch build` of the tree as it is (CI build image, synced back to
  `$XRUN_REPO`; build-wait then copies the fresh `_ttnn.so` into `ttnn/ttnn/`). Log: `$TT_SBATCH_JOBDIR/logs/<jobid>.log`
  (default `/data/$USER/jobs`). A build node with a full `/tmp` fails it in seconds: add it to `XRUN_BUILD_EXCLUDE`.
- Without it (or with `XRUN_BUILDER=sbatch`), `build_tt_metal.sbatch` fetches, fast-forwards and updates submodules in
  `$XRUN_REPO` and builds in place; log `$XRUN_DATA/logs/ttmetal-build-<jobid>.out`. Commit or stash first.
- When no build node is usable (all busy, or a full `/tmp`): `XRUN_BUILDER=node` builds on the reserved device node as a
  Slurm step, with the gcc toolchain (`XRUN_TOOLCHAIN`; the node's clang has no matching `ld.lld`) and `XRUN_NODE_JOBS`
  (default 24) jobs, kept low because the node's per-user process limit also covers the tests running there.
- Last resort: `XRUN_BUILDER=local` builds on the login node itself, niced and
  limited to `XRUN_LOCAL_JOBS` (default 16) parallel jobs, with an `ld.lld` shim when the host only has `ld.lld-<version>`.
- A build can take up to 1 hour.
- Kernel `.cpp` changes under `kernels/` need NO build (compiled at runtime); do not submit builds for them.

## Running tests / anything that touches a device
- Run it ONLY on the reserved node, as a Slurm step: `scripts/exabox/on-node "<command>"`, e.g.
  `scripts/exabox/on-node "scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/<op>/"`
  (runs in `$XRUN_REPO` through the reservation job).
- Never run `pytest`, `run_safe_pytest.sh`, `tt-smi` or `tt-probe.sh` directly on the login node.
- Never ssh to the reserved node (every login leaves a Slurm watcher thread). Never `tt-smi -r`.
- If `on-node` says there is no running reservation, stop and tell the user; do not reserve nodes yourself.

## Limits
- 512 processes/threads per node: keep few test-running agents at once.
- Python env: `$XRUN_REPO/python_env`, built on a build node with `create_venv.sbatch` (`HOME` is `$XRUN_DATA/buildhome`).
