# Exabox: build on a build node, run on a reserved node

Helper scripts for working from the Slurm login node. Paths are hardcoded to /data/dstoiljkovic (copy to your own /data/<user> and adjust).

| Script | Purpose |
|---|---|
| build-metal [branch] | sbatch the tt-metal build to a cpu_only node (fetch, checkout, submodules, build_metal.sh) |
| build-wait <jobid> | block until the build job ends; exit 0 only if BUILD_DONE |
| on-node "<cmd>" | run a command as a Slurm step inside the reserved-node job (srun --jobid --overlap) |
| build_tt_metal.sbatch | the build job (gcc-12 toolchain, ccache, /tmp free-space guard) |
| create_venv.sbatch | create python_env on a build node with HOME on /data |


You are on the Slurm login node (`slurm-login-01`). It is shared and has NO Tenstorrent devices.

## One entry point: xrun

`scripts/exabox/xrun build|test|run` decides where to run on every call:

| Where you are | `xrun build [args]` | `xrun test <args>` / `xrun run "<cmd>"` |
|---|---|---|
| Exabox login node (slurm-login-0[1-3], sbatch present) | sbatch to a build node, waits for it | Slurm step on the reserved node (on-node) |
| Anywhere else (IRD container, dev box, device node) | `./build_metal.sh <args>` | `scripts/run_safe_pytest.sh <args>` / the command |

Overrides: `XRUN_MODE=exabox|local`, `XRUN_DRY_RUN=1`, `NODE_JOB=<jobid>`. Agents should call `xrun` instead of `./build_metal.sh` and `scripts/run_safe_pytest.sh`; the same call then works on both kinds of machine.

## Building (host C++ only)
- Never run `./build_metal.sh`, `cmake` or `ninja` on the login node.
- Submit the build to a build node: `/data/dstoiljkovic/bin/build-metal [branch]` (prints a job id).
- Wait for it in the background: `/data/dstoiljkovic/bin/build-wait <jobid>` (exit 0 = built, BUILD_OK).
  Logs: `/data/dstoiljkovic/logs/ttmetal-build-<jobid>.out`. A build can take up to 1 hour.
- Kernel `.cpp` changes under `kernels/` need NO build (compiled at runtime); do not submit builds for them.

## Running tests / anything that touches a device
- Run it ONLY on the reserved node, as a Slurm step: `/data/dstoiljkovic/bin/on-node "<command>"`
  e.g. `/data/dstoiljkovic/bin/on-node "scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/<op>/"`
  (runs in /data/dstoiljkovic/tt-metal on bh-lb-a10u28 through the reservation job; set NODE_JOB=<jobid> to override).
- Never run `pytest`, `run_safe_pytest.sh`, `tt-smi` or `tt-probe.sh` directly on the login node.
- Never ssh to the reserved node (every login leaves a Slurm watcher thread). Never `tt-smi -r`.
- If `on-node` says there is no running reservation, stop and tell the user; do not reserve nodes yourself.

## Limits
- 512 processes/threads per node: keep few test-running agents at once.
- Python env: /data/dstoiljkovic/tt-metal/python_env (built on a build node; HOME for builds is /data/dstoiljkovic/buildhome).
