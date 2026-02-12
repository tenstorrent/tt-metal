# Hermetic Virtual Environment for Multi-Host

This document describes how TT-Metal multi-host jobs use a relocatable and portable Python virtual environment across Docker, CI, and shared filesystem deployments.

It is intended for:
- developers running multi-host tests locally or on physical runners,
- CI maintainers wiring workflows,
- users consuming release images that need portable Python environments.

## Why This Exists

Multi-host runs often execute from a shared workspace where:
- the venv is copied from `/opt/venv` into a workspace path such as `./python_env`,
- shells differ (`bash` vs POSIX `sh`),
- Python interpreter symlink targets may not be valid on every host.

The hermetic venv flow addresses these issues by combining:
- relocatable uv environments,
- optional POSIX-compatible activation patching,
- optional interpreter bundling for maximum portability,
- lock-safe setup in shared multi-host environments.

## Architecture Overview

1. Build or image stage creates `/opt/venv` with uv (`--relocatable`, `--link-mode copy`).
2. `scripts/patch_activate_posix.sh` (optional) patches `bin/activate` for POSIX `sh` fallback behavior.
3. Runtime multi-host setup copies `/opt/venv` to `./python_env` using `tests/scripts/multihost/setup_shared_venv.sh`.
4. Setup script optionally bundles Python into the copied venv when needed.
5. Caller activates via `eval "$(setup_shared_venv.sh --activate)"`.
6. `tt-run` inherits the active `VIRTUAL_ENV`, `PATH`, and other environment variables needed for remote dispatch.

## Key Scripts and Responsibilities

| Path | Role |
|---|---|
| `create_venv.sh` | Creates a uv-managed venv with relocatable/copy options; can bundle Python with `--bundle-python` |
| `scripts/patch_activate_posix.sh` | Optional compatibility patch for POSIX `sh`; generally unnecessary when activation happens from Bash |
| `scripts/bundle_python_into_venv.sh` | Bundles Python binaries/libs into the venv for Linux portability when symlink targets are unsafe |
| `tests/scripts/multihost/setup_shared_venv.sh` | Race-safe copy/setup of shared venv and optional activation command output |


## Local and Multi-Host Usage

Use one of these two workflows depending on where your source venv comes from.

### Workflow A: copy from Docker `/opt/venv` into workspace (`./python_env`)

This is the CI-style flow. `tests/scripts/multihost/setup_shared_venv.sh` copies from a source venv (default `/opt/venv`) into a workspace-local target (default `./python_env`), then emits activation commands.

```bash
eval "$(./tests/scripts/multihost/setup_shared_venv.sh --activate)"
```

`--activate` emits shell commands to stdout; `eval` applies them in the caller's shell so `PATH` and `VIRTUAL_ENV` are set for subsequent commands.

You can also pass explicit source and target paths:

```bash
eval "$(./tests/scripts/multihost/setup_shared_venv.sh --activate /opt/venv ./python_env)"
```

### Workflow B: developer-created distributed hermetic venv (local)

If you are setting up multi-host development locally (outside the Docker `/opt/venv` flow), create the environment with `--bundle-python` so it is self-contained across hosts:

```bash
./create_venv.sh --env-dir ./python_env --python-version 3.10 --bundle-python
source ./python_env/bin/activate
```

Why `--bundle-python` is required here:
- it removes dependency on host-specific uv Python install paths,
- it makes `./python_env` portable when shared/copied across nodes,
- it avoids broken interpreter symlinks on remote hosts.

### Run multi-host tests after activation

```bash
./tests/scripts/multihost/run_dual_t3k_tests.sh
./tests/scripts/multihost/run_dual_galaxy_tests.sh
./tests/scripts/multihost/run_quad_galaxy_tests.sh unit_tests
```

## Environment Propagation in `tt-run`

`tt-run` (implemented in `ttnn/ttnn/distributed/ttrun.py`) forwards environment variables to MPI ranks using a combination of automatic pass-through, explicit core variables, and override rules.

### `ENV_PASSTHROUGH_PREFIXES`

Environment variables in the launch shell are automatically forwarded when their names start with one of:

- `TT_`
- `ARCH_`
- `WH_`
- `TTNN_`
- `DEEPSEEK_`
- `MESH_`

This is why variables such as `ARCH_NAME`, `TTNN_CONFIG_OVERRIDES`, `DEEPSEEK_V3_HF_MODEL`, and `MESH_DEVICE` are available on remote ranks without manual `-x` wiring.

### `ENV_BLOCKLIST`

Some variables are intentionally blocked from automatic pass-through even if they match the prefixes above, because they are managed by `tt-run` or must be rank-specific:

- `TT_MESH_ID`
- `TT_MESH_HOST_RANK`
- `TT_MESH_GRAPH_DESC_PATH`
- `TT_RUN_ORIGINAL_CWD`
- `TT_METAL_MOCK_CLUSTER_DESC_PATH`
- `TT_VISIBLE_DEVICES`

### What `tt-run` always sets/passes for hermetic venv workflows

- `PATH` and `VIRTUAL_ENV` are passed from the caller shell (critical for venv-aware execution on remote hosts).
- `HOME` and `USER` are passed for OpenMPI/runtime behavior.
- Core variables like `PYTHONPATH`, `LD_LIBRARY_PATH`, `TT_METAL_HOME`, and `TT_METAL_RUNTIME_ROOT` are set with defaults/fallbacks when needed.

### Precedence model

At rank environment construction time, effective precedence is:
1. Auto pass-through via `ENV_PASSTHROUGH_PREFIXES` (minus `ENV_BLOCKLIST`)
2. `tt-run` managed defaults/derived values
3. Config `global_env`
4. Rank binding `env_overrides` (highest precedence)

## CI and Workflow Behavior

### Multi-host physical workflow

`.github/workflows/multi-host-physical.yaml` uses:
- `eval "$(./tests/scripts/multihost/setup_shared_venv.sh --activate)"`
- then the corresponding multi-host test script.

This ensures each job both sets up and activates the shared venv before invoking `tt-run` and `pytest`.

### Docker build workflow

`dockerfile/Dockerfile` stages used by `.github/workflows/build-docker-artifact.yaml`:
- set `UV_PYTHON_INSTALL_DIR=/usr/local/share/uv`,
- install Python via uv,
- create relocatable venvs with copy/managed mode,
- patch activation script for POSIX compatibility.

## Notes on `patch_activate_posix.sh`

`scripts/patch_activate_posix.sh` is a compatibility helper, not a strict requirement for all users.

- If you activate from **Bash**, patching is typically unnecessary.
- It is mainly used in **CI/Docker** paths where activation/sourcing can occur in POSIX `sh`.
- In this repo, CI applies it to keep activation robust across shell differences.

## Notes on `UV_PYTHON_INSTALL_DIR`

For system uv installs (especially in Docker/CI), use a shared stable location:

```bash
export UV_PYTHON_INSTALL_DIR=/usr/local/share/uv
uv python install 3.10
```

This improves cross-host accessibility when venvs are copied into shared workspaces.

## Troubleshooting

- **Venv not active after setup:** ensure you use `eval "$(./tests/scripts/multihost/setup_shared_venv.sh --activate)"`, not just running the script directly.
- **POSIX `sh` activation errors:** verify `scripts/patch_activate_posix.sh` has been run for the target venv.
- **Python symlink/path issues across hosts:** use `--bundle-python` in `create_venv.sh` or allow `setup_shared_venv.sh` to run bundling.
- **Concurrent setup races in shared workspace:** rely on `setup_shared_venv.sh` locking behavior rather than ad hoc `cp` flows.
