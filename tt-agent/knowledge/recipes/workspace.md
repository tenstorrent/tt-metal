# Workspace Setup

A workspace is an isolated development environment containing multiple TT repos
(tt-metal, vLLM, tt-inference-server) with a shared python venv. Workspaces
enable parallel work — each has its own branches, build artifacts, and environment.

## Layout

```
$LOCAL_DEV/workspaces/
  main/                          # Tracks upstream branches
    tt-metal/                    # branch: main
      python_env/                # Shared venv (created by create_venv.sh)
      build/                     # cmake build artifacts
    vllm/                        # branch: dev
    tt-inference-server/         # branch: dev
  my-feature/                    # Feature workspace
    tt-metal/                    # branch: $USER/my-feature (from main)
    vllm/                        # branch: $USER/my-feature (from dev)
    tt-inference-server/         # branch: $USER/my-feature (from dev)
```

## Create a workspace

```bash
WORKSPACE=$LOCAL_DEV/workspaces/<name>
mkdir -p $WORKSPACE && cd $WORKSPACE

# Clone repos
git clone git@github.com:tenstorrent/tt-metal.git
cd tt-metal && git checkout -b $USER/<name> && git submodule update --init --recursive && cd ..

git clone git@github.com:tenstorrent/vllm.git
cd vllm && git checkout -b $USER/<name> dev && cd ..

git clone git@github.com:tenstorrent/tt-inference-server.git
cd tt-inference-server && git checkout -b $USER/<name> dev && cd ..
```

For "main" workspace, skip branch creation — use upstream branches directly.

## First-time build

```bash
cd $WORKSPACE/tt-metal
bash build_metal.sh -e          # -e flag for first-time environment setup
bash create_venv.sh             # creates python_env/ with all deps
```

## Activate a workspace

```bash
export TT_METAL_HOME=$WORKSPACE/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export VLLM_TARGET_DEVICE=tt
source $TT_METAL_HOME/python_env/bin/activate
```

## Incremental rebuild (after code changes)

```bash
cd $TT_METAL_HOME && bash build_metal.sh
```

## Pull and sync

```bash
cd $TT_METAL_HOME
git pull && git submodule update --init --recursive
bash build_metal.sh
```

## Kernel iteration (no rebuild needed)

Kernels (`tt_metal/kernels/`) are compiled JIT at runtime. If you're only
editing kernel code, skip the rebuild — just re-run the test.

## Install vLLM into workspace

```bash
source $TT_METAL_HOME/python_env/bin/activate
cd $WORKSPACE/vllm
pip install -e .                # or: source tt_metal/install-vllm-tt.sh
```

## Key env vars set by activation

| Variable | Value |
|---|---|
| `TT_METAL_HOME` | `$WORKSPACE/tt-metal` |
| `PYTHONPATH` | `$TT_METAL_HOME` |
| `VLLM_TARGET_DEVICE` | `tt` |
| `VIRTUAL_ENV` | `$TT_METAL_HOME/python_env` |
| `HF_HOME` | `$LOCAL_DEV/hf_data/` |
