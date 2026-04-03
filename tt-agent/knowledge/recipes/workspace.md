# Workspace Setup

A workspace is an isolated development environment with its own tt-metal clone,
branch, build artifacts, and python venv. Multiple workspaces enable parallel work.

Some developers also clone vLLM, tt-inference-server, or other repos alongside
tt-metal — that's fine but not required. The workspace concept doesn't impose
which repos are present.

## Layout

```
$LOCAL_DEV/workspaces/
  main/
    tt-metal/                    # The only required repo
      python_env/                # Created by create_venv.sh
      build/                     # cmake build artifacts
    vllm/                        # Optional — if working on vLLM
    tt-inference-server/         # Optional — if working on inference
  my-feature/
    tt-metal/                    # branch: $USER/my-feature
    ...                          # Whatever else this task needs
```

## Create a workspace

```bash
WORKSPACE=$LOCAL_DEV/workspaces/<name>
mkdir -p $WORKSPACE && cd $WORKSPACE
git clone git@github.com:tenstorrent/tt-metal.git
cd tt-metal
git checkout -b $USER/<name>    # or stay on main
git submodule update --init --recursive
```

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
source $TT_METAL_HOME/python_env/bin/activate
```

## Incremental rebuild

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
