# Workspace Setup

A workspace is an isolated development environment with its own tt-metal clone,
branch, build artifacts, and python venv. Multiple workspaces enable parallel work.

Some developers also clone vLLM, tt-inference-server, or other repos alongside
tt-metal — that's fine but not required. The workspace concept doesn't impose
which repos are present or where workspaces live on disk.

## Layout

Workspaces can live anywhere. On remote TT machines, local storage is often at
`/localdev/$USER/` (commonly exported as `$LOCAL_DEV`). Example layout:

```
<anywhere>/workspaces/
  main/
    tt-metal/                    # The only required repo
      python_env/                # Created by create_venv.sh
      build/                     # cmake build artifacts
    vllm/                        # Optional
  my-feature/
    tt-metal/                    # branch: $USER/my-feature
    ...                          # Whatever else this task needs
```

## Create a workspace

```bash
WORKSPACE=<path>/workspaces/<name>
mkdir -p $WORKSPACE && cd $WORKSPACE
git clone git@github.com:tenstorrent/tt-metal.git
cd tt-metal
git checkout -b $USER/<name>    # or stay on main
git submodule update --init --recursive
```

Then follow `recipes/tt-metal/build.md` for first-time build and venv setup.

## Activate a workspace

```bash
export TT_METAL_HOME=$WORKSPACE/tt-metal
export PYTHONPATH=$TT_METAL_HOME
source $TT_METAL_HOME/python_env/bin/activate
```
