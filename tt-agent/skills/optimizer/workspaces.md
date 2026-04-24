# Parallel Workspaces for Optimizer

Spawn fresh workspaces for multi-hypothesis data flow optimization. Each
hypothesis gets a full clone + branch + build + venv. Parameter search
and single-hypothesis data flow use the current workspace — no spawn.

## When to spawn

- `dataflow-optimize` mode with `parallelism > 1`.

Parameter search never spawns (one build serves all parameter trials).

## Location

- **Parent**: parent of the current workspace (detect via
  `skills/run/workspace-detect.md`). Typical: `/localdev/$USER/workspaces/`.
- **Name**: `opt-<scope>-<YYYY-MM-DD>-<letter>/` (`a`, `b`, `c`, ...).
- **Final path**: `<parent>/opt-<scope>-<YYYY-MM-DD>-<letter>/tt-metal/`.

## Spawn procedure (per hypothesis)

Before spawning, check disk. Each workspace costs several GB (clone +
build). Surface the estimate if free space would drop under 10%.

```bash
NEW_WS=<parent>/opt-<scope>-<YYYY-MM-DD>-<letter>
mkdir -p $NEW_WS
git clone --reference <current-workspace>/tt-metal/.git \
          git@github.com:tenstorrent/tt-metal.git $NEW_WS/tt-metal
cd $NEW_WS/tt-metal
git checkout -b optimizer/<scope>-<YYYY-MM-DD>-<letter>
```

`--reference` shares git objects with the existing clone — near-instant.
Submodule init is unavoidable and slow; follow with the existing build
recipe at `knowledge/recipes/tt-metal/build.md`.

Note: without `--dissociate`, deleting the source workspace breaks the
spawn. Document the coupling in the final findings.

## First-time setup per workspace

Runs via `tt:run` so device/host routing is handled:

```bash
cd $NEW_WS/tt-metal
git submodule update --init --recursive
bash build_metal.sh -e       # first build, sets up env
bash create_venv.sh          # uv-backed, fast with warm cache
```

Activate with:

```bash
export TT_METAL_HOME=$NEW_WS/tt-metal
export PYTHONPATH=$TT_METAL_HOME
source $TT_METAL_HOME/python_env/bin/activate
```

Each subagent dispatched to this workspace uses the above as its shell
context's first step.

## ccache

Shared across all workspaces on the machine for maximum reuse. Resolve
at session start:

1. `$CCACHE_DIR` set → use it.
2. Else → `<parent-of-workspaces>/.ccache/`. Create if missing. Export.
3. Log the resolved path in `trend-<scope>.md` header.

Export `CCACHE_DIR` explicitly to every `tt:run` invocation and every
dispatched subagent — non-interactive subprocesses and MCP job contexts
don't source shell rc.

Report one of:
- *"ccache at <path> (X GB / Y GB max). Parallel builds share cache."*
- *"ccache not installed. Expect ~10 min cold builds per workspace."*

## Isolation

Shared across spawned workspaces:
- ccache (`CCACHE_DIR`), host toolchain/libs, the dev machine (device
  access serialized by tt-device-mcp).

Per-workspace (not shared):
- git clone + branch, `build/`, `python_env/`, `.tt-metal-cache/`,
  `TT_METAL_HOME`, `PYTHONPATH`.

Python-level edits to `ttnn/` isolate cleanly — each venv's editable
install sees its own `ttnn`.

## Tensors

Captured input tensors (by `extract.md`) live per-workspace at:

```
<workspace>/.tt-agent/tensors/<target>-<YYYY-MM-DD-HHMMSS>.pt
```

Each spawned workspace re-runs extraction in its own tree (fast — model
already built). Keep the same extraction across workspaces unless the
developer says otherwise.

Report tensor paths in `findings-optimizer-<scope>-<ts>.md`.

## Cleanup reporting

At session end, report:

```
Optimization session complete. Artifacts:

  Workspaces:
    /localdev/$USER/workspaces/opt-<scope>-<date>-a/   (winner, branch optimizer/<scope>-<date>-a)
    ...

  Inspect winner's diff:
    cd <winner-workspace>/tt-metal && git diff main...HEAD

  Cherry-pick into main workspace:
    cd <your-main-workspace>/tt-metal
    git fetch <winner-workspace>/tt-metal <branch>:<branch>
    git cherry-pick <sha>..<sha>

  Clean up:
    rm -rf /localdev/$USER/workspaces/opt-<scope>-<date>-{a,b,...}

  Tensors: <list of .pt paths>
```

Never auto-delete workspaces. The developer may inspect or resume.
