# Parallel Workspaces for Optimizer

Spawn fresh workspaces for multi-hypothesis data flow optimization. Each
hypothesis gets a full clone + branch + build + venv. Parameter search and
single-hypothesis data flow use the current workspace — no spawn.

## When to spawn

- `dataflow-optimize` with `parallelism > 1`.

Never spawn for parameter search (one build serves all parameter trials).

## Location

- **Parent**: the parent of the current workspace. Resolve by detecting the
  current workspace root (see `skills/run/workspace-detect.md`) and taking
  its parent directory. Typical: `/localdev/$USER/workspaces/`.
- **Name**: `opt-<scope>-<YYYY-MM-DD>-<letter>/` where `<letter>` is `a`,
  `b`, `c`, ... one per hypothesis.
- **Final path**: `<parent>/opt-<scope>-<YYYY-MM-DD>-<letter>/tt-metal/`.

## Spawn procedure (per hypothesis)

Before spawning, check disk: each workspace costs several GB (clone + build
artifacts). Surface the estimate to the developer if it would push free
space under 10% of the volume.

```bash
NEW_WS=<parent>/opt-<scope>-<YYYY-MM-DD>-<letter>
mkdir -p $NEW_WS
git clone --reference <current-workspace>/tt-metal/.git \
          git@github.com:tenstorrent/tt-metal.git $NEW_WS/tt-metal
cd $NEW_WS/tt-metal
git checkout -b optimizer/<scope>-<YYYY-MM-DD>-<letter>
```

`--reference` shares git objects with the existing clone — the clone itself
is near-instant. Submodule init is unavoidable and slow; follow with the
existing build recipe at `knowledge/recipes/tt-metal/build.md`. It already
runs `git submodule update --init --recursive` before build.

`--dissociate` (add to clone) is optional: it copies the referenced objects
into the new clone so the two are fully independent. Without it, deleting
the source workspace would break the spawn. Prefer keeping `--reference`
without `--dissociate` and documenting the coupling in the final findings.

## First-time setup per workspace

Runs via `tt:run` so device/host routing is handled:

```bash
cd $NEW_WS/tt-metal
git submodule update --init --recursive
bash build_metal.sh -e       # first build, sets up env
bash create_venv.sh          # uv-backed, fast with warm cache
```

After this the workspace is fully self-contained. Activate with:

```bash
export TT_METAL_HOME=$NEW_WS/tt-metal
export PYTHONPATH=$TT_METAL_HOME
source $TT_METAL_HOME/python_env/bin/activate
```

Each subagent dispatched to this workspace uses the above activation as the
first step of its shell context.

## ccache

Shared across all workspaces on the machine for maximum reuse.

Resolution at session start:

1. If `$CCACHE_DIR` is set in env → use it. Report: `"ccache at $CCACHE_DIR"`.
2. Otherwise → `<parent-of-workspaces>/.ccache/`. Create if missing. Export
   explicitly.
3. Log the resolved path in `trend-<scope>.md` header.

Detection:

```bash
command -v ccache && ccache -s    # installed? current stats?
```

Report one of:
- *"ccache at <path> (X GB / Y GB max). Parallel builds will share cache —
  first workspace may cold-ish, others should be fast."*
- *"ccache not installed. Expect ~10 min cold builds per workspace."*

Export `CCACHE_DIR` explicitly to every tt:run invocation and every
dispatched subagent. Do not rely on shell-rc inheritance; non-interactive
subprocesses and MCP job contexts often do not source it.

## Isolation notes

What is shared across spawned workspaces:
- ccache (`CCACHE_DIR`)
- the host toolchain and system libs (obviously)
- the dev machine (device access serialized by tt-device-mcp)

What is NOT shared (each workspace has its own):
- git clone and checked-out branch
- `build/` directory and compiled artifacts
- `python_env/` venv
- `.tt-metal-cache/` (tt-metal runtime cache)
- `TT_METAL_HOME`, `PYTHONPATH`

This means Python-level edits (to `ttnn/` Python code) isolate cleanly
between workspaces — no `PYTHONPATH` gymnastics needed. Each workspace's
venv sees its own editable-installed `ttnn`.

## Tensors

Input tensors captured by the `extract.md` subagent live per-workspace at:

```
<workspace>/.tt-agent/tensors/<target>-<YYYY-MM-DD-HHMMSS>.pt
```

Each spawned workspace re-runs extraction in its own tree (fast — model
already built). Tensors intentionally diverge if each hypothesis wants a
different representative sample. In practice, keep the same extraction for
an apples-to-apples comparison unless the developer says otherwise.

Tensors are not committed (they are outside the repo's tracked paths).
Report their absolute paths in `findings-optimizer-<scope>-<ts>.md` so the
developer can reuse or delete them.

## Cleanup reporting

At session end (success or stop), report to the developer:

```
Optimization session complete. Artifacts:

  Workspaces:
    /localdev/$USER/workspaces/opt-<scope>-<date>-a/   (winner, branch optimizer/<scope>-<date>-a)
    /localdev/$USER/workspaces/opt-<scope>-<date>-b/   (branch optimizer/<scope>-<date>-b)
    ...

  To inspect a winner's diff:
    cd <winner-workspace>/tt-metal && git diff main...HEAD

  To cherry-pick the winning commits into your main workspace:
    cd <your-main-workspace>/tt-metal
    git fetch <winner-workspace>/tt-metal optimizer/<scope>-<date>-a:optimizer/<scope>-<date>-a
    git cherry-pick <sha>..<sha>

  To clean up:
    rm -rf /localdev/$USER/workspaces/opt-<scope>-<date>-{a,b,...}

  Tensors:
    <list of .pt paths>
```

Never auto-delete workspaces. The developer may want to inspect or resume.
