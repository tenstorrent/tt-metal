# Workspace Detection

Detect the current workspace context and verify the environment is ready before
executing any command. If anything is missing, tell the developer what to do —
don't try to fix it silently.

## Detection Steps

### 1. Detect workspace

TT development uses workspaces — isolated directories with their own tt-metal
clone, venv, and build. See `knowledge/recipes/workspace.md` for the layout.

**Check TT_METAL_HOME:**

```bash
echo $TT_METAL_HOME
```

If set, we're in an activated workspace. If not, detect from cwd — walk up
looking for a directory containing `build_metal.sh` and `.git/`.

If neither works, ask the user where tt-metal is.

### 2. Detect repo

```bash
git remote get-url origin
```

Extract the repo name:
- `tenstorrent/tt-metal` → `tt-metal`
- `tenstorrent/vllm` → `vllm`
- `tenstorrent/tt-inference-server` → `tt-inference-server`

### 3. Check for recipe

Look for `knowledge/recipes/<repo>/index.md` in the tt-agent directory.

## Environment Readiness Check

After detection, verify the environment before proceeding. For each issue found,
tell the developer what's missing and how to fix it. Reference the relevant
recipe. **Stop and report — don't attempt to fix setup issues.**

### Python environment

```bash
echo $VIRTUAL_ENV
```

| State | Action |
|---|---|
| Venv active at `$TT_METAL_HOME/python_env` | Ready |
| Venv exists but not active | Tell developer: `source $TT_METAL_HOME/python_env/bin/activate` |
| No venv exists | Tell developer: run first-time build (see `recipes/tt-metal/build.md`) |

### Build state

| State | Action |
|---|---|
| `$TT_METAL_HOME/build/` exists | Ready (incremental build may be needed) |
| No build dir | Tell developer: first-time build needed (see `recipes/tt-metal/build.md`) |
| Kernel-only changes | Ready — no rebuild needed (JIT compiled at runtime) |

### Device access (tt-device-mcp)

Check if `tt_device_queue_status` MCP tool is available.

| State | Action |
|---|---|
| MCP tools available | Ready for device execution |
| MCP tools not available | Tell developer: install and configure tt-device-mcp, then restart Claude Code (see `recipes/developer-setup.md`) |

### Tokens and secrets (check only when needed)

Only check these when the task actually requires them (e.g., downloading a
model from HuggingFace, accessing a private repo):

| State | Action |
|---|---|
| `HF_TOKEN` not set, model download needed | Tell developer: set HF_TOKEN (see `recipes/developer-setup.md`) |
| `GH_TOKEN` not set, private repo needed | Tell developer: set GH_TOKEN (see `recipes/developer-setup.md`) |

## Output

Produce a workspace context summary plus any issues found:

```
Workspace: my-feature
Repo: tt-metal
Branch: ppetrovic/my-feature
Venv: active
Build: ready
Platform: remote (tt-device-mcp available)
Architecture: wormhole_b0
Issues: none
```

Or with issues:

```
Workspace: my-feature
Repo: tt-metal
Venv: NOT ACTIVE — run: source $TT_METAL_HOME/python_env/bin/activate
Build: ready
Platform: LOCAL ONLY — tt-device-mcp not available
  → Install: pip install git+https://github.com/tenstorrent/tt-device-mcp.git
  → Then: tt-device-mcp daemon start && tt-device-mcp claude-add-mcp
  → Then restart Claude Code
Issues: 2 (see above)
```

If there are blocking issues (no venv, no build), stop and report before
attempting to run anything.
