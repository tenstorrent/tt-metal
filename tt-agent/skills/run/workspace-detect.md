# Workspace Detection

Detect the current workspace context before executing any command. This runs
first in the tt-run pipeline and produces workspace context consumed by all
subsequent steps.

## Detection Steps

### 1. Detect workspace

TT development uses workspaces — isolated directories containing multiple repos
with a shared python venv. See `knowledge/recipes/workspace.md` for the layout.

**Check for workspace structure:**

```bash
# If TT_METAL_HOME is set, we're in an activated workspace
echo $TT_METAL_HOME
# → /localdev/$USER/workspaces/<name>/tt-metal

# Derive workspace root
WORKSPACE_DIR=$(dirname $TT_METAL_HOME)
# → /localdev/$USER/workspaces/<name>
```

**If TT_METAL_HOME is not set**, detect from cwd:

```bash
# Are we inside a workspace?
# Pattern: $LOCAL_DEV/workspaces/<name>/tt-metal/...
# Extract workspace name from path
```

If neither works, we're not in a workspace — proceed with single-repo mode.

### 2. Detect repo

From the current working directory:

```bash
git remote get-url origin
```

Extract the repo name from the remote URL:
- `tenstorrent/tt-metal` → `tt-metal`
- `tenstorrent/vllm` → `vllm`
- `tenstorrent/tt-inference-server` → `tt-inference-server`

If in a workspace, also note which other repos are present:

```bash
ls $WORKSPACE_DIR/  # tt-metal/, vllm/, tt-inference-server/
```

### 3. Check for recipe

Look for `knowledge/recipes/<repo>/index.md` in the tt-agent directory.
If found, recipes are available for this repo. If not, tt-run still works
but relies on explicit user commands or tt-learn for context.

Also check `knowledge/recipes/workspace.md` for cross-repo workspace recipes.

### 4. Check python environment

```bash
echo $VIRTUAL_ENV
# → $TT_METAL_HOME/python_env (if workspace is activated)
```

If no venv is active but one exists at `$TT_METAL_HOME/python_env/`, note it
needs activation. If none exists, a first-time build is needed.

### 5. Detect platform

Determine whether commands run locally or on a remote machine via MCP.

**Check for tt-device-mcp availability:**
- If `tt_device_queue_status` MCP tool is available → remote device execution
- If not → local execution only (build, host-side testing)

**Local Mac (no device):** Can build, run host-side Python, but cannot run
device tests. Useful for development and code review.

**Remote TT machine (via MCP):** Full capability — build, test, run on device.

### 6. Detect architecture

If on a machine with TT devices:

```bash
tt_device_exec -- "echo $ARCH_NAME"
```

If architecture cannot be detected (e.g., local Mac), note it as unknown.

## Output

Produce a workspace context summary:

```
Workspace: my-feature ($LOCAL_DEV/workspaces/my-feature)
Repo: tt-metal (also: vllm, tt-inference-server)
Branch: ppetrovic/my-feature
Venv: active ($TT_METAL_HOME/python_env)
Recipe: knowledge/recipes/tt-metal/ (available)
Platform: remote (tt-device-mcp available)
Architecture: wormhole_b0
```

This summary is consumed by the recipe loading and command composition steps.

## Build State Detection

Before running tests, check if a build exists and is current:

- **No build dir** (`$TT_METAL_HOME/build/` missing) → first-time build needed
- **Build exists** → incremental build usually sufficient
- **Kernel-only changes** → no rebuild needed (JIT compiled at runtime)
- **After git pull** → rebuild + `git submodule update --init --recursive`
