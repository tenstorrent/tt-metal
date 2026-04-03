# Workspace Detection

Detect the current workspace context before executing any command. This runs
first in the tt-run pipeline and produces workspace context consumed by all
subsequent steps.

## Detection Steps

### 1. Detect workspace

TT development uses workspaces — isolated directories with their own tt-metal
clone, venv, and build. See `knowledge/recipes/workspace.md` for the layout.

**Check TT_METAL_HOME:**

```bash
echo $TT_METAL_HOME
# → /localdev/$USER/workspaces/<name>/tt-metal
```

If set, we're in an activated workspace. Derive workspace root:

```bash
WORKSPACE_DIR=$(dirname $TT_METAL_HOME)
```

**If TT_METAL_HOME is not set**, detect from cwd — walk up looking for a
directory containing `build_metal.sh` and `.git/`.

If neither works, ask the user where tt-metal is.

### 2. Detect repo

From the current working directory:

```bash
git remote get-url origin
```

Extract the repo name:
- `tenstorrent/tt-metal` → `tt-metal`
- `tenstorrent/vllm` → `vllm`
- `tenstorrent/tt-inference-server` → `tt-inference-server`

### 3. Check for recipe

Look for `knowledge/recipes/<repo>/index.md` in the tt-agent directory.
If found, recipes are available. If not, proceed with explicit user commands
or tt-learn for context.

### 4. Check python environment

```bash
echo $VIRTUAL_ENV
# → $TT_METAL_HOME/python_env (if activated)
```

If no venv is active but one exists at `$TT_METAL_HOME/python_env/`, note it
needs activation. If none exists, a first-time build is needed.

### 5. Detect platform

**Check for tt-device-mcp availability:**
- If `tt_device_queue_status` MCP tool is available → device execution possible
- If not → local execution only (build, host-side testing)

### 6. Detect architecture

If on a machine with TT devices:

```bash
tt_device_exec -- "echo $ARCH_NAME"
```

If not detectable (e.g., local Mac), note as unknown.

## Output

```
Workspace: my-feature ($LOCAL_DEV/workspaces/my-feature)
Repo: tt-metal
Branch: ppetrovic/my-feature
Venv: active ($TT_METAL_HOME/python_env)
Recipe: knowledge/recipes/tt-metal/ (available)
Platform: remote (tt-device-mcp available)
Architecture: wormhole_b0
```

## Build State Detection

Before running tests, check if a build exists and is current:

- **No build dir** (`$TT_METAL_HOME/build/` missing) → first-time build needed
- **Build exists** → incremental build usually sufficient
- **Kernel-only changes** → no rebuild needed (JIT compiled at runtime)
- **After git pull** → rebuild + `git submodule update --init --recursive`
