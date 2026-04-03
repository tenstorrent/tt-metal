# Workspace Detection

Detect the current workspace context before executing any command. This runs
first in the tt-run pipeline and produces workspace context consumed by all
subsequent steps.

## Detection Steps

### 1. Detect repo

```bash
git remote get-url origin
```

Extract the repo name from the remote URL:
- `tenstorrent/tt-metal` → `tt-metal`
- `tenstorrent/vllm` → `vllm`
- `tenstorrent/tt-inference-server` → `tt-inference-server`

If git remote fails (not a git repo), ask the user which repo they're working in.

### 2. Check for recipe

Look for `knowledge/recipes/<repo>/index.md` in the tt-agent directory.
If found, recipes are available for this repo. If not, tt-run still works
but relies on explicit user commands or tt-learn for context.

### 3. Detect platform

Determine whether commands run locally or on a remote machine via MCP.

**Check for tt-device-mcp availability:**
- If `tt_device_queue_status` MCP tool is available → remote device execution is possible
- If not → local execution only (build, host-side testing)

**Local Mac (no device):** Can build, run host-side Python, but cannot run
device tests. Useful for development and code review.

**Remote TT machine (via MCP):** Full capability — build, test, run on device.

### 4. Detect architecture

If on a machine with TT devices:

```bash
tt_device_exec -- "echo $ARCH_NAME"
```

Or detect from `ttnn` if importable:

```python
import ttnn
arch = ttnn.device.get_arch()  # wormhole_b0, blackhole, etc.
```

If architecture cannot be detected (e.g., local Mac), note it as unknown.
Recipes and tests that need a specific arch will fail at runtime with a
clear error.

## Output

Produce a workspace context summary:

```
Repo: tt-metal
Recipe: knowledge/recipes/tt-metal/ (available)
Platform: remote (tt-device-mcp available)
Architecture: wormhole_b0
```

This summary is consumed by the recipe loading and command composition steps.
