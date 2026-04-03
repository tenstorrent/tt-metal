# Developer Environment Setup

Global setup that persists across workspaces. Set once per machine, stored in
shell profile or `~/.secrets`. This is not workspace-specific — it applies to
all workspaces on this machine.

## Tokens and secrets

Store in `~/.secrets` (not checked into git, sourced from shell profile):

```bash
# ~/.secrets
export HF_TOKEN="hf_..."          # HuggingFace — needed for model downloads
export GH_TOKEN="ghp_..."         # GitHub — needed for gh CLI and private repos
```

## Cache paths

Large model weights and build caches should live on local storage, not in `$HOME`:

```bash
export HF_HOME="/localdev/$USER/hf_data"       # HuggingFace model cache
export CCACHE_DIR="/localdev/$USER/.ccache"     # C++ compilation cache
export UV_CACHE_DIR="/localdev/$USER/.uvcache"  # uv package cache
```

Paths vary by machine. The key principle: keep large caches off NFS/home.

## System packages

These are typically needed on a fresh TT machine (one-time, may need sudo):

- `cmake` (or `pip install cmake`)
- `ccache`
- `git-lfs`
- LLVM 17+ with libc++ (`apt install libc++-17-dev libc++abi-17-dev`)
- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## TT device tools

```bash
pip install git+https://github.com/tenstorrent/tt-smi
pip install git+https://github.com/tenstorrent/tt-topology
```

## tt-device-mcp (for agent-driven development)

Requires manual setup — the agent cannot install MCP servers or restart itself.
The developer must do this before starting a Claude Code session:

```bash
pip install git+https://github.com/tenstorrent/tt-device-mcp.git
tt-device-mcp daemon start
tt-device-mcp claude-add-mcp   # adds MCP config to Claude Code settings
# Then restart Claude Code
```

If tt-run detects that tt-device-mcp tools are unavailable, it should tell the
developer to run these steps and restart, not attempt to install it.

## Shell profile

Source secrets and set cache paths from your shell profile (`.bashrc` or `.zshrc`):

```bash
source ~/.secrets
export HF_HOME="/localdev/$USER/hf_data"
export CCACHE_DIR="/localdev/$USER/.ccache"
```

## What this does NOT cover

- Workspace creation (see `recipes/workspace.md`)
- Building tt-metal (see `recipes/tt-metal/build.md`)
- Per-repo env vars (see `recipes/<repo>/env.md`)
