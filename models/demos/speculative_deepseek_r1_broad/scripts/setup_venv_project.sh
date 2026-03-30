#!/bin/bash
# Run this on the compute server when home is full. Creates a venv on PROJECT disk
# and installs packages there so pip never writes to ~/.local (which is full).
# Usage: bash models/demos/speculative_deepseek_r1_broad/scripts/setup_venv_project.sh

set -e
REPO=/proj_sw/user_dev/dchrysostomou/tt-metal
VENV=$REPO/venv
cd "$REPO"

# 1) Free a bit of space in home (remove failed/cached pip installs under .local)
echo "Cleaning home .local to free space..."
rm -rf ~/.local/lib/python3.10/site-packages/nvidia 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true
echo "Done."

# 2) Create venv on PROJECT disk (not in home)
echo "Creating venv at $VENV ..."
deactivate 2>/dev/null || true
python3 -m venv "$VENV"
source "$VENV/bin/activate"

# 3) Ensure pip never uses home
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=$REPO/.pip_cache
export PIP_USER=0
mkdir -p "$PIP_CACHE_DIR"

# 4) Upgrade pip in the venv (so it's the venv's pip)
"$VENV/bin/pip" install --no-cache-dir -q --upgrade pip

# 5) Install into venv (accelerate required by transformers when low_cpu_mem_usage=True)
echo "Installing packages into venv (no cache)..."
"$VENV/bin/pip" install --no-cache-dir torch transformers==4.46.3 safetensors huggingface_hub 'accelerate>=0.26.0'

echo ""
echo "=== Done. Use this venv from now on ==="
echo "  source $VENV/bin/activate"
echo "  export PYTHONNOUSERSITE=1"
echo "  export HF_HOME=$REPO/hf_cache"
echo "  export PYTHONPATH=$REPO:\$PYTHONPATH"
echo ""
echo "Then run your scripts with:  python models/demos/speculative_deepseek_r1_broad/scripts/..."
echo "Or add to your shell profile: source $REPO/venv/bin/activate && export PYTHONNOUSERSITE=1 HF_HOME=$REPO/hf_cache PYTHONPATH=$REPO:\$PYTHONPATH"
