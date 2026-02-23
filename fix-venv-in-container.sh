#!/bin/bash
# Run inside the container to fix venv: install Python where the host venv symlink points.
# The venv's python -> /home/dmadic/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10

set -e
TARGET_DIR="/home/dmadic/.local/share/uv/python"
CURRENT_USER="${SUDO_USER:-$USER}"

# 1. Create directory (with sudo if needed) and give ownership to current user
if [ ! -d "$TARGET_DIR" ]; then
  sudo mkdir -p "$TARGET_DIR"
  sudo chown -R "$CURRENT_USER:$CURRENT_USER" /home/dmadic/.local
fi

# 2. Install Python with uv AS THE NORMAL USER so UV_PYTHON_INSTALL_DIR is used
export UV_PYTHON_INSTALL_DIR="$TARGET_DIR"
uv python install 3.10.19

echo "Done. Activate venv and run: source python_env/bin/activate && python --version"
