#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REF_FILE="$SCRIPT_DIR/ttexalens_ref.txt"

if [[ ! -f "$REF_FILE" ]]; then
  echo "ttexalens_ref.txt not found at: $REF_FILE" >&2
  exit 1
fi

REF="$(head -n1 "$REF_FILE" | tr -d '[:space:]')"
if [[ -z "$REF" ]]; then
  echo "ttexalens_ref.txt is empty" >&2
  exit 1
fi

# Determine whether to use uv or pip
if command -v uv &>/dev/null; then
  PIP_CMD="uv pip"
  echo "Using uv for package management"
else
  PIP_CMD="python3 -m pip"
  echo "uv not found, falling back to pip"
fi

# Uninstall tt-exalens if already installed
$PIP_CMD uninstall -y tt-exalens >/dev/null 2>&1 || true

# Install tt-exalens to the requested version
echo "Installing tt-exalens version: $REF"
$PIP_CMD install --extra-index-url https://test.pypi.org/simple/ tt-exalens=="$REF"
