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

# Uninstall tt-exalens if already installed
echo "Installing tt-exalens version: $REF"
python3 -m pip uninstall -y tt-exalens >/dev/null 2>&1 || true

# Figure download link
PYTHON_CP_VERSION="$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
DOWNLOAD_LINK="https://github.com/tenstorrent/tt-exalens/releases/download/v$REF/tt_exalens-$REF-$PYTHON_CP_VERSION-$PYTHON_CP_VERSION-linux_x86_64.whl"

# Check if download link exists
if ! curl -sSfL -r 0-0 "$DOWNLOAD_LINK" -o /dev/null; then
  echo "Prebuilt wheel not found for version $REF. Installing from source."
  python3 -m pip install "git+https://github.com/tenstorrent/tt-exalens.git@v$REF"
else
  python3 -m pip install "$DOWNLOAD_LINK"
fi
