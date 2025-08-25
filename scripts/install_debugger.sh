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

echo "Installing ttexalens at ref: $REF"
python3 -m pip uninstall -y ttexalens >/dev/null 2>&1 || true
python3 -m pip install "git+https://github.com/tenstorrent/tt-exalens.git@$REF"
