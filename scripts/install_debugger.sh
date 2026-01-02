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
pip install --extra-index-url https://test.pypi.org/simple/ tt-exalens=="$REF"
