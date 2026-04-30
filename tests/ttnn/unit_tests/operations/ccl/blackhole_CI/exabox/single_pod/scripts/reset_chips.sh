#!/bin/bash
# Reset chips on all 4 single-pod hosts in parallel (~60s wallclock).
# Required before the first test of a session and after any hung run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_hosts.sh"
TT_METAL_HOME="${TT_METAL_HOME:-/data/llong/tt-metal}"

for h in $SINGLE_POD_HOSTS; do
  ssh -o BatchMode=yes "$h" "$TT_METAL_HOME/python_env/bin/tt-smi -glx_reset_auto" &
done
wait
echo "[reset_chips] done."
