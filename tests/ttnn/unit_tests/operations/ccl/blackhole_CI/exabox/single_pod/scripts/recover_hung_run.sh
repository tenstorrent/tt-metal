#!/bin/bash
# Kill local + remote pytest/prte processes after a hung run. Run reset_chips.sh
# afterwards before launching the next test.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_hosts.sh"

echo "[recover] killing local launcher processes..."
pkill -9 -f tt-run    2>/dev/null || true
pkill -9 -f prterun   2>/dev/null || true
pkill -9 -f "pytest.*single_pod" 2>/dev/null || true

echo "[recover] killing remote pytest/prted on each host..."
for h in $SINGLE_POD_HOSTS; do
  ssh -o BatchMode=yes "$h" "pkill -9 -f pytest 2>/dev/null; pkill -9 -f prted 2>/dev/null" &
done
wait
echo "[recover] done. Run scripts/reset_chips.sh before launching the next test."
