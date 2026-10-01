#!/bin/bash
# Kill local + remote pytest/prte processes after a hung run. Run reset_chips.sh
# afterwards before launching the next test.

case "${1:-}" in
  -h|--help)
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Force-kills lingering tt-run / prterun / pytest / prted processes locally
and on every host in \$HOSTS. Use after a hung test run.

Required environment:
  HOSTS            Space- or comma-separated 4-host list. NO DEFAULT —
                   set per-shell, e.g.
                     export HOSTS="hostA hostB hostC hostD"

When to run:
  - After ctrl-C'ing a hung run.
  - When the next test refuses to start (chip locks held, port collisions).
  - Symptoms: 'Waiting for lock CHIP_IN_USE_*_PCIe' in the next run's log.

Always follow with reset_chips.sh before the next test:
  ./recover_hung_run.sh && ./reset_chips.sh

Examples:
  bash $0
  HOSTS="h1 h2 h3 h4" bash $0
EOF
    exit 0
    ;;
  "") ;;
  *)
    echo "[error] unexpected argument: $1" >&2
    echo "Run with --help for usage." >&2
    exit 2
    ;;
esac

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_hosts.sh"

echo "[recover] killing local launcher processes..."
pkill -9 -f tt-run    2>/dev/null || true
pkill -9 -f prterun   2>/dev/null || true
pkill -9 -f "pytest.*deepseek_pipeline_tests" 2>/dev/null || true

echo "[recover] killing remote pytest/prted on each host..."
for h in $HOSTS; do
  ssh -o BatchMode=yes "$h" "pkill -9 -f pytest 2>/dev/null; pkill -9 -f prted 2>/dev/null" &
done
wait
echo "[recover] done. Run scripts/reset_chips.sh before launching the next test."
