#!/bin/bash
# Reset chips on all 4 single-pod hosts in parallel (~60s wallclock).
# Required before the first test of a session and after any hung run.

case "${1:-}" in
  -h|--help)
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Resets chips on all hosts in \$HOSTS in parallel by running
'tt-smi -glx_reset_auto' on each. Wallclock ≈ 60 s.

Required environment:
  TT_METAL_HOME    Repo root (used to find tt-smi). Default:
                     /data/llong/tt-metal

Optional environment:
  HOSTS Space- or comma-separated host list (4 hosts). Default in _hosts.sh:
                     bh-glx-110-c07u02 bh-glx-110-c07u08
                     bh-glx-110-c08u02 bh-glx-110-c08u08
                   *** OVERRIDE THIS for a different cluster. ***

When to run:
  - Before the first test of a session.
  - After any hung run (after recover_hung_run.sh).
  - Whenever you see 'Device N init: failed to initialize FW' in test logs.

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

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_hosts.sh"
TT_METAL_HOME="${TT_METAL_HOME:-/data/llong/tt-metal}"

echo "[reset_chips] resetting on: $HOSTS"
for h in $HOSTS; do
  ssh -o BatchMode=yes "$h" "$TT_METAL_HOME/python_env/bin/tt-smi -glx_reset_auto" &
done
wait
echo "[reset_chips] done."
