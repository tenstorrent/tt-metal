#!/usr/bin/env bash
# galaxy-kit collect.sh — pull results home and emit the replication ledger.
#
#   collect.sh -w <workdir> [-d <dest>] [-r <reps>] [-H "<provenance>"]
#
# Streams results/ + wlogs/ back through the relay (no relay disk), then
# runs lib/ledger.py -> REPLICATION-LEDGER.tsv / -PAIRS.tsv / -VERDICTS.tsv
# in the workdir.
set -uo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
source "$KIT/lib/remote.sh"

WORK=""; REPS=5; HEAD=""
while [ $# -gt 0 ]; do
  case "$1" in
    -w) WORK=$2; shift 2;;
    -d) LK_DEST=$2; shift 2;;
    -r) REPS=$2; shift 2;;
    -H) HEAD=$2; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
: "${WORK:?-w <workdir> required}"
route_check || exit 2
exa_get "cd $LK_DEST && tar czf - results wlogs 2>/dev/null" \
  > "$WORK/galaxy-results.tar.gz" || exit 2
ls -la "$WORK/galaxy-results.tar.gz"
mkdir -p "$WORK/galaxy"
tar -C "$WORK/galaxy" -xzf "$WORK/galaxy-results.tar.gz"
python3 "$KIT/lib/ledger.py" --work "$WORK" --results "$WORK/galaxy/results" \
  --reps "$REPS" ${HEAD:+--headline "$HEAD"}
echo "COLLECT-DONE ($WORK)"
