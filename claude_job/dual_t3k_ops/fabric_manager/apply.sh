#!/usr/bin/env bash
# Recreate the fabric-controller so it reads its config + FSD from THIS repo dir,
# making the web UI "FSD (factory)" source render all 16 chips (GUIDE.md §8b).
#
# Read-only / observability only — does NOT touch the fabric, so no moreh-lock
# needed. Safe to re-run; `--restart unless-stopped` makes it survive reboots.
# Run this on the launcher (t3k-node-b / 192.168.1.243), where the repo is local.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=ghcr.io/tenstorrent/tt-fabric-manager:latest

docker rm -f fabric-controller 2>/dev/null || true
docker run -d --name fabric-controller --network host --restart unless-stopped \
  -v "$HERE/controller.yaml:/etc/fabric-manager/config.yaml:ro" \
  -v "$HERE/fsd:/etc/fabric-manager/fsd:ro" \
  "$IMAGE" \
  tt-fabric-manager-controller --config /etc/fabric-manager/config.yaml

echo "controller restarted; polling until both agents re-register (up to 40s)..."
for i in $(seq 1 20); do
  sleep 2
  n=$(curl -s "http://192.168.1.243:8080/api/physical-topology?source=fsd" \
      | python3 -c 'import json,sys;print(len(json.load(sys.stdin).get("asicDescriptors",[])))' 2>/dev/null || echo 0)
  if [ "$n" = "16" ]; then echo "FSD (factory) view: 16 ASICs ✓"; break; fi
done
[ "${n:-0}" = "16" ] || echo "WARNING: FSD view returned $n ASICs (expected 16) — agents may still be registering; re-check in a moment."
echo "Open http://192.168.1.243:8080 and set Source = FSD (factory)."
