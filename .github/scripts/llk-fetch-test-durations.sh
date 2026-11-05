#!/usr/bin/env bash
# Fetch the per-test durations that the LLK legs upload as test-durations-<arch>-*,
# from the newest successful run of <workflow-file> on <branch> for <event>.
# Writes <out-dir>/<arch>.json (test id -> seconds, summed over all files of the
# arch) and found=true|false to $GITHUB_OUTPUT. Exits 0 when nothing is found.
#
# Usage: llk-fetch-test-durations.sh <workflow-file> <branch> <event> <out-dir>
set -euo pipefail

WORKFLOW="${1:?usage: llk-fetch-test-durations.sh <workflow-file> <branch> <event> <out-dir>}"
BRANCH="${2:?usage: llk-fetch-test-durations.sh <workflow-file> <branch> <event> <out-dir>}"
EVENT="${3:?usage: llk-fetch-test-durations.sh <workflow-file> <branch> <event> <out-dir>}"
OUT="${4:?usage: llk-fetch-test-durations.sh <workflow-file> <branch> <event> <out-dir>}"
DOWNLOAD="$(mktemp -d)"

for run in $(gh run list -w "$WORKFLOW" -b "$BRANCH" -e "$EVENT" -s success -L 5 --json databaseId --jq '.[].databaseId'); do
  gh run download "$run" -p 'test-durations-*' -D "$DOWNLOAD" && echo "durations from run $run" && break
done

python3 - "$DOWNLOAD" "$OUT" << 'PY'
import json
import os
import sys
from pathlib import Path

download, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
found = False
for arch in ("wormhole", "blackhole", "quasar"):
    summed = {}
    for leg in sorted(download.glob(f"test-durations-{arch}-*/.test_durations")):
        for test, seconds in json.loads(leg.read_text()).items():
            summed[test] = summed.get(test, 0.0) + seconds
    print(f"{arch}: durations for {len(summed)} tests")
    if summed:
        (out / f"{arch}.json").write_text(json.dumps(summed))
        found = True
if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        handle.write(f"found={str(found).lower()}\n")
print(f"found={str(found).lower()}")
PY
