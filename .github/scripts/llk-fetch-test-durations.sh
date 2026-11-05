#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD="$(mktemp -d)"

for run in $(gh run list -w "$1" -b "$2" -e "$3" -s success -L 5 --json databaseId --jq '.[].databaseId'); do
  gh run download "$run" -p 'test-durations-*' -D "$DOWNLOAD" && break
done

python3 - "$DOWNLOAD" "$4" << 'PY'
import json
import os
import sys
from pathlib import Path

download, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
found = False
for arch in ("wormhole", "blackhole"):
    summed = {}
    for leg in sorted(download.glob(f"test-durations-{arch}-*/.test_durations")):
        for test, seconds in json.loads(leg.read_text()).items():
            summed[test] = summed.get(test, 0.0) + seconds
    print(f"{arch}: durations for {len(summed)} tests")
    if summed:
        (out / f"{arch}.json").write_text(json.dumps(summed))
        found = True
with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
    handle.write(f"found={str(found).lower()}\n")
PY
