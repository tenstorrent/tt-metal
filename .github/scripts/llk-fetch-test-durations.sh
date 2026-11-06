#!/usr/bin/env bash
# Sum the test durations of the newest run of $1 that stored any, into $4.
# Most merge-queue runs skip the LLK perf legs, so the window has to be wide;
# the artifact listing keeps a miss to one cheap API call, not a download.
set -euo pipefail

ROOT="$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)"

ARGS=(-w "$1" -e "$3" -s success -L 60)
[ -n "$2" ] && ARGS+=(-b "$2")

DOWNLOAD="$(mktemp -d)"
for run in $(gh run list "${ARGS[@]}" --json databaseId --jq '.[].databaseId'); do
  names="$(gh api --paginate "repos/$GH_REPO/actions/runs/$run/artifacts" --jq '.artifacts[].name')"
  grep -q '^test-durations-' <<<"$names" || continue
  # A half-finished download must not be summed with the next run's files.
  DOWNLOAD="$(mktemp -d)"
  gh run download "$run" -p 'test-durations-*' -D "$DOWNLOAD" && break
done

python3 "$ROOT/tt_metal/tt-llk/perf/sum_split_durations.py" "$DOWNLOAD" "$4"
