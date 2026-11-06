#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD="$(mktemp -d)"

ARGS=(-w "$1" -e "$3" -s success -L 20)
[ -n "$2" ] && ARGS+=(-b "$2")

for run in $(gh run list "${ARGS[@]}" --json databaseId --jq '.[].databaseId'); do
  gh run download "$run" -p 'test-durations-*' -D "$DOWNLOAD" && break
done

python3 "${BASH_SOURCE[0]%/*}/utils/llk_sum_test_durations.py" "$DOWNLOAD" "$4"
