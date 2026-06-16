#!/usr/bin/env bash
# Runner-side gate for the TTI release stage: the customer-facing release report
# and run notes must be copied into the target model's doc/tti_release directory.
# Exit 0 pass, 1 advisory, 2 critical, 3 error.
set -uo pipefail

if [ -n "${MODEL_DIR:-}" ]; then
  model_dir="$MODEL_DIR"
elif [ -n "${HF_MODEL:-}" ]; then
  slug=$(printf '%s' "$HF_MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_//; s/_$//')
  model_dir="models/autoports/$slug"
else
  echo "Neither MODEL_DIR nor HF_MODEL is set; cannot scope the check to the target model." >&2
  exit 3
fi

release_dir="$model_dir/doc/tti_release"
if [ ! -d "$release_dir" ]; then
  echo "Missing TTI release evidence directory: $release_dir" >&2
  exit 2
fi

report_count=$(find "$release_dir" -type f -path '*/release/report_*.md' 2>/dev/null | wc -l | tr -d ' ')
if [ "$report_count" -lt 1 ]; then
  report_count=$(find "$release_dir" -maxdepth 2 -type f -name 'report_*.md' 2>/dev/null | wc -l | tr -d ' ')
fi
if [ "$report_count" -lt 1 ]; then
  echo "No release report_*.md found under $release_dir." >&2
  exit 2
fi

notes="$release_dir/RUN_NOTES.md"
if [ ! -s "$notes" ]; then
  echo "Missing non-empty RUN_NOTES.md under $release_dir." >&2
  exit 2
fi

if ! grep -Eq 'EXIT_CODE=0|exit code:?[[:space:]]*0|exited[[:space:]]+0|completed successfully' "$notes"; then
  echo "RUN_NOTES.md does not record a successful release workflow exit." >&2
  exit 2
fi

if ! grep -Eiq 'physical host|loudbox|wh-lb-|docker image|tt-inference-server|repo tag|git head|git sha' "$notes"; then
  echo "RUN_NOTES.md is missing host/version evidence expected for handoff." >&2
  exit 1
fi

echo "TTI release evidence present under $release_dir ($report_count release report(s))."
