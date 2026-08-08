#!/usr/bin/env bash
# Decides whether the cron that just fired is the one that means 06:00 Europe/Berlin.
#
# GitHub cron is UTC-only and DST-blind, so sync-upstream.yml schedules BOTH candidate hours
# (04:00Z = 06:00 CEST, 05:00Z = 06:00 CET) and this script drops the wrong twin. It keys off
# WHICH cron fired, never off the current wall clock: scheduled runs are best-effort and start
# late routinely (observed: a 06:00 cron beginning at 08:27Z), so a wall-clock hour comparison
# would skip the sync on every delayed run.
#
# FAILS OPEN. Anything unrecognised -- empty cron (manual dispatch), an unexpected tz offset, a
# cron string we do not know -- returns true. A spurious extra sync is a no-op ("already up to
# date"); a spurious skip silently drifts the branch behind upstream, which is the failure that
# actually costs something.
#
# Usage: berlin_gate.sh <fired-cron> [reference-date]
#   stdout: exactly "true" or "false"   (the gate decision, for GITHUB_OUTPUT)
#   stderr: human-readable reasoning
set -euo pipefail

FIRED="${1:-}"
REF="${2:-now}"

SUMMER_CRON="0 4 * * *"   # 04:00 UTC == 06:00 CEST
WINTER_CRON="0 5 * * *"   # 05:00 UTC == 06:00 CET

OFFSET=$(TZ=Europe/Berlin date -d "$REF" +%z)

case "$OFFSET" in
  +0200) WANT="$SUMMER_CRON" ;;
  +0100) WANT="$WINTER_CRON" ;;
  *)
    echo "gate: unexpected Europe/Berlin offset '$OFFSET' -- failing OPEN" >&2
    echo "true"; exit 0 ;;
esac

if [ -z "$FIRED" ]; then
  echo "gate: no cron supplied (manual dispatch, or missing event payload) -- failing OPEN" >&2
  echo "true"; exit 0
fi

if [ "$FIRED" = "$WANT" ]; then
  echo "gate: offset $OFFSET wants '$WANT' and '$FIRED' fired -> RUN" >&2
  echo "true"
elif [ "$FIRED" = "$SUMMER_CRON" ] || [ "$FIRED" = "$WINTER_CRON" ]; then
  echo "gate: offset $OFFSET wants '$WANT' but '$FIRED' fired -> skip (off-DST twin)" >&2
  echo "false"
else
  echo "gate: unrecognised cron '$FIRED' -- failing OPEN" >&2
  echo "true"
fi
