#!/usr/bin/env bash
# Print (default) or install (--install) the crontab lines for the scheduled
# BH sweeps. The orchestrator installs this after merge; agents only print.
#
# Both entries are flock-guarded (a still-running sweep is never doubled) and
# log to ~/sfpi-uplift/sweep-logs/. The sweeps themselves take the device
# locks per HANDOFF §1(5); these outer flocks only prevent overlapping cron
# starts of the same sweep.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOGDIR="$HOME/sfpi-uplift/sweep-logs"

LINES="# tt-llk SFPU 2x2 sweeps (installed by corpus/install_sweep_cron.sh)
0 2 * * * mkdir -p $LOGDIR && flock -n /tmp/tt-sweep-nightly.cron.lock -c '$HERE/nightly_bh_sweep.sh' >> $LOGDIR/nightly-\$(date +\\%Y\\%m\\%d).log 2>&1
0 4 * * 0 mkdir -p $LOGDIR && flock -n /tmp/tt-sweep-weekly.cron.lock -c '$HERE/weekly_bh_sweep.sh' >> $LOGDIR/weekly-\$(date +\\%Y\\%m\\%d).log 2>&1"

if [ "${1:-}" = "--install" ]; then
  ( crontab -l 2>/dev/null | grep -v 'install_sweep_cron.sh\|nightly_bh_sweep.sh\|weekly_bh_sweep.sh'; echo "$LINES" ) | crontab -
  echo "installed:"
  crontab -l | tail -3
else
  echo "# Not installed. Review, then run: $0 --install"
  echo "$LINES"
fi
