#!/usr/bin/env bash
# ensure-cron.sh — keep the SDPA watcher's cron scheduler alive across reboots.
#
# This container's rootfs is ephemeral: a reboot wipes installed OS packages
# (cron included) while $HOME and /localdev persist. There is no systemd, so
# nothing auto-starts at boot. The one hook that always fires is a login shell,
# so ~/.bashrc calls this script; it is a fast no-op when the scheduler is
# healthy and self-repairs (detached, so it never blocks the shell) otherwise.
#
# Idempotent and safe to run from every interactive shell.
set -u

SDPA_HOME="$HOME/.sdpa-watch"
LOG="$SDPA_HOME/bootstrap.log"
LOCK="$SDPA_HOME/.bootstrap.lock"
MARKER='.sdpa-watch/watch.sh'
CRON_LINE='0 * * * * $HOME/.sdpa-watch/watch.sh >> $HOME/.sdpa-watch/watch.log 2>&1'

daemon_up()  { pgrep -x cron >/dev/null 2>&1; }
tab_present() { crontab -l 2>/dev/null | grep -qF "$MARKER"; }

# Fast path: scheduler already installed, running, and scheduled.
if daemon_up && tab_present; then
  exit 0
fi

# Repair. May apt-install (network + sudo), so run detached and single-flighted
# (a burst of logins right after a reboot must not race on the dpkg lock).
(
  exec 9>"$LOCK"
  flock -n 9 || exit 0   # another repair already in flight

  ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
  echo "[$(ts)] repair start (daemon_up=$(daemon_up && echo 1 || echo 0) tab_present=$(tab_present && echo 1 || echo 0))"

  if [[ ! -x /usr/sbin/cron ]]; then
    echo "[$(ts)] cron package missing — installing"
    sudo -n apt-get install -y cron >/dev/null 2>&1 \
      && echo "[$(ts)] cron installed" \
      || echo "[$(ts)] ERROR: apt install cron failed (need network + passwordless sudo)"
  fi

  if ! daemon_up; then
    echo "[$(ts)] starting cron daemon"
    sudo -n service cron start >/dev/null 2>&1 \
      && echo "[$(ts)] cron daemon started" \
      || echo "[$(ts)] ERROR: 'service cron start' failed"
  fi

  if ! tab_present; then
    echo "[$(ts)] installing crontab entry"
    # Strip any pre-existing watcher lines first so a login burst after reboot
    # can never accumulate duplicates, then append exactly one.
    ( crontab -l 2>/dev/null | grep -vF "$MARKER"; echo "$CRON_LINE" ) | crontab - \
      && echo "[$(ts)] crontab entry installed" \
      || echo "[$(ts)] ERROR: crontab install failed"
  fi

  echo "[$(ts)] repair done"
) >> "$LOG" 2>&1 &
disown 2>/dev/null || true
exit 0
