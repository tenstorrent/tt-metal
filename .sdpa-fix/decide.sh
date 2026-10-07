#!/usr/bin/env bash
# Keeps the Slack decision-button listener (decide.py) running. Cron calls this
# every 5 minutes: if a listener already holds the lock it exits at once,
# otherwise it becomes the listener (so a crash or reboot heals within 5 min).
# Does nothing until the Socket Mode app token exists.
set -uo pipefail
FIX_HOME="$HOME/.sdpa-fix"
exec 9>"$FIX_HOME/.decide.lock"
flock -n 9 || exit 0
[[ -s "$FIX_HOME/slack_app_token" ]] || exit 0
source "$FIX_HOME/config.sh"
mkdir -p "$FIX_HOME/logs"
SLACK_APP_TOKEN="$(tr -d '[:space:]' < "$FIX_HOME/slack_app_token")"
SLACK_BOT_TOKEN="$(tr -d '[:space:]' < "$SLACK_BOT_TOKEN_FILE")"
export SLACK_APP_TOKEN SLACK_BOT_TOKEN DECIDERS
exec python3 "$FIX_HOME/decide.py" >> "$FIX_HOME/logs/decide.log" 2>&1
