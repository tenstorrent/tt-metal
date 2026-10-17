#!/bin/bash
# usage: ledger_log.sh <ledger.md> <text...>
# Append one line to the ledger's Log section with the CLOCK's UTC time (`- HH:MM text`), never an estimated
# one. Use it for every log line; a hand-written time is the thing this script exists to prevent.
set -u
ledger=$1; shift
printf -- '- %s %s\n' "$(date -u +%H:%M)" "$*" >> "$ledger"
