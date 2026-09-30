#!/bin/bash
# Queue ONE project job on blx03s broker, refusing while another smarton job is running or queued
# (one-project-device-job rule). Usage, on blx03: tmp/blx03/submit.sh <timeout_s> <command...>
# Prints the broker job id. Exit 75 = another project job is active; try later.
t=$1; shift
active=$(tt-device-mcp status 1 2>&1 | sed -n "/^RUNNING/,/^RECENT/p" | grep -w smarton)
if [ -n "$active" ]; then echo "busy: $active"; exit 75; fi
cd /home/smarton/fasth3/tt-metal && tt-device-mcp run-bg "$*" -w "$PWD" -t "$t"
