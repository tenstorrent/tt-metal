#!/bin/bash
# t503: after staging, copy the job script to blx03 and queue A/B attempts r1, r2 on blx03's serial runner.
set -eo pipefail
D=/home/smarton/fasth3/tt-metal/tt-project/t503; RC=/home/smarton/fasth3/tt-metal/tt-project/state/runs/1898/stage503.rc
for i in $(seq 1 200); do [ -e $RC ] && break; sleep 30; done
[ "$(cat $RC 2>/dev/null)" = 0 ] || { echo "staging failed rc=$(cat $RC 2>/dev/null)"; exit 4; }
ssh g14blx03 "mkdir -p /var/tmp/fasth3/t503/plug"
scp -q $D/plug498.py g14blx03:/var/tmp/fasth3/t503/plug/plug498.py
scp -q $D/run503.sh g14blx03:/var/tmp/fasth3/t503/run503.sh.tmp && ssh g14blx03 "mv /var/tmp/fasth3/t503/run503.sh.tmp /var/tmp/fasth3/t503/run503.sh"
E=/home/smarton/fasth3/tt-metal/tt-project/harness/templates/blx03-runner/blx03-enqueue.sh
$E $D/spec-r1.txt
$E $D/spec-r2.txt
echo "QUEUED503 $(date -u +%T)"
