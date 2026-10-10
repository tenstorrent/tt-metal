#!/bin/bash
# t315 retry_when: exit 0 once blx03 (build done) or blx01 runs at full AICLK, i.e. its newest device-opening broker
# job log has no "AICLK failed to settle" warning; 1 otherwise (ssh failure counts as not yet).
newest_clean() {  # $1 host: 0 if the newest device-opening job ran unclamped
  timeout 25 ssh -o ConnectTimeout=10 -o BatchMode=yes $1 '
    cd /var/log/tt-device-broker || exit 1
    for f in $(ls -t 2026-*.log | head -15); do
      grep -q "Opening user mode device driver" "$f" || continue
      grep -q "AICLK failed to settle" "$f" && exit 1 || exit 0
    done; exit 1'
}
if timeout 25 ssh -o ConnectTimeout=10 -o BatchMode=yes g14blx03 'grep -q "T315_DRIVER_DONE setup 0" /var/tmp/fasth3/t315/driver.log' \
   && newest_clean g14blx03; then exit 0; fi
newest_clean blx01 && exit 0
# a failed blx03 build also wakes the task (to fix it)
timeout 25 ssh -o ConnectTimeout=10 -o BatchMode=yes g14blx03 'grep -qE "T315_DRIVER_DONE setup [1-9]" /var/tmp/fasth3/t315/driver.log' && exit 0
exit 1
