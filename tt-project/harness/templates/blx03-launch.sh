#!/bin/bash
# Launch a task's driver detached ON blx03 so it survives g15blx02 reboots.
# Usage: blx03-launch.sh <task> <driver path on blx03>   e.g. blx03-launch.sh t97 ~/fasth3/t97drv/driver.sh
# Driver: copy of templates/blx03/driver.sh (originally tmp/blx03/drv_template/, archived on
# ttp/t48-notes-archive); prefer the serial runner in templates/blx03-runner/. It must log
# "T<n>_DRIVER_DONE <stage> <rc>" to /var/tmp/fasth3/<task>/driver.log, as the template does.
set -eu
T=$1; DRV=$2; V=/var/tmp/fasth3/$T
ssh g14blx03 "mkdir -p $V && setsid nohup bash $DRV > $V/driver.out 2>&1 < /dev/null & echo launched pid \$!"
# ssh exits 255 when blx03 is down; map every non-0 to 1 so the harness keeps sleeping.
echo "retry_when: ssh -o BatchMode=yes -o ConnectTimeout=20 g14blx03 'grep -q _DRIVER_DONE $V/driver.log' || exit 1"
