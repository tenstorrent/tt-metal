#!/bin/bash
# exit 0 once the t274 driver failed before submitting, or its broker job has finished; 1 otherwise.
D=/var/tmp/fasth3/t274/drv
[ -e $D/driver.marker ] || exit 1
[ -s $D/job.id ] || exit 0
s=$(timeout 50 tt-device-mcp status -j $(cat $D/job.id) 2>&1 | sed -n 's/^Status: *//p' | head -1)
case "$s" in queued | running | pending | "") exit 1 ;; *) exit 0 ;; esac
