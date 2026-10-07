#!/bin/bash
# 255: tunnel down. 1: copy (run 849) still running, or it passed but /data has < 230 GB free.
# 0: copy failed (needs a look), or copy passed and /data has >= 230 GB free (submit e2e).
timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=15 exabox-login true || exit 255
RC=/home/smarton/fasth3/tt-metal/tt-project/state/runs/849/t160-pcopy2.rc
[ -e $RC ] || exit 1
grep -q PCOPY_OK /home/smarton/fasth3/tt-metal/tt-project/state/runs/849/t160-pcopy2.log || exit 0
avail=$(timeout 30 ssh -o BatchMode=yes exabox-login "df -B1G --output=avail /data | tail -1") || exit 255
[ "${avail// /}" -ge 230 ] && exit 0
exit 1
