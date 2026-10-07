#!/bin/bash
# 0: chunk verify + recopy (run 817) ended, ok or not. 1: still running. 255: tunnel down.
timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=15 exabox-login true || exit 255
/home/smarton/fasth3/tt-metal/tt-project/harness/bin/ttp detach --check /home/smarton/fasth3/tt-metal/tt-project/state/runs/817/t160-pverify.rc >/dev/null 2>&1 && exit 0
exit 1
