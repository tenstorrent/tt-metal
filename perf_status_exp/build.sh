#!/bin/bash
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522
cd $W
source /localdev/mbezulj/tt-metal/python_env/bin/activate
TS=$(date +%Y%m%d_%H%M%S)
LOG=/localdev/mbezulj/logs/merge8_build_$TS.log
git submodule update --init --recursive > $LOG 2>&1
echo "SUBMOD_RC=$?" >> $LOG
nice -n 10 ./build_metal.sh -c >> $LOG 2>&1
RC=$?
echo "BUILD_RC=$RC $TS" >> $LOG
echo "BUILD_RC=$RC $LOG" >> /localdev/mbezulj/logs/merge8_build_done.txt
