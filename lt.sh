#!/bin/bash
# usage: lt.sh <host> <logname> "<ENV...>" <pytest args...>   (launch a device job on a host under the flock, in background)
H=$1; N=$2; ENVS=$3; shift 3
W=/mnt/tt-data/ssinghal/wt/spec_int
ARGS=$(printf '%q ' "$@")
CMD="export $ENVS DSV41_SPEC_PRINT=2; timeout 14400 pytest -x -s -q $ARGS > /mnt/tt-data/ssinghal/dsv4-logs/pf_spec_int_$N.log 2>&1"
ssh -o BatchMode=yes 10.82.97.$H "nohup $W/run.sh $(printf '%q' "$CMD") > /dev/null 2>&1 &" 2>/dev/null
