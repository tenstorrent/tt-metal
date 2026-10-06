#!/bin/bash
# Exit 0 when t160 can go on: prep failed (needs a fix), or copy + prep done and a dit node qualifies.
/home/smarton/fasth3/tt-metal/tt-project/harness/bin/ttp detach --check /home/smarton/fasth3/tt-metal/tt-project/state/runs/648/t160-copy.rc >/dev/null || exit 1
exec ssh -o BatchMode=yes -o ConnectTimeout=10 exabox-login 'F=/data/smarton/fasth3; grep -q "PREP_RC=[1-9]" $F/prep.log 2>/dev/null && exit 0; grep -q "PREP_RC=0" $F/prep.log 2>/dev/null || exit 1; bash $F/qualify.sh >/dev/null'
