#!/bin/bash
# PC samples of ring 1-hop and Y-torus N/S 1-hop writes (deadlock avoidance + first-level acks), v4 and v3q.
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
V=fabric_exp/variants
bash $CON/pcsv_runs.sh $V/v4 v4ring Ring ring1
bash $CON/pcsv_runs.sh $V/v4 v4tory TorusY torusns
bash $CON/pcsv_runs.sh $V/v3q v3qring Ring ring1
bash $CON/pcsv_runs.sh $V/v3q v3qtory TorusY torusns
