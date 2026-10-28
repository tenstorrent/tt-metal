#!/bin/bash
# Small-write and mixed-command gaps (mix suite, job 045): PC samples of v4j (= v4) and v3q on 1-hop 16 B, forwarded
# 16 B (two senders through one router) and EW_MixCmd (write, scatter, fused atomic, atomic).
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
V=fabric_exp/variants
bash $CON/pcsv_runs.sh $V/v4 v4js 2D ew16 fwd16 mixcmd
bash $CON/pcsv_runs.sh $V/v3q v3qs 2D fwd16 mixcmd
