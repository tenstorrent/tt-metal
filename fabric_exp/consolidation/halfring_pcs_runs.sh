#!/bin/bash
# Why completion batching under first-level acks (v4f2) makes half-ring multicast 2-3x slower than v4f1: PC samples.
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
V=fabric_exp/variants
bash $CON/pcsv_runs.sh $V/v4f2 v4f2r Ring halfring ring1
bash $CON/pcsv_runs.sh $V/v4f1 v4f1r Ring halfring ring1
