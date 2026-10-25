#!/bin/bash
# cmpall.sh A_DIR B_DIR OUT_LIST: textcmp every ELF in A against B; list the differing ones
T=/home/aagarwal/tt-metal/fabric_exp/consolidation/textcmp.sh
A=$1; B=$2; : > $3; same=0; diff=0; miss=0
for e in $A/*/*/*.elf; do rel=${e#$A/}; b=$B/$rel; if [ ! -f $b ]; then miss=$((miss+1)); continue; fi
  if bash $T $e $b; then same=$((same+1)); else diff=$((diff+1)); echo $rel >> $3; fi; done
echo "same=$same diff=$diff missing=$miss"
