#!/bin/bash
# SP2 follow-up, Part C: 4-galaxy layout projection. Usage: batch_sp2_c.sh <hop_sp4_ms> <hop_sp2_ms>
cd "$(dirname "$0")"; OUT=results_sp2/sim_layouts; mkdir -p $OUT
H4=${1:?hop sp4}; H2=${2:?hop sp2}
C4=results/coeffs.json; C2=results_sp2/coeffs_sp2.json
common="--n 4000 --widths 4096,8192 --policies fcfs,cost --align-recompute --embed-stage0-only --latency --split-search"
budget () { python3 -c "print(round((1500 - ($1 - 1) * $2) / ($1 + 1), 1))"; }
sc () {  # id coeffs stages pipelines hop tag
  local b; b=$(budget $3 $5)
  python3 m3_budget_sim.py --coeffs $2 --stages $3 --pipelines $4 --hop-ms $5 --budget-ms $b $common > $OUT/$1_$6.txt
  echo "$1 $6: stages=$3 pipelines=$4 hop=$5 budget_ms=$b -> $OUT/$1_$6.txt"
}
for pass in measured hop15; do
  h4=$H4; h2=$H2; [ $pass = hop15 ] && h4=15 && h2=15
  sc L-a $C4 8 1 $h4 $pass
  sc L-b $C2 4 4 $h2 $pass
  sc L-c $C2 16 1 $h2 $pass
  sc L-d $C4 2 4 $h4 $pass
done
