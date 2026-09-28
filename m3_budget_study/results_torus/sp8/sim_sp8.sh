#!/bin/bash
# SP=8 4-galaxy layout projection (like batch_sp2_c.sh). L-e: 4-stage pipeline of whole 8x4 SP=8 galaxies;
# L-f: 4 independent galaxies, one 60-layer SP=8 stage each. Tags: hopNN = coeffs_sp8.json (as measured),
# hopNNf = coeffs_sp8_fast_w48.json (fast-mode depth runs, fit on W 4096/8192 only).
S=/home/vmelnykov/tt-metal/m3_budget_study; cd $S; source ../python_env/bin/activate
OUT=results_torus/sp8/sim; mkdir -p $OUT
common="--n 4000 --widths 4096,8192 --policies fcfs,cost --align-recompute --embed-stage0-only --latency --split-search"
budget () { python3 -c "print(round((1500 - ($1 - 1) * $2) / ($1 + 1), 1))"; }
sc () {  # id coeffs stages pipelines hop tag
  local b; b=$(budget $3 $5)
  python3 m3_budget_sim.py --coeffs $2 --stages $3 --pipelines $4 --hop-ms $5 --budget-ms $b $common > $OUT/$1_$6.txt
  echo "$1 $6: stages=$3 pipelines=$4 hop=$5 budget_ms=$b -> $OUT/$1_$6.txt"
}
for c in "results_torus/sp8/coeffs_sp8.json:" "results_torus/sp8/coeffs_sp8_fast_w48.json:f"; do
  C=${c%%:*}; t=${c##*:}
  sc L-e $C 4 1 15 hop15$t &
  sc L-e $C 4 1 30 hop30$t &
  sc L-f $C 1 4 0 hop0$t &
  wait
done
python3 sim_summary.py $OUT > $OUT/summary.txt
