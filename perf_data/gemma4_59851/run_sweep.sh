#!/bin/bash
# Full #59851 sweep: 3 independent profiler runs of every projection x chunk; attention at prod/LoFi/HiFi2.
cd /home/mmanzoor/compiler/tt-metal
source activate
D=/data/mmanzoor/matmul_chunk_sweep
for i in 1 2 3; do
  O=$D/run$i; rm -rf $O; mkdir -p $O
  python -m tracy -r -p -o $O models/demos/gemma4_d_p/scripts/matmul_chunk_sweep.py \
    --meta $O/meta.jsonl --attn-fidelity prod lofi hifi2 --reps 10 > $O/run.log 2>&1
  echo "run$i exit $? $(date)"
done
python models/demos/gemma4_d_p/scripts/matmul_chunk_sweep_report.py $D/run1 $D/run2 $D/run3 --out $D > $D/summary.txt 2>&1
echo "SWEEP DONE $(date)"
