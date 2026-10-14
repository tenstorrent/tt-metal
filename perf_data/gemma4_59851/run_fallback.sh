#!/bin/bash
# Experiment: LoFi + fp32 acc on ttnn's default program config at large C (the model's fallback is HiFi2/bf16 acc).
cd /home/mmanzoor/compiler/tt-metal
source activate
D=/data/mmanzoor/matmul_chunk_sweep
until grep -q "FOLLOWUPS DONE" $D/followups.out 2>/dev/null; do sleep 30; done
for i in 1 2; do
  O=$D/fallback$i; rm -rf $O; mkdir -p $O
  python -m tracy -r -p -o $O models/demos/gemma4_d_p/scripts/matmul_chunk_sweep.py --meta $O/meta.jsonl \
    --fallback-lofi --attn-fidelity lofi --tag-prefix fb- --chunks 12288 16384 24576 32768 --reps 10 > $O/run.log 2>&1
  echo "fallback$i exit $? $(date)"
done
python models/demos/gemma4_d_p/scripts/matmul_chunk_sweep_report.py $D/fallback1 $D/fallback2 --out $D/fallback1 > $D/fallback_summary.txt 2>&1
echo "FALLBACK DONE $(date)"
