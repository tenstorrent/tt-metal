#!/bin/bash
# After run_sweep.sh: (1) real TP-rank-0 weights vs random at model chunks, (2) galaxy layer perf for op shares.
cd /home/mmanzoor/compiler/tt-metal
source activate
D=/data/mmanzoor/matmul_chunk_sweep
O=$D/real; rm -rf $O; mkdir -p $O
python -m tracy -r -p -o $O models/demos/gemma4_d_p/scripts/matmul_chunk_sweep.py --meta $O/meta.jsonl \
  --weights $D/tp0_weights.pt --tag-prefix real- --chunks 1024 2048 4096 8192 16384 --reps 10 > $O/run.log 2>&1
echo "real exit $? $(date)"
python models/demos/gemma4_d_p/scripts/matmul_chunk_sweep_report.py $O --out $O > $O/summary.txt 2>&1
T=models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n
for size in 2048 4096 8192 16384; do
  P=$D/layer_perf/sz$size; rm -rf $P; mkdir -p $P
  PREFILL_SUMMARIES=$D/layer_perf/summaries python -m tracy -p -r -v -o $P --check-exit-code --op-support-count 20000 \
    -m "pytest ${T}[blackhole-chunkci-both-sz${size}-ctx_256k-8x4] -vv --tb=short --timeout=7200" > $P/run.log 2>&1
  echo "layer_perf sz$size exit $? $(date)"
done
echo "FOLLOWUPS DONE $(date)"
