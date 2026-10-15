#!/bin/bash
# Usage: run_layer.sh <tag> [chunk_idx=64]  -> tracy run of one global layer at 2K, ops CSV under $W/<tag>
set -o pipefail
W=/data/mmanzoor/global_sdpa_gap_c04u02
TAG=$1; CHUNK=${2:-64}
cd /home/mmanzoor/compiler/tt-metal && source activate
python -m tracy -r -p -v -o $W/$TAG -m pytest -sv \
  "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk${CHUNK}-global-sz2048-ctx_256k-8x4]" \
  > $W/$TAG.log 2>&1
echo "exit $?" >> $W/$TAG.log
REP=$(ls -td $W/$TAG/reports/*/ 2>/dev/null | head -1)
[ -n "$REP" ] && python3 -I $W/layer_ops.py $REP > $W/$TAG.ops.txt 2>&1
tail -3 $W/$TAG.log
