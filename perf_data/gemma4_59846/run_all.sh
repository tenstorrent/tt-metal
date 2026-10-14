#!/bin/bash
# Resume the global sweep (C=256 prefixes 0/16K/64K/128K already in global_sweep.jsonl), then the sliding sweep.
D=/data/mmanzoor/sdpa_chunk_sweep
T=tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_gemma4_global_chunk_sweep
$D/run_global_sweep.sh \
  --deselect "$T[chunk256-prefix0]" --deselect "$T[chunk256-prefix16384]" \
  --deselect "$T[chunk256-prefix65536]" --deselect "$T[chunk256-prefix131072]" \
  2>&1 | tee -a $D/global_sweep.log
echo "GLOBAL SWEEP EXIT ${PIPESTATUS[0]} $(date)" | tee -a $D/global_sweep.log
$D/run_swa_sweep.sh 2>&1 | tee $D/swa_sweep.log
echo "SWA SWEEP EXIT ${PIPESTATUS[0]} $(date)" | tee -a $D/swa_sweep.log
