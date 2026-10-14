#!/bin/bash
# Sliding sweep (fixed reference scale, production-style cache) and global accuracy of the selected tunings (#59846).
D=/data/mmanzoor/sdpa_chunk_sweep
$D/run_swa_sweep.sh 2>&1 | tee $D/swa_sweep.log
echo "SWA SWEEP EXIT ${PIPESTATUS[0]} $(date)" | tee -a $D/swa_sweep.log
cd /home/mmanzoor/compiler/tt-metal && source activate
pytest -sv "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_gemma4_global_chunk_sweep_accuracy" 2>&1 | tee $D/global_accuracy.log
echo "GLOBAL ACCURACY EXIT ${PIPESTATUS[0]} $(date)" | tee -a $D/global_accuracy.log
