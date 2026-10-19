#!/bin/bash
# Correctness of the selected global tunings: 18 items, ~35 min.
source /data/mmanzoor/sdpa_chunk_sweep_c04u02/env.sh
pytest -sv "$T::test_ring_joint_attention_gemma4_global_chunk_sweep_accuracy" "$@" 2>&1 | tee -a $D/global_accuracy.log
