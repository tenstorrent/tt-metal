#!/bin/bash
# Sliding-window sweep: 60 items, ~45 min.
source /data/mmanzoor/sdpa_chunk_sweep_c04u02/env.sh
GEMMA4_SDPA_SWEEP_OUT=$D/swa_sweep.jsonl pytest -sv "$T::test_ring_joint_attention_gemma4_swa_chunk_sweep" "$@" 2>&1 | tee -a $D/swa_sweep.log
