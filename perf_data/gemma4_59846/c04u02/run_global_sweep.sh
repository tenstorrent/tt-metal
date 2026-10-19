#!/bin/bash
# Global attention sweep: 25 (C, prefix) items, ~4 min each.
source /data/mmanzoor/sdpa_chunk_sweep_c04u02/env.sh
GEMMA4_SDPA_SWEEP_OUT=$D/global_sweep.jsonl pytest -sv "$T::test_ring_joint_attention_gemma4_global_chunk_sweep" "$@" 2>&1 | tee -a $D/global_sweep.log
