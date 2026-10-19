#!/bin/bash
# PR #59484's production perf check (global + sliding, C=2048/4096/8192, final 256K chunk). Compare with the old box.
source /data/mmanzoor/sdpa_chunk_sweep_c04u02/env.sh
pytest -sv "$T::test_ring_joint_attention_gemma4_chunked_perf_check" "$@" 2>&1 | tee $D/pr_perf_check.log
