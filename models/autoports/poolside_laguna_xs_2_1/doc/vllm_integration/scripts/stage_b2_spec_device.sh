#!/bin/bash
# Stage B2 — on-device ngram spec-decode: correctness (token-identical to plain greedy) + measured speedup
# at long context. Config from B1 (host replay, STRONG 2.5x projected): min_n=1 (NgramProposer default),
# K=16, max_n=10, verify_mode=prefill (the k128 verify path). prompt-mode=code = realistic code-copying
# proxy for the agent workload (B1's 2.5x was on the real .traj.json trajectories).
#
# The driver's --baseline is greedy-VIA-VERIFY (same verify path -> isolates drafting efficiency). The
# production-relevant reference is the fast TRACED decode t/s/u; take that from the Stage D served sweep at
# matching ISL (32768) and report the cross-harness production speedup with that caveat.
set +e
cd /tmp
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PYTHONPATH=/home/ttuser/dev/tt-metal
export TT_LAGUNA_WEIGHT_CACHE_DISABLE=1
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
LOG=$BASE/doc/vllm_integration/spec_decode_accept/b2_device.log   # <-- TAIL THIS
PY=/home/ttuser/.tenstorrent-venv/bin/python
$PY -u -m models.autoports.poolside_laguna_xs_2_1.doc.vllm_integration.scripts.spec_decode_driver \
  --mode both --isl-acc 4096 --osl-acc 64 --isl 32768 --osl 128 \
  --draft-len 16 --ngram-max-n 10 --prompt-mode code --baseline \
  --log "$LOG"
echo "=== B2 DONE ===" | tee -a "$LOG"
grep -E "RESULT|ACCURACY|speedup|mean_accept" "$LOG" | tail -25
tt-smi -r all >/dev/null 2>&1; sleep 8
