#!/bin/bash
# Gemma4 sliding-window SDPA chunk sweep (#59846 follow-on). Branch mmanzoor/gemma4-sdpa-small-chunk-sweep.
cd /home/mmanzoor/compiler/tt-metal
source activate
export GEMMA4_SDPA_SWEEP_OUT=${GEMMA4_SDPA_SWEEP_OUT:-/data/mmanzoor/sdpa_chunk_sweep/swa_sweep.jsonl}
export GEMMA4_SDPA_SWEEP_REPEATS=${GEMMA4_SDPA_SWEEP_REPEATS:-6}
pytest -sv "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_gemma4_swa_chunk_sweep" "$@"
