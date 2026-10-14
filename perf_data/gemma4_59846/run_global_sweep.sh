#!/bin/bash
# Gemma4 global SDPA small-chunk sweep (#59846). Branch mmanzoor/gemma4-sdpa-small-chunk-sweep.
cd /home/mmanzoor/compiler/tt-metal
source activate
export GEMMA4_SDPA_SWEEP_OUT=${GEMMA4_SDPA_SWEEP_OUT:-/data/mmanzoor/sdpa_chunk_sweep/global_sweep.jsonl}
export GEMMA4_SDPA_SWEEP_REPEATS=${GEMMA4_SDPA_SWEEP_REPEATS:-6}
pytest -sv "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_gemma4_global_chunk_sweep" "$@"
