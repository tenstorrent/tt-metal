#!/usr/bin/env bash
# #58817 legacy bmm port, one CI job: bits and ELF identity of every variant, then the op-level ABBA.
# usage: ci_leg.sh bh|wh [asserts 0|1]
set +e +o pipefail
cd "$(git rev-parse --show-toplevel)"
export CI_FRESH=1 CI_ELFCFG=1 MM_REPS=6 MM_CALLS=16 BITS_ASSERTS=${2:-0} CI_BASES="old main"
bash scratch_r3/run_ci.sh scratch_r3/plan_leg_$1.txt
