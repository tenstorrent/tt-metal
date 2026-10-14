#!/usr/bin/env bash
# #58817 legacy bmm port, proofs on the candidate head merged with main: bits and ELF identity (the port's cases and
# the earlier rounds' 145 outputs and mock sweep), then the in-process alternation of the head's rule against the tile MOP.
# usage: ci_legf.sh bh|wh [asserts 0|1]
set +e +o pipefail
cd "$(git rev-parse --show-toplevel)"
export CI_FRESH=1 CI_ELFCFG=1 BITS_ASSERTS=${2:-0} ALT_ROUNDS=11 ALT_CALLS=8 ALT_VARIANTS=0,1,4 CI_LIMIT=2400
export ALT_NAMES="1=the head's rule;4=tile MOP, A/A copy"
bash scratch_r3/run_ci.sh scratch_r3/plan_legf_$1.txt
