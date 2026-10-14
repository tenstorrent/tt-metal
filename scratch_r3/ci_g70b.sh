#!/usr/bin/env bash
# #58817: the g70b DRAM-sharded FF1/FF3 outputs that differ from main on the 3c202f9944d merge: determinism, a bisect of
# main's #55639 llk_math_matmul.h on each side, and a float64 reference.
set +e +o pipefail
cd "$(git rev-parse --show-toplevel)"
export CI_FRESH=1 CI_ELFCFG=1 BITS_ASSERTS=${1:-0}
bash scratch_r3/run_ci.sh scratch_r3/plan_g70b.txt
python scratch_r3/tools/g70b_cmp.py generated/r3ci
