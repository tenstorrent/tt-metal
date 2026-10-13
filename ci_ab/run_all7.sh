#!/usr/bin/env bash
# One P150 job for V7: op-level A/B (run_prof7.sh), bits (run_bits2.sh), and the ROW MAX harness tests with asserts on.
D=$(dirname $0)
bash $D/run_prof7.sh
bash $D/run_bits2.sh
FUNC_LABELS="V7" bash $D/run_func.sh
exit 0
