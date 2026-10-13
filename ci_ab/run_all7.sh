#!/usr/bin/env bash
# One job on this merge for V7 (this head) against H (the previous head's row max): op-level A/B, bits, the harness A/B
# with counters, and the ROW MAX harness tests with asserts on.
D=$(dirname $0)
bash $D/run_prof7.sh
bash $D/run_bits2.sh
bash $D/run_llk_ab.sh
[ -n "${SKIP_FUNC:-}" ] || FUNC_LABELS="V7" bash $D/run_func.sh
exit 0
