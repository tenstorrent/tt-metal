#!/usr/bin/env bash
# The partial-scaler MAX row check on H and V1 (ttnn), then the functional ROW MAX harness tests with asserts on (V3).
source $(dirname $0)/common.sh
for side in H V1; do use $side; timeout -s INT -k 30 900 python3 $S/tools/partial.py $side 2>&1 | grep -E "^PARTIAL|Error|TT_THROW" | head -12; done
use H
FUNC_LABELS=${FUNC_LABELS:-V3} bash $S/run_func.sh
exit 0
