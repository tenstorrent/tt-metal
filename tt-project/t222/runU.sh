#!/bin/bash
# t222 job U (blx01 broker): the 2-D sharded NA unit test from the overlay.
set -o pipefail
source /var/tmp/fasth3/t219/drv/common.sh
O=$T/u; mkdir -p $O; L=$O/run.log
echo "[t222U] host=$(hostname) $(date -u '+%F %T') UTC" | tee -a $L
cd $OV
timeout ${PY_S:-540} python -m pytest -x -v -p no:cacheprovider models/tt_dit/tests/unit/test_neighborhood_bricked_w_sharded.py -k 2d_sharded 2>&1 | tee -a $L
rc=${PIPESTATUS[0]}
echo "T222_EXIT=$rc" | tee -a $L
exit $rc
