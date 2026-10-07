#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): exhaustive HiFi2-rule bit dump, HiFi4 reference first, then HiFi2, then detail.
cd /work
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
export EB_DUMP_DIR=/tmp/eb_hifi2_dump EB_R3_LOG_RULE=1
T=tests/eb_r3_ci/test_eb_hifi2_dump.py
F='/^DUMP|EB_R3_RULE|passed|failed|skipped|rror/ && !seen[$0]++'
echo "##### ref (HiFi4) $(date -u +%T)"
EB_DUMP_STAGE=ref EB_R3_NO_HIFI2=1 TT_METAL_CACHE=/tmp/ebd_h4 python3 -m pytest -p no:cacheprovider -q -s $T 2>&1 | awk "$F"
echo "##### cmp (HiFi2 rule) $(date -u +%T)"
EB_DUMP_STAGE=cmp TT_METAL_CACHE=/tmp/ebd_h2 python3 -m pytest -p no:cacheprovider -q -s $T 2>&1 | awk "$F"
echo "##### detail (HiFi4) $(date -u +%T)"
EB_DUMP_STAGE=detail EB_R3_NO_HIFI2=1 TT_METAL_CACHE=/tmp/ebd_h4 python3 -m pytest -p no:cacheprovider -q -s $T -k scalar_lhs 2>&1 | awk "$F"
echo "##### end $(date -u +%T)"; du -sh /tmp/eb_hifi2_dump
