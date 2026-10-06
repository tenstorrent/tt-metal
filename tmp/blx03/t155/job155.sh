#!/bin/bash
# #155 runner job on blx03: time one conv3d blocking of exact_s2_res (C_in=C_out=512, k=3, halo) with the
# vol2col_rm guard lifted. The test opens the full 4x8 mesh, then create_submesh(2,4).
# Usage: job155.sh <tag> <blocking> [<tag of the job that must have passed first>]
# Exit 20/21/22: gate refused (build, earlier job, recent tray-2 incident); the device is not opened.
TAG=$1; BLK=$2; NEED=${3:-}
BASE=/home/smarton/fasth3/tt-metal; B=/home/smarton/fasth3/t48; V=/var/tmp/fasth3/t155; S=$V/src
SL=/var/log/tt-device-broker/server.log; LOG=$V/$TAG.log
F=$B/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp
say() { echo "[t155] $(date -u '+%F %T') $*" | tee -a $LOG; }
: > $LOG
say "tag=$TAG blocking=$BLK build=$(git -C $B log -1 --format='%h %s') src=$(cat $S/REV)"
grep -q 'BUILD155_DONE rc=0' $V/build.log && grep -q 'Unaligned blocks get exactly num_patches pages' $F \
  || { say "GATE build: t152 fix not built in $B"; exit 20; }
if [ -n "$NEED" ]; then
  grep -q '1 ok, 0 failed' $V/$NEED.log && grep -q "^T155_PASS" $V/$NEED.log \
    || { say "GATE $NEED did not pass"; exit 21; }
fi
# The tray-2 rule: no job within 30 min of a tray-2 (chips 8-15) incident.
since=$(date -u -d '-30 min' '+%F %T')
t2=$(awk -v s="$since" 'substr($0,1,19) > s' $SL | grep -E 'trays \[[0-9, ]*2[],]|chips 8-15|tray 2\b' | tail -1)
[ -n "$t2" ] && { say "GATE tray-2 incident in the last 30 min: ${t2:0:200}"; exit 22; }
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache
export TT_CONV3D_ALLOW_UNALIGNED_VOL2COL=1 SWEEP_CHECK_ALL=1 SWEEP_ONLY_BLOCKINGS="$BLK" SWEEP_MAX_SECONDS=300 SWEEP_OUT_DIR=$V/results_$TAG
cd $S
timeout 780 python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=760 \
  "models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py::test_bruteforce_sweep_ltx25_544p_145f_halo" \
  -k "exact_s2_res" 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
say "pytest rc=$rc"
python - $V/results_$TAG >> $LOG 2>&1 <<'PY'
import glob, json, sys
for p in glob.glob(sys.argv[1] + "/*.json"):
    d = json.load(open(p))
    print(f"T155_TIMES table={d['table_blocking']} table_us={d['table_us']} results={d['all_results']}")
PY
grep T155_TIMES $LOG
pcc=$(sed -n "s/.*Output check best vs table:.*'pcc': \([0-9.e-]*\).*/\1/p" $LOG | tail -1)
if [ $rc = 0 ] && grep -q '1 ok, 0 failed' $LOG && [ -n "$pcc" ] && python -c "import sys; sys.exit(float('$pcc') < 0.9999)"; then
  echo "T155_PASS pcc=$pcc" | tee -a $LOG
else
  echo "T155_FAIL rc=$rc pcc=${pcc:-none}" | tee -a $LOG; [ $rc = 0 ] && rc=30
fi
exit $rc
