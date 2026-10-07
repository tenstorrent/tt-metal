#!/bin/bash
# #155 broker job on blx01: time one conv3d blocking of exact_s2_res (C_in=C_out=512, k=3, halo) with the vol2col_rm
# guard lifted, on the t152-fixed build. The test opens the full 4x8 mesh, then create_submesh(2,4).
# Usage: run155.sh <tag> <blocking> <inner timeout s>
TAG=$1; BLK=$2; TO=${3:-390}
F=/var/tmp/fasth3; T=$F/t155; B=$T/b; S=$T/src; LOG=$T/$TAG.log
say() { echo "[t155] $(date -u '+%F %T') $*" | tee -a $LOG; }
: > $LOG
say "tag=$TAG blocking=$BLK build=$(git -C $B log -1 --format='%h %s') src=$(cat $S/REV) epoch=$(date +%s)"
grep -q 'blocks get exactly num_patches pages' $B/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp \
  && grep -q 'BUILD155_DONE rc=0' $T/setup.log || { say "GATE build missing"; exit 20; }
source $F/t48/python_env/bin/activate
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
export TT_METAL_HOME=$B TT_METAL_RUNTIME_ROOT=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$T/jit
export TT_CONV3D_ALLOW_UNALIGNED_VOL2COL=1 SWEEP_CHECK_ALL=1 SWEEP_ONLY_BLOCKINGS="$BLK" SWEEP_MAX_SECONDS=200 SWEEP_OUT_DIR=$T/results_$TAG
cd $S
timeout $TO python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=$((TO - 20)) \
  "models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py::test_bruteforce_sweep_ltx25_544p_145f_halo" \
  -k "exact_s2_res" 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
say "pytest rc=$rc"
python - $T/results_$TAG >> $LOG 2>&1 <<'PY'
import glob, json, sys
for p in glob.glob(sys.argv[1] + "/*.json"):
    d = json.load(open(p))
    print(f"T155_TIMES table={d['table_blocking']} table_us={d['table_us']} results={d['all_results']}")
PY
pcc=$(sed -n "s/.*Output check best vs table:.*'pcc': \([0-9.e-]*\).*/\1/p" $LOG | tail -1)
if [ $rc = 0 ] && grep -q '1 ok, 0 failed' $LOG && [ -n "$pcc" ] && python -c "import sys; sys.exit(float('$pcc') < 0.9999)"; then
  echo "T155_PASS pcc=$pcc" | tee -a $LOG
else
  echo "T155_FAIL rc=$rc pcc=${pcc:-none}" | tee -a $LOG; [ $rc = 0 ] && rc=30
fi
echo "T155_EXIT=$rc end_epoch=$(date +%s)" | tee -a $LOG
exit $rc
