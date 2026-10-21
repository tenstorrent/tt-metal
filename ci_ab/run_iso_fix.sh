#!/usr/bin/env bash
# Scratch job (#58706 COL), not for merge. Which isolate run type, if any, stops a side. Builds nc for F and the side,
# then runs one isolate type (RT) over the COL set with -x -v: F first (control), then the side. Prints the passed count
# and the first failing id for each.
# usage: RT=UNPACK_ISOLATE SIDE=fin bash ci_ab/run_iso_diag.sh <part i/n>
set +e +o pipefail
PART=${1:-0/5}; SIDE=${SIDE:-fin}; RT=${RT:-UNPACK_ISOLATE}; export CI_RED_DIM=Column
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $(git log --oneline -1 | cut -c1-60), $(hostname), $(date -u +%T), part $PART, side $SIDE, rt $RT"
place() { while read -r rel; do [[ -n $rel ]] && cp "$S/v/$1/$rel" "$R/$rel"; done < $S/v/files.txt; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$?"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$?"
T=$R/tt_metal/tt-llk/tests
cp $S/sweep/perf_reduce_allcfg.py $T/python_tests/; cp $S/sweep/reduce_allcfg_perf.cpp $T/sources/
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=400 CI_RED_PART=$PART
cd $T/python_tests
python -m pytest --collect-only -q -p no:cacheprovider perf_reduce_allcfg.py 2>/dev/null | grep "::" > /tmp/ids.txt
echo "== ids $(wc -l < /tmp/ids.txt)"
mapfile -t IDS < /tmp/ids.txt
B=$HOME/iso_b; rm -rf $B; mkdir -p $B
for side in ${SIDES:-F $SIDE}; do
  place $side; d=$B/$side; mkdir -p $d
  RUNNER_TEMP=$d TMPDIR=$d LLK_PERF_RUN_TYPES=$RT timeout -s INT -k 60 2400 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q "${IDS[@]}" > /tmp/b_$side.txt 2>&1
  echo "== build $side rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b_$side.txt | tail -1)"
done
for side in ${SIDES:-F $SIDE}; do
  RUNNER_TEMP=$B/$side TMPDIR=$B/$side LLK_PERF_RUN_TYPES=$RT PERF_RUN_TAG=iso_$side timeout -s INT -k 60 1200 \
    python -m pytest --compile-consumer -x -v -p no:cacheprovider -o timeout_method=thread --timeout 60 "${IDS[@]}" > /tmp/r_$side.txt 2>&1
  echo "== iso $side rc=$? passed $(grep -ac ' PASSED' /tmp/r_$side.txt) of ${#IDS[@]}"
  grep -a -E ' (FAILED|PASSED)' /tmp/r_$side.txt | grep -aE 'Column-Sum-(HiFi[234])-r16f2x2' | cut -c1-200 | sed 's/.*allcfg//'
  grep -a -m3 -E "TIMED OUT|Timeout|PerfSchemaError" /tmp/r_$side.txt | cut -c1-260
done
echo "== DONE $(date -u +%T)"
