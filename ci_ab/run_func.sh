#!/usr/bin/env bash
# #59149 regression audit: functional tt-llk tests with LLK asserts on, on one CI Blackhole machine (17_CI_AB.md), for the
# head this branch carries: every pack untilize test id and the fast untilize ids whose math loop changed.
# usage: bash ci_ab/run_func.sh
set +e +o pipefail
R=${LLKAB_ROOT:-/work}; S=$R/ci_ab; T=${LLKAB_TMP:-/tmp/llkab}; rm -rf $T; mkdir -p $T
unset RUNNER_TEMP TT_LLK_DISABLE_ASSERTS
export CHIP_ARCH=blackhole
echo "== start $(date -u +%T) $(git -C $R rev-parse HEAD 2>/dev/null)"

python3 -m venv $T/venv > $T/venv.log 2>&1
$T/venv/bin/pip install -q --disable-pip-version-check -r $R/tt_metal/tt-llk/tests/requirements.txt > $T/pip.log 2>&1
echo "== pip rc=$? $(tail -n 2 $T/pip.log | tr '\n' ' ')"
PY=$T/venv/bin/python

want=$(sed -n "s/^sfpi_version='\(.*\)'/\1/p" $R/tt_metal/tt-llk/tests/sfpi-version)
SF=""
for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do
  if [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]]; then
    have=$(cat $c/sfpi.version 2>/dev/null)
    echo "== sfpi candidate $c version '$have' (want $want)"
    if [[ -z $SF && ( -z $have || $have == "$want" ) ]]; then SF=$c; fi
  fi
done
if [[ -z $SF ]]; then
  (cd $R/tt_metal/tt-llk/tests && bash ./setup_testing_env.sh > $T/sfpi.log 2>&1); echo "== sfpi download rc=$? $(tail -n 1 $T/sfpi.log)"
  SF=$R/tt_metal/tt-llk/tests/sfpi
else
  rm -rf $R/tt_metal/tt-llk/tests/sfpi; ln -s $SF $R/tt_metal/tt-llk/tests/sfpi
fi
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version 2>&1 | sed -n 1p)"

cat > $T/pyrun.py <<'PYEOF'
import sys, pytest
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
sys.exit(pytest.main(["-p", "no:cacheprovider", "--override-ini=log_cli=false", "-q", *sys.argv[2:], *ids]))
PYEOF

harness() {  # harness <build> <ids> <log> [pytest args...]
  local b=$1 ids=$2 log=$3; shift 3
  mkdir -p $T/b_$b
  (cd $R/tt_metal/tt-llk/tests/python_tests && \
   LLK_HOME=$R/tt_metal/tt-llk TMPDIR=$T/b_$b timeout -s INT -k 60 ${LIM:-2400} $PY $T/pyrun.py $ids "$@") > $log 2>&1
}

for set in pu fu; do
  harness $set $S/ids_func_$set.txt $T/prod_$set.log --compile-producer -n ${LLKAB_NPROD:-16}
  echo "== build $set rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/prod_$set.log | tail -n 1)"
  harness $set $S/ids_func_$set.txt $T/cons_$set.log --compile-consumer -o timeout_method=thread --timeout 120
  echo "== functional $set, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_$set.log | tail -n 1)"
  grep -aE '^(FAILED|ERROR) ' $T/cons_$set.log | cut -c1-400 | sed -n 1,40p
done
echo "== end $(date -u +%T)"
