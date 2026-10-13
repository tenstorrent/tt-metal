#!/usr/bin/env bash
# DEST hand-off job 1 (knowledge/process/17_CI_AB.md): one P150, tt-llk harness, the nodes of ci_ab/nodes.txt.
# Variants (ci_ab/v/<V>, files of ci_ab/v/files.txt placed at their tree paths before each build and run): B the
# combo tree's hand-off, B2 the same code in its own build (A/A), M1 math clears at acquire and the pack release
# drops ZEROACC, M0 no clear at all (probe). NC builds for B B2 M1 M0, WC (counter) builds for B M1 M0.
# NC runs: two blocks, each forward then backward through B M1 M0 B2 (four runs per variant); WC: one run each.
# Guard: the job stops unless B2 builds B's ELFs, M1 differs from B in math and pack, M0 differs in pack.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $R, $(git log --oneline -1), $(date -u +%T)"
src() { case $1 in B2) echo B ;; *) echo $1 ;; esac; }
place() { while read -r rel; do [[ -n $rel ]] && cp "$S/v/$1/$rel" "$R/$rel"; done < $S/v/files.txt; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.txt | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1 PERF_KEEP_RUNS=400
cd $R/tt_metal/tt-llk/tests/python_tests
mapfile -t NODES < $S/nodes.txt
echo "== nodes ${#NODES[@]}"
build() { # $1 variant $2 dir $3 extra flag
  place $(src $1); rm -rf $2
  RUNNER_TEMP=$2 TMPDIR=$2 timeout -s INT -k 60 1200 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q $3 "${NODES[@]}" > /tmp/b_$1$3.txt 2>&1
  echo "== build $1 $3 rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/b_$1$3.txt | tail -1)"
}
for v in B B2 M1 M0; do build $v $HOME/dh_$v ""; done
for v in B M1 M0; do build $v $HOME/dhw_$v --dump-perf-counters; done
python $S/tools/elfcheck.py B=$HOME/dh_B B2=$HOME/dh_B2 M1=$HOME/dh_M1 M0=$HOME/dh_M0 | tee /tmp/elf.txt | grep ELFSUM
python - /tmp/elf.txt <<'PY' || { echo "== GUARD FAIL"; exit 1; }
import re, sys
s = {m.group(1): tuple(map(int, m.group(2, 3, 4, 5, 6))) for m in (re.match(r"ELFSUM (\S+) dirs=(\d+) common=(\d+) differ_from_first unpack=(\d+) math=(\d+) pack=(\d+)", l) for l in open(sys.argv[1])) if m}
n = s["B"][0]
ok = n >= 10 and s["B2"][2:] == (0, 0, 0) and s["M1"][3] == n and s["M1"][4] == n and s["M0"][4] == n and s["M0"][3] == 0
print("== GUARD", "ok" if ok else "FAIL", s); sys.exit(0 if ok else 1)
PY
run() { # $1 variant $2 tag $3 dir $4 extra flag
  place $(src $1)
  RUNNER_TEMP=$3 TMPDIR=$3 PERF_RUN_TAG=$2 timeout -s INT -k 60 1200 python -m pytest --compile-consumer -p no:cacheprovider -q -rfE $4 "${NODES[@]}" > /tmp/r_$2.txt 2>&1
  echo "== run $1 $2 rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$2.txt | tail -1)"
  grep -a -m3 -E "^FAILED|^ERROR" /tmp/r_$2.txt | cut -c1-240
  for f in $(ls $LLK_HOME/perf_data/runs/$2/*/*.csv 2>/dev/null | grep -v -e post -e counters); do echo "== csv $2 $(basename $f)"; cat "$f"; echo "== end csv"; done
  for f in $(ls $LLK_HOME/perf_data/runs/$2/*/*.counters.csv 2>/dev/null); do echo "== ccsv $2 $(basename $f)"; python $S/tools/cfilter.py "$f"; echo "== end ccsv"; done
}
VS="B M1 M0 B2"; RV="B2 M0 M1 B"; j=0
for blk in 1 2; do for v in $VS $RV; do j=$((j + 1)); run $v dh_${j}_$v $HOME/dh_$v ""; done; done
for v in B M1 M0; do j=$((j + 1)); run $v dhw_${j}_$v $HOME/dhw_$v --dump-perf-counters; done
cd $R; place B
echo "== DONE $(date -u +%T)"
