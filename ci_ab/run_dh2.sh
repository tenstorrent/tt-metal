#!/usr/bin/env bash
# DEST hand-off job 2: as job 1 (run_dh1.sh) with M1b (math clears at acquire without the STALLWAIT), the eltwise add row
# in its own pytest call (job 1 lost its CSV row), and the math-bound controls. Variants B B2 M1 M1b M0; WC B M1 M1b.
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
mapfile -t N1 < $S/nodes2.txt
mapfile -t N2 < $S/nodes_add.txt
build() { place $(src $1); rm -rf $2
  for g in 1 2; do
    if [[ $g == 1 ]]; then NN=("${N1[@]}"); else NN=("${N2[@]}"); fi
    RUNNER_TEMP=$2/g$g TMPDIR=$2/g$g timeout -s INT -k 60 1200 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q $3 "${NN[@]}" > /tmp/b_$1$3_$g.txt 2>&1
    echo "== build $1 $3 g$g rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/b_$1$3_$g.txt | tail -1)"
  done; }
for v in B B2 M1 M1b M0; do build $v $HOME/dh_$v ""; done
for v in B M1 M1b; do build $v $HOME/dhw_$v --dump-perf-counters; done
python $S/tools/elfcheck.py B=$HOME/dh_B/g1 B2=$HOME/dh_B2/g1 M1=$HOME/dh_M1/g1 M1b=$HOME/dh_M1b/g1 M0=$HOME/dh_M0/g1 | grep ELFSUM
python $S/tools/elfcheck.py B=$HOME/dh_B/g2 B2=$HOME/dh_B2/g2 M1=$HOME/dh_M1/g2 M1b=$HOME/dh_M1b/g2 M0=$HOME/dh_M0/g2 | grep ELFSUM | tee /tmp/elf2.txt
python - /tmp/elf2.txt <<'PY' || { echo "== GUARD FAIL"; exit 1; }
import re, sys
s = {m.group(1): tuple(map(int, m.group(2, 3, 4, 5, 6))) for m in (re.match(r"ELFSUM (\S+) dirs=(\d+) common=(\d+) differ_from_first unpack=(\d+) math=(\d+) pack=(\d+)", l) for l in open(sys.argv[1])) if m}
ok = s["B"][0] >= 1 and s["B2"][2:] == (0, 0, 0) and s["M1"][3] >= 1 and s["M1b"][3] >= 1 and s["M0"][4] >= 1
print("== GUARD", "ok" if ok else "FAIL", s); sys.exit(0 if ok else 1)
PY
run() { place $(src $1)
  for g in 1 2; do
    if [[ $g == 1 ]]; then NN=("${N1[@]}"); else NN=("${N2[@]}"); fi
    RUNNER_TEMP=$3/g$g TMPDIR=$3/g$g PERF_RUN_TAG=$2_g$g timeout -s INT -k 60 1200 python -m pytest --compile-consumer -p no:cacheprovider -q -rfE $4 "${NN[@]}" > /tmp/r_$2_$g.txt 2>&1
    echo "== run $1 $2 g$g rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$2_$g.txt | tail -1)"
    grep -a -m3 -E "^FAILED|^ERROR" /tmp/r_$2_$g.txt | cut -c1-240
    for f in $(find $LLK_HOME/perf_data/runs/$2_g$g -name "*.csv" 2>/dev/null | grep -v -e post -e counters); do echo "== csv $2 $(basename $f)"; cat "$f"; echo "== end csv"; done
    for f in $(find $LLK_HOME/perf_data/runs/$2_g$g -name "*.counters.csv" 2>/dev/null); do echo "== ccsv $2 $(basename $f)"; python $S/tools/cfilter.py "$f"; echo "== end ccsv"; done
  done; }
VS="B M1 M1b M0 B2"; RV="B2 M0 M1b M1 B"; j=0
for blk in 1 2; do for v in $VS $RV; do j=$((j + 1)); run $v dh_${j}_$v $HOME/dh_$v ""; done; done
for v in B M1 M1b; do j=$((j + 1)); run $v dhw_${j}_$v $HOME/dhw_$v --dump-perf-counters; done
cd $R; place B
echo "== DONE $(date -u +%T)"
