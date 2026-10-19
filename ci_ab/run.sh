#!/usr/bin/env bash
# Scratch job (tilize all-configuration sweep, not for merge). tt-llk harness on one CI Blackhole runner (17_CI_AB.md).
# Tree: the PRs' base 129d0c50379 (main, M) plus ci_ab/. Sides: M = base files, F = #59143 head d7436a953de, B = #60672 head
# 0c5727239c3, M2 = M again in its own build (A/A). place <side> copies that side's version of every file either head
# changes (ci_ab/files.txt; v/<side>.absent removed) into /work, then the sweep modules. Each side builds once into its own
# fresh dir (NC: every run type of the module; WC: L1_TO_L1 with --dump-perf-counters). Guard: elfguard.py (M2 == M on
# every variant, each head side differs somewhere), else the job stops. NC L1_TO_L1 runs ABBA: RUN forward then backward,
# twice; then the isolates and the counter runs once per side in ISO. Results: tar.gz base64 blocks in the log.
# usage: run.sh <ids file> "<RUN sides>" "<ISO sides>" "<guard-only sides>"
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
IDS=$S/$1; RUN=$2; ISO=$3; GO=$4
echo "== checkout $R, $(git log --oneline -1 | cut -c1-80), $(hostname), $(date -u +%T), ids $(wc -l < $IDS)"
src() { case $1 in M2) echo M ;; *) echo $1 ;; esac; }
place() {
  local s=$(src $1)
  while read -r f; do [[ -n $f ]] && rm -f "$R/$f"; done < $S/v/$s.absent
  (cd $S/v/$s && find . -type f | while read -r f; do mkdir -p "$R/$(dirname $f)"; cp "$f" "$R/$f"; done)
  cp $S/sweep/*.py $R/tt_metal/tt-llk/tests/python_tests/; cp $S/sweep/*.cpp $R/tt_metal/tt-llk/tests/sources/
}
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -1 /tmp/setup.txt)"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=400
cd $R/tt_metal/tt-llk/tests/python_tests
mapfile -t NODES < $IDS
NP=$(nproc); NJ=$(( NP > 16 ? 16 : NP ))
PY=(python -m pytest -p no:cacheprovider -q -o timeout_method=thread --timeout 300)
EC=""; WC=""
for v in $RUN $GO; do
  place $v; rm -rf $HOME/tz_$v; rts=L1_TO_L1; [[ " $ISO " == *" $v "* ]] && rts=""
  LLK_PERF_RUN_TYPES=$rts RUNNER_TEMP=$HOME/tz_$v TMPDIR=$HOME/tz_$v timeout -s INT -k 60 3000 "${PY[@]}" --compile-producer -n $NJ "${NODES[@]}" > /tmp/b_$v.txt 2>&1
  echo "== build $v rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/b_$v.txt | tail -1)"
  grep -a -E "^(FAILED|ERROR)" /tmp/b_$v.txt | cut -c1-200 | head -5
  EC="$EC $v=$HOME/tz_$v"
done
python $S/elfguard.py $EC --must $(echo $RUN | tr " " "\n" | grep -v -E "^M2?$" | paste -sd,) --out /tmp/elf.txt | tee /tmp/guard.txt
cut -d' ' -f2- /tmp/elf.txt | sort | uniq -c | sort -rn | head -12
grep -q "^GUARD ok" /tmp/guard.txt || { echo "== GUARD FAIL: the sides did not build as expected"; exit 1; }
for v in $ISO; do
  place $v; rm -rf $HOME/tw_$v
  LLK_PERF_RUN_TYPES=L1_TO_L1 RUNNER_TEMP=$HOME/tw_$v TMPDIR=$HOME/tw_$v timeout -s INT -k 60 3000 "${PY[@]}" --compile-producer --dump-perf-counters -n $NJ "${NODES[@]}" > /tmp/w_$v.txt 2>&1
  echo "== wc build $v rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/w_$v.txt | tail -1)"
done
runc() {  # runc <side> <tag> <run types> [extra]
  local v=$1 tag=$2 rts=$3; shift 3
  LLK_PERF_RUN_TYPES=$rts RUNNER_TEMP=$HOME/tz_$v TMPDIR=$HOME/tz_$v PERF_RUN_TAG=$tag timeout -s INT -k 60 3000 \
    "${PY[@]}" --compile-consumer "$@" "${NODES[@]}" > /tmp/r_$tag.txt 2>&1
  echo "== run $tag rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/r_$tag.txt | tail -1)"
  grep -a -E "^(FAILED|ERROR)" /tmp/r_$tag.txt | cut -c1-160 | head -3
}
RV=$(echo $RUN | tr ' ' '\n' | tac | tr '\n' ' ')
j=0
for blk in 1 2; do
  for vs in "$RUN" "$RV"; do
    for v in $vs; do j=$((j + 1)); place $v; runc $v nc_${j}_$v L1_TO_L1; done
  done
done
for v in $ISO; do place $v; runc $v iso_$v UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE; done
for v in $ISO; do
  place $v
  LLK_PERF_RUN_TYPES=L1_TO_L1 RUNNER_TEMP=$HOME/tw_$v TMPDIR=$HOME/tw_$v PERF_RUN_TAG=wc_$v timeout -s INT -k 60 3000 \
    "${PY[@]}" --compile-consumer --dump-perf-counters "${NODES[@]}" > /tmp/r_wc_$v.txt 2>&1
  echo "== run wc_$v rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/r_wc_$v.txt | tail -1)"
done
cd $LLK_HOME/perf_data/runs && tar czf /tmp/res.tgz nc_* iso_* wc_* 2>/dev/null
cp /tmp/elf.txt /tmp/res_elf.txt; (cd $HOME/tz_M/tt-llk-build/sources && tar czf /tmp/bh.tgz */*/build.h); tar czf /tmp/elf.tgz -C /tmp res_elf.txt guard.txt bh.tgz
for f in res elf; do echo "== b64 $f begin $(stat -c %s /tmp/$f.tgz)"; base64 -w 0 /tmp/$f.tgz; echo; echo "== b64 $f end"; done
echo "== DONE $(date -u +%T)"
