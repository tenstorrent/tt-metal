#!/usr/bin/env bash
# Scratch job (untilize allcfg sweep), not for merge. tt-llk harness on one CI Blackhole runner (17_CI_AB.md).
# Sides (ci_ab/sides/<S>.tar.gz, the tt-llk dirs of each tree with the allcfg modules, placed at the checkout's paths before
# every build and run): M = origin/main, H1 = #59149 head, H2 = #60646 head, M2 = M again (A/A, a sample of the slice).
# Each side builds once into its own fresh dir. Timing: forward M H1 H2 M2, then backward M2 H2 H1 M (ABBA).
# Counters (CTR=1): one L1_TO_L1 counter build and run per side M, H1, H2. Guard: ELF identity (elfcheck.py) or stop.
# Env: UT_SLICE=k/n (case slice), AA_N (A/A keeps every AA_N-th case of the slice), UT_PICK (case id file), CTR.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $R, $(git log --oneline -1 | cut -c1-80), $(hostname), $(nproc) cpus, $(date -u +%T)"
L=$R/tt_metal/tt-llk; DIRS="tt_llk_blackhole common tests/sources tests/helpers tests/python_tests"
place() { for d in $DIRS; do rm -rf "$L/$d"; done; tar -xzf $S/sides/$1.tar.gz -C $L; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -1 /tmp/setup.txt)"
export CHIP_ARCH=blackhole LLK_HOME=$L LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=400
export UT_RUN_TYPES=L1_TO_L1,PACK_ISOLATE
[[ -n ${UT_PICK:-} ]] && export UT_PICK=$S/$UT_PICK
K=${UT_SLICE%/*}; N=${UT_SLICE#*/}; AA_N=${AA_N:-8}
MODS="perf_untilize_allcfg.py perf_fast_untilize_allcfg.py"
NP=$(( $(nproc) > 16 ? 16 : $(nproc) ))
src() { [[ $1 == M2 ]] && echo M || echo $1; }
slice() { [[ $1 == M2 ]] && echo "$K/$((N * AA_N))" || echo "$UT_SLICE"; }
build() {  # side dir flags...
  local s=$1 d=$2; shift 2; place $(src $s); rm -rf $d; mkdir -p $d
  RUNNER_TEMP=$d TMPDIR=$d UT_SLICE=$(slice $s) timeout -s INT -k 60 2400 python -m pytest --compile-producer -n $NP -p no:cacheprovider -q "$@" $MODS > /tmp/b_$s.txt 2>&1
  echo "== build $s rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b_$s.txt | tail -1)"
  grep -a -m3 "error:" /tmp/b_$s.txt | cut -c1-240
}
run() {  # kind side dir tag flags...
  local k=$1 s=$2 d=$3 tag=$4; shift 4; place $(src $s)
  RUNNER_TEMP=$d TMPDIR=$d UT_SLICE=$(slice $s) PERF_RUN_TAG=$tag timeout -s INT -k 60 3000 \
    python -m pytest --compile-consumer -p no:cacheprovider -q -rfE "$@" $MODS > /tmp/r_$tag.txt 2>&1
  echo "== run $tag rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1)"
  grep -a -m4 -E "^FAILED|TIMED OUT|Timeout" /tmp/r_$tag.txt | cut -c1-240
  python $S/ut_dump.py $k $tag $s $L/perf_data/runs/$tag
}
for s in M H1 H2 M2; do build $s $HOME/ut_$s; done
python $S/elfcheck.py M=$HOME/ut_M M2=$HOME/ut_M2 H1=$HOME/ut_H1 H2=$HOME/ut_H2 | tee /tmp/elf.txt
grep -q "^GUARD ok" /tmp/elf.txt || { echo "== GUARD FAIL: the sides did not build as expected"; exit 1; }
j=0
for s in M H1 H2 M2 M2 H2 H1 M; do j=$((j + 1)); run T $s $HOME/ut_$s ut_${j}_$s; done
if [[ ${CTR:-0} == 1 ]]; then
  export UT_RUN_TYPES=L1_TO_L1
  for s in M H1 H2; do build $s $HOME/utc_$s --dump-perf-counters; done
  python $S/elfcheck.py M=$HOME/utc_M H1=$HOME/utc_H1 H2=$HOME/utc_H2 | grep -E "GUARD|vs"
  for s in M H1 H2; do run C $s $HOME/utc_$s utc_$s --dump-perf-counters; done
fi
place M
echo "== DONE $(date -u +%T)"
