#!/usr/bin/env bash
# Scratch job (fast untilize BFP input arms), not for merge. tt-llk harness on one CI Blackhole runner (17_CI_AB.md).
# Sides (ci_ab/sides/<S>.tar.gz): M = origin/main, H1 = #59149 head, D1..D4 = H1 with unpack diagnostics (wrong data, timing only), all carrying
# the harness edits (production pack_src for BFP input, dvalid count macro); M2 = M again (A/A).
# Each side builds once into its own fresh dir. Timing: forward M H1 D1 D2 D3 D4 M2, then backward (ABBA). Guard: elfcheck_fu.py.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $R, $(git log --oneline -1 | cut -c1-80), $(hostname), $(nproc) cpus, $(date -u +%T)"
L=$R/tt_metal/tt-llk; DIRS="tt-llk/tt_llk_blackhole tt-llk/common tt-llk/tests/sources tt-llk/tests/helpers tt-llk/tests/python_tests hw/ckernels/blackhole/metal/llk_api hw/inc/internal/tt-1xx/blackhole"
place() { local x; for x in $DIRS; do rm -rf "$R/tt_metal/$x"; done; tar -xzf $S/sides/$1.tar.gz -C $R/tt_metal; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -1 /tmp/setup.txt)"
export CHIP_ARCH=blackhole LLK_HOME=$L LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=400
MODS="perf_fast_untilize_allcfg.py"
NP=$(( $(nproc) > 16 ? 16 : $(nproc) ))
src() { [[ $1 == M2 ]] && echo M || echo $1; }
pick() { case $1 in M|H1) echo pick_A.txt;; *) echo pick_B.txt;; esac; }
rts() { case $1 in M|H1|M2) echo L1_TO_L1,PACK_ISOLATE,UNPACK_ISOLATE;; D1) echo L1_TO_L1,UNPACK_ISOLATE;; *) echo UNPACK_ISOLATE;; esac; }
build() {  # side dir
  local s=$1 d=$2; place $(src $s); rm -rf $d; mkdir -p $d
  cd $L/tests/python_tests
  UT_PICK=$S/$(pick $s) UT_RUN_TYPES=$(rts $s) RUNNER_TEMP=$d TMPDIR=$d timeout -s INT -k 60 2400 python -m pytest --compile-producer -n $NP -p no:cacheprovider -q $MODS > /tmp/b_$s.txt 2>&1
  echo "== build $s rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b_$s.txt | tail -1)"
  grep -a -m3 "error:" /tmp/b_$s.txt | cut -c1-240; grep -qaE '[0-9]+ passed' /tmp/b_$s.txt || tail -5 /tmp/b_$s.txt | cut -c1-240; cd $R
}
run() {  # side dir tag
  local s=$1 d=$2 tag=$3; place $(src $s)
  cd $L/tests/python_tests
  UT_PICK=$S/$(pick $s) UT_RUN_TYPES=$(rts $s) RUNNER_TEMP=$d TMPDIR=$d PERF_RUN_TAG=$tag timeout -s INT -k 60 1200 \
    python -m pytest --compile-consumer -p no:cacheprovider -q -rfE $MODS > /tmp/r_$tag.txt 2>&1
  echo "== run $tag rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1)"
  grep -a -m4 -E "^FAILED|TIMED OUT|Timeout" /tmp/r_$tag.txt | cut -c1-240
  grep -qaE '[0-9]+ passed' /tmp/r_$tag.txt || tail -5 /tmp/r_$tag.txt | cut -c1-240; cd $R
  python $S/ut_dump3.py $tag $s $L/perf_data/runs/$tag
}
ARMS=${ARMS:-"M H1 D1 D2 D3 D4"}
for s in $ARMS M2; do build $s $HOME/ut_$s; done
python $S/elfcheck_fu.py M=$HOME/ut_M M2=$HOME/ut_M2 $(for a in $ARMS; do [[ $a != M ]] && echo $a=$HOME/ut_$a; done) | tee /tmp/elf.txt
grep -q "^GUARD ok" /tmp/elf.txt || { echo "== GUARD FAIL: the sides did not build as expected"; exit 1; }
j=0
FWD="$ARMS M2"; BWD=$(echo $FWD | tr ' ' '\n' | tac | tr '\n' ' ')
for s in $FWD $BWD; do j=$((j + 1)); run $s $HOME/ut_$s ut_${j}_$s; done
place M
echo "== DONE $(date -u +%T)"
