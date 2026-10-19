#!/usr/bin/env bash
# Scratch job (#61433 scalar reduce, #58820 head 8522fd08bab), not for merge. All-configuration reduce sweep restricted
# by CI_RED_DIM (default Scalar), sides $SIDES (ci_ab/v/<side>/ files placed at their tree paths before each build).
# Each side builds once per form in a fresh dir: nc (four run types) and wc (L1_TO_L1 with counters). L1_TO_L1 in the
# order SIDES then reversed, twice; isolates and counters once a side. Guard: elfguard.py (MUSTDIFF pairs must differ).
# usage: run_sc_sweep.sh <part i/n>
set +e +o pipefail
PART=${1:-0/1}; SIDES=${SIDES:-main head B1 B3}; export CI_RED_DIM=${CI_RED_DIM:-Scalar} MUSTDIFF=${MUSTDIFF:-"main:head head:B1 B1:B3"}
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $(git log --oneline -1 | cut -c1-60), $(hostname), $(date -u +%T), part $PART, sides $SIDES, dim $CI_RED_DIM, nproc $(nproc)"
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
B=$HOME/redsc_b; rm -rf $B; mkdir -p $B; NP=$(( $(nproc) < 16 ? $(nproc) : 16 ))
for side in $SIDES; do
  place $side
  echo "== files $side: $(while read -r rel; do [[ -n $rel ]] && sha1sum "$R/$rel" | cut -c1-10; done < $S/v/files.txt | tr '\n' ' ')"
  for form in nc wc; do
    d=$B/${side}_$form; mkdir -p $d; fl=(); rts=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE
    [[ $form == wc ]] && { fl=(--dump-perf-counters); rts=L1_TO_L1; }
    RUNNER_TEMP=$d TMPDIR=$d LLK_PERF_RUN_TYPES=$rts timeout -s INT -k 60 2400 python -m pytest --compile-producer -n $NP \
      -p no:cacheprovider -q "${fl[@]}" "${IDS[@]}" > /tmp/b_${side}_$form.txt 2>&1
    echo "== build ${side}_$form rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b_${side}_$form.txt | tail -1)"
  done
done
args=(); for side in $SIDES; do args+=($side=$B/${side}_nc); done
python $S/sweep/elfguard.py "${args[@]}" > /tmp/elf.txt; grc=$?
grep -E "^(PAIR|GUARD)" /tmp/elf.txt
[[ $grc == 0 ]] || { echo "== GUARD FAIL"; exit 1; }
run() {
  local tag=$1 side=$2 form=$3 rts=$4; shift 4
  RUNNER_TEMP=$B/${side}_$form TMPDIR=$B/${side}_$form LLK_PERF_RUN_TYPES=$rts PERF_RUN_TAG=$tag timeout -s INT -k 60 2400 \
    python -m pytest --compile-consumer -p no:cacheprovider -q -o timeout_method=thread --timeout 120 "$@" "${IDS[@]}" > /tmp/r_$tag.txt 2>&1
  echo "== run $tag rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1)"
}
REV=$(echo $SIDES | tr ' ' '\n' | tac | tr '\n' ' ')
j=0
for blk in 1 2; do for side in $SIDES $REV; do j=$((j + 1)); run sc_${j}_L_$side $side nc L1_TO_L1; done; done
for side in $SIDES; do run sc_iso_$side $side nc UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE; done
for side in $SIDES; do run sc_wc_$side $side wc L1_TO_L1 --dump-perf-counters; done
cd $LLK_HOME/perf_data/runs; cp /tmp/elf.txt sc_elf.txt
echo "== payload begin"
{ find sc_* -name "*.csv" ! -name "*.post.csv"; echo sc_elf.txt; } | tar czf - -T - 2>/dev/null | base64 -w 200
echo "== payload end"
echo "== DONE $(date -u +%T)"
