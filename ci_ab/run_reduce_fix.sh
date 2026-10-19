#!/usr/bin/env bash
# Scratch job (#58820 zero-regression fix, reduce all-configuration sweep), not for merge. Same harness and rules as
# run_reduce_allcfg.sh with four sides: main (base reduce files, per-tile calls where head has a block call), head
# (21545319754), Z (head + the held-scaler MOP + BFP blocks tile by tile on both threads).
# L1_TO_L1: main head Z Z head main, twice; isolates and counters once a side. Guard: elfpairs.py.
# usage: run_reduce_fix.sh <part i/n>   (part 0 also runs the block max modules)
set +e +o pipefail
PART=${1:-0/4}; SIDES="main head Z"
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $(git log --oneline -1 | cut -c1-60), $(date -u +%T), part $PART, nproc $(nproc)"
place() { while read -r rel; do [[ -n $rel ]] && cp "$S/v/$1/$rel" "$R/$rel"; done < $S/v/files.txt; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$?"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$?"
T=$R/tt_metal/tt-llk/tests
cp $S/sweep/perf_reduce_allcfg.py $S/sweep/perf_reduce_block_max_ext.py $T/python_tests/
cp $S/sweep/reduce_allcfg_perf.cpp $T/sources/
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=400 CI_RED_PART=$PART
cd $T/python_tests
python -m pytest --collect-only -q -p no:cacheprovider perf_reduce_allcfg.py 2>/dev/null | grep "::" > /tmp/ids.txt
[[ ${PART%%/*} == 0 ]] && python -m pytest --collect-only -q -p no:cacheprovider perf_reduce_block_max.py perf_reduce_block_max_ext.py 2>/dev/null | grep "::" >> /tmp/ids.txt
echo "== ids $(wc -l < /tmp/ids.txt)"
mapfile -t IDS < /tmp/ids.txt
B=$HOME/redfix_b; rm -rf $B; mkdir -p $B; NP=$(( $(nproc) < 16 ? $(nproc) : 16 ))
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
python $S/sweep/elfpairs.py main=$B/main_nc head=$B/head_nc Z=$B/Z_nc > /tmp/elf.txt; grc=$?
grep -E "^(PAIR|GUARD)" /tmp/elf.txt
[[ $grc == 0 ]] || { echo "== GUARD FAIL"; exit 1; }
run() {
  local tag=$1 side=$2 form=$3 rts=$4; shift 4
  RUNNER_TEMP=$B/${side}_$form TMPDIR=$B/${side}_$form LLK_PERF_RUN_TYPES=$rts PERF_RUN_TAG=$tag timeout -s INT -k 60 2400 \
    python -m pytest --compile-consumer -p no:cacheprovider -q -o timeout_method=thread --timeout 120 "$@" "${IDS[@]}" > /tmp/r_$tag.txt 2>&1
  echo "== run $tag rc=$? $(date -u +%T)"
}
j=0
for blk in 1 2; do for side in main head Z Z head main; do j=$((j + 1)); run rf_${j}_L_$side $side nc L1_TO_L1; done; done
for side in $SIDES; do run rf_iso_$side $side nc UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE; done
for side in $SIDES; do run rf_wc_$side $side wc L1_TO_L1 --dump-perf-counters; done
cd $LLK_HOME/perf_data/runs; cp /tmp/elf.txt rf_elf.txt
echo "== payload begin"
{ find rf_* -name "*.csv" ! -name "*.post.csv"; echo rf_elf.txt; } | tar czf - -T - 2>/dev/null | base64 -w 200
echo "== payload end"
echo "== DONE $(date -u +%T)"
