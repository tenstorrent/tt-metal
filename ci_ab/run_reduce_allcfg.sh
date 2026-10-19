#!/usr/bin/env bash
# Scratch job (reduce all-configuration sweep, #58820 head against its main base), not for merge. tt-llk harness on one CI
# Blackhole runner (17_CI_AB.md). Tree: #58820's head plus ci_ab/. Sides: main = the base's llk_math_reduce.h and
# llk_unpack_AB_reduce.h (per-tile calls where the head has a block call), head = the head's; ci_ab/v/<side>/ files are
# placed at their tree paths before each build. Each side builds once per form in a fresh dir: nc (four run types) and wc
# (L1_TO_L1 with counters). Runs: L1_TO_L1 main head head main, twice (ABBA, the repeats are the A/A); then the three
# isolates and the counter run once a side. Guard: elfhash.py (main != head on some variant, same variant set).
# usage: run_reduce_allcfg.sh <part i/n>   (part 0 also runs the block max modules)
set +e +o pipefail
PART=${1:-0/4}
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $R, $(git log --oneline -1 | cut -c1-80), $(hostname), $(date -u +%T), part $PART, nproc $(nproc)"
place() { while read -r rel; do [[ -n $rel ]] && cp "$S/v/$1/$rel" "$R/$rel"; done < $S/v/files.txt; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.txt | tr '\n' ' ')"
T=$R/tt_metal/tt-llk/tests
cp $S/sweep/perf_reduce_allcfg.py $S/sweep/perf_reduce_block_max_ext.py $T/python_tests/
cp $S/sweep/reduce_allcfg_perf.cpp $T/sources/
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=400 CI_RED_PART=$PART
cd $T/python_tests
python -m pytest --collect-only -q -p no:cacheprovider perf_reduce_allcfg.py 2>/dev/null | grep "::" > /tmp/ids.txt
if [[ ${PART%%/*} == 0 ]]; then
  python -m pytest --collect-only -q -p no:cacheprovider perf_reduce_block_max.py perf_reduce_block_max_ext.py 2>/dev/null | grep "::" >> /tmp/ids.txt
fi
echo "== ids $(wc -l < /tmp/ids.txt)"
mapfile -t IDS < /tmp/ids.txt
B=$HOME/redall_b; rm -rf $B; mkdir -p $B
for side in main head; do
  place $side
  echo "== files $side: $(while read -r rel; do [[ -n $rel ]] && sha1sum "$R/$rel" | cut -c1-10; done < $S/v/files.txt | tr '\n' ' ')"
  for form in nc wc; do
    d=$B/${side}_$form; mkdir -p $d; fl=(); rts=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE
    [[ $form == wc ]] && { fl=(--dump-perf-counters); rts=L1_TO_L1; }
    RUNNER_TEMP=$d TMPDIR=$d LLK_PERF_RUN_TYPES=$rts timeout -s INT -k 60 2400 python -m pytest --compile-producer -n $(( $(nproc) < 16 ? $(nproc) : 16 )) \
      -p no:cacheprovider -q "${fl[@]}" "${IDS[@]}" > /tmp/b_${side}_$form.txt 2>&1
    echo "== build ${side}_$form rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b_${side}_$form.txt | tail -1)"
    grep -a -m3 -E "FAILED|error:" /tmp/b_${side}_$form.txt | cut -c1-300
  done
done
python $S/sweep/elfhash.py main=$B/main_nc head=$B/head_nc > /tmp/elf.txt; grc=$?
grep -E "^(SRC|GUARD)" /tmp/elf.txt
[[ $grc == 0 ]] || { echo "== GUARD FAIL"; exit 1; }
run() {  # tag side form run_types [extra]
  local tag=$1 side=$2 form=$3 rts=$4; shift 4
  RUNNER_TEMP=$B/${side}_$form TMPDIR=$B/${side}_$form LLK_PERF_RUN_TYPES=$rts PERF_RUN_TAG=$tag timeout -s INT -k 60 2400 \
    python -m pytest --compile-consumer -p no:cacheprovider -q -o timeout_method=thread --timeout 120 "$@" "${IDS[@]}" > /tmp/r_$tag.txt 2>&1
  echo "== run $tag rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1)"
  grep -a -m3 -E "FAILED|TIMED OUT|Timeout" /tmp/r_$tag.txt | cut -c1-200
}
j=0
for blk in 1 2; do
  for side in main head head main; do j=$((j + 1)); run ra_${j}_L_$side $side nc L1_TO_L1; done
done
for side in main head; do run ra_iso_$side $side nc UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE; done
for side in head main; do run ra_wc_$side $side wc L1_TO_L1 --dump-perf-counters; done
cd $LLK_HOME/perf_data/runs
echo "== payload begin"
cp /tmp/elf.txt ra_elf.txt
{ find ra_* -name "*.csv" ! -name "*.post.csv"; echo ra_elf.txt; } | tar czf - -T - 2>/dev/null | base64 -w 200
echo "== payload end"
echo "== DONE $(date -u +%T)"
