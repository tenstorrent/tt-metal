#!/usr/bin/env bash
# Counters on this head (#58818 edfe22b1860) plus the scratch block-pack harness edit, tt-llk harness: the nodes
# ci_cnt/patterns.txt picks from ci_cnt/modules.txt, every run type,
# a no-counter build (timing, two runs) and a counter build (one run). Measurement only, no variants.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_cnt; cd $R
echo "== checkout $R, $(git log --oneline -1), $(date -u +%T)"
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.txt | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=400
export LLK_PERF_RUN_TYPES=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE
cd $R/tt_metal/tt-llk/tests/python_tests
python -m pytest --collect-only -q -p no:cacheprovider $(cat $S/modules.txt) > /tmp/all.txt 2>&1; echo "== collected $(grep -c '::' /tmp/all.txt)"
python $S/pick.py /tmp/all.txt $S/patterns.txt | sort -t: -k1,1 -s > /tmp/nodes.txt; cat /tmp/nodes.txt
mapfile -t NN < /tmp/nodes.txt
for mode in nc wc; do
  flag=""; [[ $mode == wc ]] && flag=--dump-perf-counters
  RUNNER_TEMP=$HOME/b_$mode TMPDIR=$HOME/b_$mode timeout -s INT -k 60 1500 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q $flag "${NN[@]}" > /tmp/b_$mode.txt 2>&1
  echo "== build $mode rc=$? $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/b_$mode.txt | tail -1)"
  tail -15 /tmp/b_$mode.txt | cut -c1-300
  reps="1 2"; [[ $mode == wc ]] && reps="1"
  for rep in $reps; do
    tag=${mode}_$rep
    RUNNER_TEMP=$HOME/b_$mode TMPDIR=$HOME/b_$mode PERF_RUN_TAG=$tag timeout -s INT -k 60 1500 python -m pytest --compile-consumer -p no:cacheprovider -q -rfE $flag "${NN[@]}" > /tmp/r_$tag.txt 2>&1
    echo "== run $tag rc=$? $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1)"
    grep -a -m5 -E "^FAILED|^ERROR" /tmp/r_$tag.txt | cut -c1-240
    grep -aq passed /tmp/r_$tag.txt || tail -60 /tmp/r_$tag.txt | cut -c1-300
    for f in $(find $LLK_HOME/perf_data/runs/$tag -name "*.csv" 2>/dev/null | grep -v -e post -e counters); do echo "== csv $tag $(basename $f)"; cat "$f"; echo "== end csv"; done
    for f in $(find $LLK_HOME/perf_data/runs/$tag -name "*.counters.csv" 2>/dev/null); do echo "== ccsv $tag $(basename $f)"; python $S/cfilter.py "$f"; echo "== end ccsv"; done
  done
done
echo "== DONE $(date -u +%T)"
