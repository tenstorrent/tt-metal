#!/usr/bin/env bash
# Scratch job (#60463 drain A/B on the pack untilize init forms), not for merge. tt-llk harness on one CI Blackhole runner
# (17_CI_AB.md). The tree is the merge commit of #60463 (56b34fc0780) plus #60144's perf module files. Variants (ci_ab/v/<V>,
# the files of ci_ab/v/files.txt, placed at their tree paths before each build and run): A = #60463's parent (4993389394d) device
# files, B = the merge commit's, A2 and B2 = the same files again in their own builds (A/A and B/B copies). Each variant builds once
# in a fresh build dir. Runs go forward A B A2 B2 then backward B2 A2 B A, twice (ABBA); in a forward run one pytest call does the
# three forms (once, standard, custom), in a backward run one call per form, last form first. L1_TO_L1 and PACK_ISOLATE.
# Guard: A != B, A == A2, B == B2 on the pack ELF of every run type and form, else the job stops.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $R, $(git log --oneline -1), $(hostname), $(date -u +%T)"
src() { case $1 in A2) echo A ;; B2) echo B ;; *) echo $1 ;; esac; }
place() { while read -r rel; do [[ -n $rel ]] && cp "$S/v/$1/$rel" "$R/$rel"; done < $S/v/files.txt; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.txt | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1,PACK_ISOLATE PERF_KEEP_RUNS=400
cd $R/tt_metal/tt-llk/tests/python_tests
N=perf_matmul_pack_untilize.py
VS="A B A2 B2"
EC=""
for v in $VS; do
  place $(src $v); rm -rf $HOME/pud_$v
  echo "== files $v: $(while read -r rel; do [[ -n $rel ]] && sha1sum "$R/$rel" | cut -c1-10; done < $S/v/files.txt | tr '\n' ' ')"
  RUNNER_TEMP=$HOME/pud_$v TMPDIR=$HOME/pud_$v timeout -s INT -k 60 900 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q $N > /tmp/b_$v.txt 2>&1
  echo "== build $v rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b_$v.txt | tail -1)"
  EC="$EC $v=$HOME/pud_$v"
done
python $S/elfcheck_drain.py $EC | tee /tmp/elf.txt
grep -q "^GUARD ok" /tmp/elf.txt || { echo "== GUARD FAIL: the variants did not build as expected"; exit 1; }
RV=$(echo $VS | tr ' ' '\n' | tac | tr '\n' ' ')
j=0
for blk in 1 2; do
  for dir in fwd rev; do
    if [[ $dir == fwd ]]; then vs=$VS; forms="all"; else vs=$RV; forms="custom standard once"; fi
    for v in $vs; do
      place $(src $v)
      for f in $forms; do
        j=$((j + 1)); tag=pud_${j}_${v}_${f}; sel=(); [[ $f != all ]] && sel=(-k $f)
        RUNNER_TEMP=$HOME/pud_$v TMPDIR=$HOME/pud_$v PERF_RUN_TAG=$tag timeout -s INT -k 60 600 \
          python -m pytest --compile-consumer -p no:cacheprovider -q -rfE "${sel[@]}" $N > /tmp/r_$tag.txt 2>&1
        echo "== run $tag ($dir) rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1)"
        [[ $j == 1 ]] && grep -a -m5 -E "FAILED|Error|error" /tmp/r_$tag.txt | cut -c1-200
        for c in $(ls $LLK_HOME/perf_data/runs/$tag/*/perf_matmul_pack_untilize*.csv 2>/dev/null | grep -v post); do echo "== csv $tag"; cat "$c"; echo "== end csv"; done
      done
    done
  done
done
echo "== DONE $(date -u +%T)"
