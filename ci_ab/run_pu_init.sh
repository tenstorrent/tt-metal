#!/usr/bin/env bash
# Scratch job (#60144 per-block rows), not for merge. tt-llk harness on one CI Blackhole runner (17_CI_AB.md). The job's tree
# is main with #60144's perf module; the PR changes no device file, so the table is the baseline on main's kernels. The
# module's three pack untilize init forms (once before the loop, standard and custom per block) are built once (producer),
# then run on the device four times, forward, reverse, forward, reverse (reverse = one pytest call per form, last form
# first), L1_TO_L1 and PACK_ISOLATE; every run's CSV goes to the log between markers. Guard: the pack ELFs of the three
# forms must differ in every run type (the template parameter reached the compiler).
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $R, $(git log --oneline -1), $(hostname), $(date -u +%T)"
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.txt | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1,PACK_ISOLATE PERF_KEEP_RUNS=400
cd $R/tt_metal/tt-llk/tests/python_tests
N=perf_matmul_pack_untilize.py
rm -rf $HOME/pui_build; mkdir -p $HOME/pui_build
RUNNER_TEMP=$HOME/pui_build TMPDIR=$HOME/pui_build timeout -s INT -k 60 900 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q $N > /tmp/b.txt 2>&1
echo "== build rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b.txt | tail -1)"
python $S/elfcheck_pu.py $HOME/pui_build | tee /tmp/elf.txt
grep -q "^GUARD ok" /tmp/elf.txt || { echo "== GUARD FAIL: the init forms did not build apart"; exit 1; }
j=0
for pass in 1 2; do
  for dir in fwd rev; do
    if [[ $dir == fwd ]]; then forms="all"; else forms="custom standard once"; fi
    for f in $forms; do
      j=$((j + 1)); tag=pu_${j}_${dir}_${f}; sel=(); [[ $f != all ]] && sel=(-k $f)
      RUNNER_TEMP=$HOME/pui_build TMPDIR=$HOME/pui_build PERF_RUN_TAG=$tag timeout -s INT -k 60 600 \
        python -m pytest --compile-consumer -p no:cacheprovider -q -rfE "${sel[@]}" $N > /tmp/r_$tag.txt 2>&1
      echo "== run $tag rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1)"
      [[ $j == 1 ]] && grep -a -m5 -E "FAILED|Error|error" /tmp/r_$tag.txt | cut -c1-200
      for c in $(ls $LLK_HOME/perf_data/runs/$tag/*/perf_matmul_pack_untilize*.csv 2>/dev/null | grep -v post); do echo "== csv $tag"; cat "$c"; echo "== end csv"; done
    done
  done
done
echo "== DONE $(date -u +%T)"
