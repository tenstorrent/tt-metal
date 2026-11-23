#!/bin/bash
# Static code size and TT/TTI counts of one finished HAL run, rebuilt compile-only twice:
#   assert_on  = default test build (-DENABLE_LLK_ASSERT), control: must equal the run's postcheck tt_tti_counts
#   assert_off = TT_LLK_DISABLE_ASSERTS=1 (removes only -DENABLE_LLK_ASSERT, test_config.py ~553-557; perf jobs use it)
# Same build as the harness code-size step (original + HAL test in one --compile-producer session, -k add),
# only with fewer workers and nice 19. No card access.
# Never writes run dirs or tools/perf_cycles/ws_8a: own hard-link copy of ws_8a, every replaced file removed first,
# pytest cache provider off.
# Usage (inside a bh-30 board container that runs nothing else): build_counts.sh <name> <run dir>
set -uo pipefail
B=/proj_sw/user_dev/ctr-sbratic; A=$B/tools/asserts_off; N=${1:?name}; R=${2:?run dir}
W=$A/ws/$N; T=$W/tt_metal/tt-llk/tests; O=$A/out/$N
RT=$R/ws/tt_metal/tt-llk/tests
JOBS=${JOBS:-8}
[[ -d $RT/hal_llk && -f $RT/sources/eltwise_binary_hal_test.cpp && -f $RT/python_tests/test_eltwise_binary_hal.py ]] \
  || { echo "$N: run files missing in $RT"; exit 1; }
mkdir -p $O $A/ws
[[ -d $W ]] || cp -al $B/tools/perf_cycles/ws_8a $W || { echo "$N: cp -al failed"; exit 1; }
rm -rf $T/hal_llk $T/.pytest_cache $W/tt_metal/tt-llk/perf_data
rm -f $T/sources/eltwise_binary_hal_test.cpp $T/python_tests/test_eltwise_binary_hal.py
cp -a $RT/hal_llk $T/hal_llk
cp -a $RT/sources/eltwise_binary_hal_test.cpp $T/sources/eltwise_binary_hal_test.cpp
cp -a $RT/python_tests/test_eltwise_binary_hal.py $T/python_tests/test_eltwise_binary_hal.py
same_llk=$(cd $RT/hal_llk && md5sum * | (cd $T/hal_llk && md5sum -c --quiet - >/dev/null 2>&1 && echo yes || echo NO))
same_cpp=$(cmp -s $RT/sources/eltwise_binary_hal_test.cpp $T/sources/eltwise_binary_hal_test.cpp && echo yes || echo NO)
same_py=$(cmp -s $RT/python_tests/test_eltwise_binary_hal.py $T/python_tests/test_eltwise_binary_hal.py && echo yes || echo NO)
same_orig=$(cmp -s $RT/sources/eltwise_binary_test.cpp $T/sources/eltwise_binary_test.cpp \
  && cmp -s $RT/python_tests/test_eltwise_binary.py $T/python_tests/test_eltwise_binary.py && echo yes || echo NO)
{ echo "name=$N"; echo "run_dir=$R"; echo "hal_llk_files=$(ls $T/hal_llk | tr '\n' ' ')"
  echo "hal_llk_md5_equal_to_run=$same_llk"; echo "hal_test_cpp_equal_to_run=$same_cpp"
  echo "hal_test_py_equal_to_run=$same_py"; echo "original_test_cpp_py_equal_to_run=$same_orig"; } > $O/SOURCE.txt
[[ $same_llk == yes && $same_cpp == yes && $same_py == yes && $same_orig == yes ]] \
  || { cat $O/SOURCE.txt; echo "$N: SETUP CHECK FAILED"; exit 1; }

source $B/infra/hal-env-ird/.venv/bin/activate; export PATH=$PATH:$HOME/.local/bin
TOOLS=$T/sfpi/compiler/bin
for mode in assert_on assert_off; do
  D=$O/$mode; rm -rf $D; mkdir -p $D/codesize_tmp
  if [[ $mode == assert_off ]]; then export TT_LLK_DISABLE_ASSERTS=1; else unset TT_LLK_DISABLE_ASSERTS; fi
  echo "TT_LLK_DISABLE_ASSERTS=${TT_LLK_DISABLE_ASSERTS:-<unset>}" > $D/env.txt
  start=$(date +%s)
  (cd $T && TMPDIR=$D/codesize_tmp nice -n 19 timeout 2h pytest python_tests/test_eltwise_binary.py \
     python_tests/test_eltwise_binary_hal.py -k add --compile-producer -n $JOBS -q -p no:cacheprovider \
     > $D/compile.txt 2>&1); rc=$?
  echo "$N $mode: compile exit $rc, $(( $(date +%s) - start )) s, $(grep -E '(passed|failed|error)' $D/compile.txt | tail -1)" \
    | tee -a $O/status.txt
  ELFRAY=$B/tools/elfray nice -n 19 python3 $B/tools/tt_tti_run6/tt_tti_counts.py $D --objdump $TOOLS/riscv-tt-elf-objdump \
    --md $D/tt_tti_counts.md --json $D/tt_tti_counts.json > $D/tt_tti_counts_stdout.txt 2>&1 \
    || echo "$N $mode: tt_tti_counts FAILED" | tee -a $O/status.txt
  nice -n 19 python3 $B/harness/hal-harness-run5rules/codesize.py --artefacts $D/codesize_tmp/tt-llk-build --tools $TOOLS \
    --orig eltwise_binary_test.cpp --hal eltwise_binary_hal_test.cpp --json $D/codesize.json --md $D/codesize.md \
    > /dev/null 2>&1 || echo "$N $mode: codesize FAILED" | tee -a $O/status.txt
done
unset TT_LLK_DISABLE_ASSERTS
echo "$N DONE $(date -u +%FT%TZ)" | tee -a $O/status.txt
