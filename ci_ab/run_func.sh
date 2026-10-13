#!/usr/bin/env bash
# Functional ROW MAX tests (test_reduce.py, test_reduce_block.py) with LLK asserts on, for each label in $FUNC_LABELS.
set +e +o pipefail
unset RUNNER_TEMP
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; SL=/tmp/hr_func; cd $R; rm -rf $SL; mkdir -p $SL
python3 -m venv $SL/venv && $SL/venv/bin/pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip.log 2>&1; PY=$SL/venv/bin/python3
[[ -x $PY ]] || PY=$(command -v python3)
SF=""; for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]] && { SF=$c; break; }; done
if [[ -z $SF ]]; then (cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $SL/sfpi.log 2>&1); SF=$R/tt_metal/tt-llk/tests/sfpi; fi
echo "== sfpi $($SF/compiler/bin/riscv-tt-elf-g++ --version | head -1)"
P="tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon tt_metal/tools/profiler tt_metal/sfpi-version ttnn/cpp/ttnn/operations/experimental"
for L in ${FUNC_LABELS:-V1}; do
  D=$SL/$L; mkdir -p $D $SL/b_$L; tar -C $R -cf - $P | tar -C $D -xf -
  [[ -d $S/variants/$L/tree ]] && (cd $S/variants/$L/tree && find . -type f ! -name .keep) | while read -r f; do cp "$S/variants/$L/tree/$f" "$D/$f"; done
  ln -sfn $SF $D/tt_metal/tt-llk/tests/sfpi
  echo "== $L reduce header $(sha256sum $D/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_reduce.h | cut -c1-12)"
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/func_rowmax.txt
    NP=$(( $(nproc) / 2 )); (( NP > 16 )) && NP=16; (( NP < 4 )) && NP=4
    env CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TMPDIR=$SL/b_$L timeout -s INT -k 60 2400 $PY -m pytest --compile-producer -n $NP -p no:cacheprovider \
      --override-ini=log_cli=false -q "${N[@]}" < /dev/null > $SL/prod_$L.log 2>&1
    echo "== built $L $(grep -aE '[0-9]+ (passed|failed|error)' $SL/prod_$L.log | tail -1)"
    env CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TMPDIR=$SL/b_$L timeout -s INT -k 60 2400 $PY -m pytest --compile-consumer -p no:cacheprovider \
      --override-ini=log_cli=false -q -o timeout_method=thread "${N[@]}" < /dev/null > $SL/func_$L.log 2>&1 )
  echo "== func $L rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $SL/func_$L.log | tail -1)"; grep -aE "^FAILED|^ERROR" $SL/func_$L.log | head -10 | cut -c1-250
done
exit 0
