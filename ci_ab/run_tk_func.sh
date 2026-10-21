#!/usr/bin/env bash
# Scratch, not for merge. TopK functional tests (golden checked, LLK asserts on) per variant placed into the one checkout,
# each with its own build dir: compile all nodes, then run them. usage: run_tk_func.sh <variant>... (FUNC_MODS overrides
# the modules). Prints per variant passed / failed and the first failures.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; L=$R/tt_metal/tt-llk
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; (cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1)
export CHIP_ARCH=blackhole LLK_HOME=$L
unset TT_LLK_DISABLE_ASSERTS
cd $L/tests/python_tests
MODS=${FUNC_MODS:-test_topk_network.py test_topk.py}
for v in "$@"; do
  while read -r rel; do [[ -n $rel ]] && cp "$S/variants/$v/$rel" "$R/$rel"; done < $S/files.txt
  echo "== func $v files $(while read -r rel; do md5sum < $R/$rel | cut -c1-6; done < $S/files.txt | md5sum | cut -c1-12) $(date -u +%T)"
  RUNNER_TEMP=/tmp/fb_$v TMPDIR=/tmp/fb_$v timeout 3000 python -m pytest --compile-producer -n $(nproc) -p no:cacheprovider -q $MODS > /tmp/fp_$v.log 2>&1
  echo "== func $v produce rc=$? $(grep -E 'passed|failed|error' /tmp/fp_$v.log | tail -1 | cut -c1-150)"
  RUNNER_TEMP=/tmp/fb_$v TMPDIR=/tmp/fb_$v timeout 3600 python -m pytest --compile-consumer -p no:cacheprovider --timeout=300 -o timeout_method=thread -q -rfE $MODS > /tmp/fc_$v.log 2>&1
  echo "== func $v run rc=$? $(grep -E 'passed|failed|error' /tmp/fc_$v.log | tail -1 | cut -c1-150) $(date -u +%T)"
  grep -a -E "^(FAILED|ERROR) " /tmp/fc_$v.log | head -5 | cut -c1-300
done
echo "== done $(date -u +%T)"
