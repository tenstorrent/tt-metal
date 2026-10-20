#!/usr/bin/env bash
# Scratch, not for merge. TopK functional tests of the checked-out tree (golden checked, LLK asserts on).
set +e +o pipefail
R=$(git rev-parse --show-toplevel); L=$R/tt_metal/tt-llk
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; (cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1)
export CHIP_ARCH=blackhole LLK_HOME=$L
unset TT_LLK_DISABLE_ASSERTS
cd $L/tests/python_tests
MODS="test_topk_network.py test_topk.py test_topk_const_leak.py test_topk_finalize_uint16.py test_topk_narrow_tag.py test_sfpu_generic_moe_gate_topk.py"
echo "== func $(git log -1 --format=%h) $(date -u +%T)"
RUNNER_TEMP=/tmp/fb TMPDIR=/tmp/fb timeout 3000 python -m pytest --compile-producer -n $(nproc) -p no:cacheprovider -q $MODS > /tmp/fp.log 2>&1
echo "== func produce rc=$? $(grep -E 'passed|failed|error' /tmp/fp.log | tail -1 | cut -c1-150)"
RUNNER_TEMP=/tmp/fb TMPDIR=/tmp/fb timeout 3600 python -m pytest --compile-consumer -p no:cacheprovider --timeout=300 -o timeout_method=thread -q -rfE $MODS > /tmp/fc.log 2>&1
echo "== func run rc=$? $(grep -E 'passed|failed|error' /tmp/fc.log | tail -1 | cut -c1-150) $(date -u +%T)"
grep -a -c "TENSIX TIMED OUT" /tmp/fc.log | sed 's/^/== timed out lines /'
grep -a -E "^(FAILED|ERROR) " /tmp/fc.log | head -8 | cut -c1-300
echo "== done $(date -u +%T)"
