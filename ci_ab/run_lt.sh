#!/usr/bin/env bash
# Scratch job, not for merge: tt-llk untilize tests with LLK asserts on (LT side = LIM2 + original fast_untilize_test.cpp). Env: IDS=<file in ci_ab>.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $R, $(git log --oneline -1 | cut -c1-80), $(hostname), $(date -u +%T)"
L=$R/tt_metal/tt-llk; DIRS="tt-llk/tt_llk_blackhole tt-llk/common tt-llk/tests/sources tt-llk/tests/helpers tt-llk/tests/python_tests hw/ckernels/blackhole/metal/llk_api hw/inc/internal/tt-1xx/blackhole"
for x in $DIRS; do rm -rf "$R/tt_metal/$x"; done; tar -xzf $S/sides/LT.tar.gz -C $R/tt_metal
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$?"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$?"
export CHIP_ARCH=blackhole LLK_HOME=$L
d=$HOME/lt_$IDS; rm -rf $d; mkdir -p $d; cd $L/tests/python_tests
mapfile -t N < $S/$IDS
RUNNER_TEMP=$d TMPDIR=$d timeout -s INT -k 60 3000 python -m pytest --compile-producer -n 8 -p no:cacheprovider --override-ini=log_cli=false -q "${N[@]}" > /tmp/b.txt 2>&1
echo "== build rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b.txt | tail -1)"
RUNNER_TEMP=$d TMPDIR=$d timeout -s INT -k 60 3000 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -q -rfE "${N[@]}" > /tmp/r.txt 2>&1
echo "== run rc=$? $(grep -aE '[0-9]+ (passed|failed)' /tmp/r.txt | tail -1)"
grep -a -E "^FAILED|TIMED OUT" /tmp/r.txt | head -20 | cut -c1-240
echo "== DONE $(date -u +%T)"
