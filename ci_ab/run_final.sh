#!/usr/bin/env bash
# Writer census (positive and negative control) and the reduce test module on this commit.
set +e +o pipefail
R=/work; [ -d $R/ci_ab ] || R=$(git rev-parse --show-toplevel); cd $R
echo "== runner $(hostname) head $(git log -1 --format='%h %s' | cut -c1-80)"
export TT_METAL_CACHE=/tmp/fin_cache; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
python3 ci_ab/tools/census.py 2>&1 | grep -E "CENSUS|Error|FATAL"
python -m pytest -o timeout_method=thread -q -rfE tests/ttnn/unit_tests/operations/reduce 2>&1 | tail -15
exit 0
