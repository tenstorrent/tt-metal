#!/usr/bin/env bash
# Scratch A/B job of the misc family (regression audit phase 2), not for merge. One job, one machine: the #60123 rows
# whose configuration the PR changes and main can run (perf_eltwise_binary_sfpu_mask, every tile pair), main's mask code
# (RA: #60121 merged, the adapter and main's body, the head's python) against #60123 merged (the checkout), each side
# with its own fresh build dir, alternated ABBA in two blocks, plus an A/A pair of the head.
set +e +o pipefail
cd /work; unset PYTHONPATH
# the harness builds under $RUNNER_TEMP when set (GHA), which would put every side in one build dir
unset RUNNER_TEMP
LT=tt_metal/tt-llk
OUT=/work/generated/mk; rm -rf $OUT; mkdir -p $OUT
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s')"
uv venv -q /tmp/llkenv > $OUT/venv.log 2>&1
uv pip install -q --python /tmp/llkenv/bin/python -r $LT/tests/requirements.txt >> $OUT/venv.log 2>&1 || { echo "venv failed"; tail -30 $OUT/venv.log; exit 1; }
PY=/tmp/llkenv/bin/python
(cd $LT/tests && ./setup_testing_env.sh > $OUT/sfpi.log 2>&1) ; tail -2 $OUT/sfpi.log
TOUCHED=$(cd ci_ab/v && find . -type f | sed 's#^\./[^/]*/##' | sort -u)
apply() { git checkout -q HEAD -- $TOUCHED; [[ $1 == head* ]] || cp -r ci_ab/v/$1/. .; }
ids() { mapfile -t N < ci_ab/ids/$1.txt; }
export CHIP_ARCH=blackhole LLK_HOME=/work/$LT LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=4000
prod() { # prod <variant> <build> <ids>
  apply $1; ids $3
  (cd $LT/tests/python_tests && TT_LLK_DISABLE_ASSERTS=1 TMPDIR=/tmp/b_$2 PERF_RUN_TAG=prod_$2 timeout 1800 $PY -m pytest --compile-producer -n 8 \
    -p no:cacheprovider --override-ini=log_cli=false -q "${N[@]}" > $OUT/prod_$2.log 2>&1)
  echo "prod $2 ($1) rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/prod_$2.log | tail -1)"
}
cons() { # cons <variant> <build> <ids> <tag>
  apply $1; ids $3
  (cd $LT/tests/python_tests && TT_LLK_DISABLE_ASSERTS=1 TMPDIR=/tmp/b_$2 PERF_RUN_TAG=$4 timeout 1500 $PY -m pytest --compile-consumer \
    -p no:cacheprovider --override-ini=log_cli=false -q -o timeout_method=thread "${N[@]}" > $OUT/$4.log 2>&1)
  local rc=$?; echo "run $4 $2 rc=$rc $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/$4.log | tail -1)"
  grep -q "TIMED OUT" $OUT/$4.log && { echo "TENSIX TIMED OUT in $4, stopping"; exit 2; }
}
pair() { # pair <tag> <A variant> <A build> <B variant> <B build> <ids>
  local k=0
  for b in 1 2; do for s in A B B A; do k=$((k+1))
    if [[ $s == A ]]; then cons $2 $3 $6 ${1}_A$k; else cons $4 $5 $6 ${1}_B$k; fi
  done; done
}
prod RA RA mask_bin; prod head H mask_bin; prod head H2 mask_bin
pair mk RA RA head H mask_bin
pair maa head H head_aa H2 mask_bin
apply head
echo "==BEGIN_RESULTS"
tar -C $LLK_HOME/perf_data/runs -czf - $(cd $LLK_HOME/perf_data/runs && ls -d mk_* maa_* 2>/dev/null) | base64 -w 200
echo "==END_RESULTS"
for f in $OUT/prod_RA.log $OUT/prod_H.log $OUT/mk_A1.log $OUT/mk_B2.log; do echo "== tail $f"; grep -aE 'FAILED|ERROR|passed|failed' $f | tail -5; done
