#!/usr/bin/env bash
# Scratch, not for merge. Diagnose the head's allcfg runs that end the pytest session: each id runs in its own pytest
# process (M main base, H head, HA head with LLK asserts on); a failing run prints its last lines.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; L=$R/tt_metal/tt-llk; FL=$S/files.txt
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; echo "== pip rc=$?"
(cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1); echo "== setup rc=$?"
export CHIP_ARCH=blackhole LLK_HOME=$L LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=1000 LLK_PERF_RUN_TYPES=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE
place() { while read -r rel; do [[ -n $rel ]] && cp "$S/variants/$1/$rel" "$R/$rel"; done < $FL; }
cd $L/tests/python_tests
for spec in M:ids/diag_main.txt H:ids/diag_head.txt HA:ids/diag_head.txt; do
  v=${spec%%:*}; ids=$S/${spec##*:}; d=${v%A}; place $d
  if [[ $v == HA ]]; then unset TT_LLK_DISABLE_ASSERTS; else export TT_LLK_DISABLE_ASSERTS=1; fi
  mapfile -t NODES < $ids
  RUNNER_TEMP=/tmp/d_$v TMPDIR=/tmp/d_$v timeout 1200 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q "${NODES[@]}" > /tmp/dprod_$v.log 2>&1
  echo "== produce $v rc=$? $(grep -E 'passed|failed|error' /tmp/dprod_$v.log | tail -1 | cut -c1-120)"
  n=0
  for id in "${NODES[@]}"; do
    n=$((n + 1))
    RUNNER_TEMP=/tmp/d_$v TMPDIR=/tmp/d_$v PERF_RUN_TAG=d_${v}_$n timeout 300 python -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -q -rfE "$id" > /tmp/drun_${v}_$n.log 2>&1
    rc=$?
    echo "== $v $n rc=$rc $(grep -E 'passed|failed|error|skipped' /tmp/drun_${v}_$n.log | tail -1 | cut -c1-100) :: ${id:30:170}"
    if [[ $rc != 0 ]]; then grep -a -v "^\s*$" /tmp/drun_${v}_$n.log | grep -a -i -E "error|assert|timeout|hang|exit|fail|exception|raise|tensix|core" | tail -25 | cut -c1-260; fi
  done
done
echo "== done"
