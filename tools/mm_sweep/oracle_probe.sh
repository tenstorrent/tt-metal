#!/bin/bash
# Probe which oracle geometries run vs hang (60s budget each). Args after BIN: --m..--nsb.
cd /localdev/cglagovich/tt-metal
source /home/cglagovich/bh_env.sh; source python_env/bin/activate >/dev/null 2>&1
BIN=build_Release/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm
run() {
  local tag="$1"; shift
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole \
    timeout -s TERM 70 $BIN --unified "$@" --num-tests 4 >/tmp/orc_$tag.txt 2>&1
  local rc=$?
  local res=$(grep -aoE "max_rel_err [0-9.]+|PASS|FAIL" /tmp/orc_$tag.txt | tail -2 | tr '\n' ' ')
  echo "PROBE $tag rc=$rc  $res"
}
run smallN_sm1  --m 256 --k 2048 --n 1024 --ksplit 8 --nslice 1 --msplit 1 --kb 1 --nsb 1
run smallN_sm2  --m 256 --k 2048 --n 1024 --ksplit 4 --nslice 1 --msplit 2 --kb 2 --nsb 2
run deepK_sm1   --m 256 --k 6144 --n 768  --ksplit 12 --nslice 1 --msplit 1 --kb 2 --nsb 1
run mt1_ref     --m 32  --k 6144 --n 4608 --ksplit 12 --nslice 1 --msplit 1 --kb 2 --nsb 1
echo "PROBE DONE"
