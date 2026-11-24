#!/bin/bash
# usage: env... zp.sh label [k]   (zone probe of the raw-first copy; AGRN_EXP_OP=base for the op)
cd /home/malimpic/localdev/2026_11_23/1317_malimpic_distributed_rms_norm_all_gather_gen/clones/all_gather_rms_norm_run1/tt-metal
source python_env/bin/activate
T=tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_experiments/gather_raw_first
O=ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/gather_raw_first/zones
mkdir -p $O
export TT_METAL_PROFILER_DIR=/tmp/agrn_rawprof_$1; rm -rf $TT_METAL_PROFILER_DIR
TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=0 TT_METAL_DEVICE_PROFILER=1 AGRN_PROFILE_ZONES=1 timeout 900 scripts/run_safe_pytest.sh --no-precompile $T/probe_raw.py -k "${2:-focus}" > /tmp/zp_raw_$1.log 2>&1
grep "SAFE_PYTEST_RESULT\|passed\|failed" /tmp/zp_raw_$1.log | tail -2
python tests/ttnn/unit_tests/operations/all_gather_rms_norm/zone_timeline.py $TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv > $O/zones_$1.txt 2>&1
