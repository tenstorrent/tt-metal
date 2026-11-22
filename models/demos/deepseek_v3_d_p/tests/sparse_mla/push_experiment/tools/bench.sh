#!/usr/bin/env bash
# bench.sh <tag> [layers] [starts] [dprint 0/1]
S=${PUSH_EXP_DIR:-/tmp/push_exp}
cd /home/pjosipovic/tt-metal && source python_env/bin/activate && unset TT_METAL_ENV TT_METAL_CCACHE_KERNEL_SUPPORT
export BENCH_INBOX=${BENCH_INBOX:-0} BENCH_PAYLOAD=${BENCH_PAYLOAD:-6144} MESH_DEVICE=TG TT_METAL_SHM_TRACKING_DISABLED=1 LOGURU_LEVEL=WARNING BENCH_LAYERS=${2:-14,62} BENCH_STARTS=${3:-1039360}
if [ "${4:-0}" = 1 ]; then export TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_CHIPS=${DPCHIPS:-0} TT_METAL_DPRINT_RISCVS=BR; fi
timeout 1500 scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_zz_push_bench_scratch.py -s > $S/bench_$1.log 2>&1
grep -oE "BENCH .*" $S/bench_$1.log; grep SAFE_PYTEST_RESULT $S/bench_$1.log
if [ "${4:-0}" = 1 ]; then grep -oE "PUSHPROF.*" $S/bench_$1.log | awk '{w=$3; dir=$5; link=$7; pk=$9; d2=$14; d3=$15; key=w" dir"dir" l"link; n[key]++; s[key]+=d3; f[key]+=d2; p[key]=pk} END {for (k in n) printf "%s pk %d fence_us %.0f data_done_us %.0f\n", k, p[k], f[k]/n[k]/1350, s[k]/n[k]/1350}' | sort -k1,1n; fi
