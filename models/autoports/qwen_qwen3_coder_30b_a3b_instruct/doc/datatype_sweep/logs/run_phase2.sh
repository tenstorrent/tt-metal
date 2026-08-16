#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
G="$(git rev-parse --short HEAD) + uncommitted stage-07 sweep"
hdr () { printf '# cmd: %s\n# git: %s\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n' "$1" "$G" "$(date -Is)"; }

# 1. Is a bfp8 KV cache broken, or just imprecise? Op-level, seconds.
C="python $D/doc/datatype_sweep/probes/kv_bfp8_diagnosis.py"
hdr "$C" > $L/kv_bfp8_diagnosis.log; eval $C >> $L/kv_bfp8_diagnosis.log 2>&1
echo "=== kv diagnosis exit $? ==="

# 2. Structural check of the stacked rows -- resolved in0_block_w for bw64.
C="python $D/doc/datatype_sweep/probes/structural_probe.py --layers 2 --only R23_gateup_bw64,R24_gateup32_down24,R25_gateup64_down24"
hdr "$C" > $L/structural_stacked.log; eval $C >> $L/structural_stacked.log 2>&1
echo "=== structural stacked exit $? ==="

# 3. The stacked rows at 48 layers.
C="python $D/doc/datatype_sweep/probes/sweep_runner.py --include-stacked --only R23_gateup_bw64,R24_gateup32_down24,R25_gateup64_down24"
hdr "$C" > $L/sweep_stacked.log; eval $C >> $L/sweep_stacked.log 2>&1
echo "=== stacked sweep exit $? ==="
