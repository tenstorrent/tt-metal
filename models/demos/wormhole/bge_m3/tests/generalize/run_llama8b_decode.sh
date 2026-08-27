#!/usr/bin/env bash
# Llama-3.1-8B decode (M=32 = 1 token tile) matmul sweep across all distinct
# linear-layer GEMM shapes, all 3 impls, core grids {16,32,64}.
# Each (shape,impl,grid) is its own invocation so the --max-configs budget is
# never starved across combos. Writes one CSV per combo under out/llama8b/.
set -u
cd "$(dirname "$0")/../../../../../.." || exit 1   # -> repo root
source python_env/bin/activate
source /localdev/gtobar/bge_optimization/local_env.sh >/dev/null 2>&1
export TT_VISIBLE_DEVICES=0

SW=models/demos/wormhole/bge_m3/tests/generalize/matmul_sweep.py
OUT=models/demos/wormhole/bge_m3/tests/generalize/out/llama8b
mkdir -p "$OUT"

# shape name -> "K N"  (M is fixed at 32 for decode)
declare -A SHAPES=(
  [q_wo]="4096 4096"      # Q (Wq) and Wo (attn out): both 4096x4096
  [kv]="4096 1024"        # K (Wk) and V (Wv): 4096x1024
  [w1_w3]="4096 14336"    # W1 (gate) and W3 (up): 4096x14336
  [w2]="14336 4096"       # W2 (down): 14336x4096
)

M=32
GRIDS="16 32 64"

for name in "${!SHAPES[@]}"; do
  read -r K N <<< "${SHAPES[$name]}"
  for impl in minmatmul matmul2d matmul1d; do
    # decode: M is a single tile, so minmatmul M_block must be 1.
    MB="--m-blocks 1"
    if [ "$impl" != "minmatmul" ]; then MB=""; fi
    for cores in $GRIDS; do
      csv="$OUT/${name}_${impl}_M${M}_K${K}_N${N}_g${cores}.csv"
      echo "=== ${name} | ${impl} | M${M} K${K} N${N} | cores=${cores} ==="
      timeout 1200 python3 "$SW" --M "$M" --K "$K" --N "$N" \
        --impl "$impl" --grids "$cores" $MB \
        --k-blocks 1 2 4 8 16 --n-blocks 1 2 4 8 16 32 \
        --dtypes bfloat8_b --fidelities LoFi --out-memcfgs dram l1 \
        --iters 5 --max-configs 60 --csv "$csv" \
        2>&1 | grep -iE "BEST ${impl}|trials:" | grep -viE "Config\{"
    done
  done
done
echo "ALL DONE"
