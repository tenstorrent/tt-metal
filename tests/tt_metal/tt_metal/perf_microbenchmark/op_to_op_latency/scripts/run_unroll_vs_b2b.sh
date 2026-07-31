#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Unroll-vs-back-to-back experiment. For each repeat count K, run the SAME total workload two ways
# under trace replay and compare wall-clock:
#   back-to-back : --num-programs K --kernel-unroll 1  (K separate program enqueues; each program
#                  ends in a write barrier and there is an op-to-op sync between enqueues)
#   unrolled     : --num-programs 1 --kernel-unroll K  (one program; kernels repeat the workload K
#                  times with NO barrier between reps -- one write barrier only at the very end)
# The delta is the op-to-op sync-barrier cost removed by fusing the K executions. per_boundary_us
# = (b2b - unroll) / (K-1). Everything else (cores/volume/CBs/math/NoCs) is held at the locked
# Stage-A config.
#
# Env: CORES, PAGES, NOPS, OUT_CB, IN_CB, N, KS (repeat counts), REPS, OUT.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)}"
cd "${TT_METAL_HOME}" || exit 1
BIN="${TT_METAL_HOME}/build_RelWithDebInfo/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
OUTDIR="${OUTDIR:-${TT_METAL_HOME}/generated/profiler/op_to_op_runs/stage_a}"
OUT="${OUT:-${OUTDIR}/unroll_vs_b2b.csv}"
mkdir -p "${OUTDIR}"

CORES="${CORES:-56}"
PAGES="${PAGES:-256}"
NOPS="${NOPS:-800}"     # balanced math (locked Stage-A operating point)
OUT_CB="${OUT_CB:-32}"
IN_CB="${IN_CB:-16}"
N="${N:-4}"
KS="${KS:-2 4 8}"
REPS="${REPS:-3}"

# common locked config; $1 = num_programs, $2 = kernel_unroll
run_elapsed() {
  local nprog="$1" unroll="$2"
  "${BIN}" --skip-output-validation --lean-compute --reader-noc 0 --writer-noc 1 \
    --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
    --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "${OUT_CB}" --writer-end-barrier-mode 0 \
    --num-pages-per-core "${PAGES}" --num-programs "${nprog}" --kernel-unroll "${unroll}" \
    --compute-nops "${NOPS}" --use-trace --trace-warmup-replays 2 --num-active-cores "${CORES}" \
    2>&1 | grep -oE 'programs in [0-9]+ us' | grep -oE '[0-9]+'
}

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:int((a[NR/2]+a[NR/2+1])/2)}'; }

echo "K,b2b_us,unroll_us,gain_us,per_boundary_us,pct_saved" > "${OUT}"
for K in ${KS}; do
  b2b=(); unr=()
  for ((r=0; r<REPS; r++)); do b2b+=("$(run_elapsed "${K}" 1)"); done
  for ((r=0; r<REPS; r++)); do unr+=("$(run_elapsed 1 "${K}")"); done
  mb=$(median "${b2b[@]}"); mu=$(median "${unr[@]}")
  [ -z "${mb}" ] || [ -z "${mu}" ] && { echo "K=${K} FAILED (b2b='${b2b[*]}' unr='${unr[*]}')"; continue; }
  line=$(awk -v k="${K}" -v b="${mb}" -v u="${mu}" 'BEGIN{
    g=b-u; pb=(k>1)?g/(k-1):0; pct=(b>0)?100.0*g/b:0;
    printf "%d,%d,%d,%d,%.2f,%.1f", k, b, u, g, pb, pct}')
  echo "${line}" >> "${OUT}"
  echo "K=${K}: b2b=${mb}us (${b2b[*]}) unroll=${mu}us (${unr[*]}) -> ${line}"
done
echo "UNROLL_VS_B2B_COMPLETE -> ${OUT}"
