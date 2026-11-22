#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# [RING,RING] 4x4 bring-up diagnostics on an SC4 CI allocation, step by step; every step runs even if an earlier
# one failed, and prints its rank-log tails. Steps (select with RR_DIAG_STEPS="1 2 3"):
#   1  SC4 carve (8 tori, 4 seams over two hosts) through tt-run discovery: fabric init + ring all-gather on
#      both axes of every mesh (test_rr_prefill_mesh_allgather.py).
#   2  torus-seam-torus on two cabled hosts: D2D one service per host row block (test_rr_prefill_d2d_seam.py).
#   3  GLM-5.3 L5 one chunk on ONE two-host seam stage (test_prefill_transformer_chunked no-PCC, torus-xy-4x4-sp1).
set -uo pipefail
: "${TT_METAL_HOME:?}"
cd "${TT_METAL_HOME}"
W="${PREFILL_SUMMARIES:-/tmp}/rr_diag"
mkdir -p "${W}/ttrun-cwd"
TTRUN_DIR="${TTRUN_DIR:-/etc/ttop}"
TCP="${PREFILL_TCP_INTERFACE:-ens5f0np0}"
HOSTS=$(awk 'NF && $1 !~ /^#/ {printf "%s,", $1}' "${TTRUN_DIR}/hostfile" | sed 's/,$//')
IFS=, read -r -a HOST_LIST <<< "${HOSTS}"
TOPO="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration"
STEPS="${RR_DIAG_STEPS:-1 2 3}"
echo "===== rr_diag: hosts ${HOSTS} (from ${TTRUN_DIR}/hostfile); steps ${STEPS}"
ls "${TT_METAL_HOME}"/build*/test/tt_metal/tt_fabric/test_physical_discovery 2>/dev/null || echo "(no test_physical_discovery binary)"

ttrun() {  # name timeout_s hosts mgd -- command...
  local name=$1 tmo=$2 hosts=$3 mgd=$4; shift 4
  local logs="${W}/${name}"
  rm -rf "${logs}"; mkdir -p "${logs}"
  echo "===== [${name}] hosts=${hosts} mgd=$(basename "${mgd}")  $(date -u +%T)"
  ( cd "${W}/ttrun-cwd" && timeout -k 30 "${tmo}" python3 "${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
      --skip-executable-check --tcp-interface "${TCP}" --force-rediscovery \
      --mesh-graph-descriptor "${mgd}" --hosts "${hosts}" \
      --mpi-args "--bind-to none --tag-output --allow-run-as-root --wdir ${TT_METAL_HOME} --output-filename ${logs}/r -x PATH -x LD_LIBRARY_PATH" \
      bash -lc "cd '${TT_METAL_HOME}'; export PYTHONPATH='${TT_METAL_HOME}'; export PYTHONUNBUFFERED=1; $*" ) \
      > "${logs}/ttrun.log" 2>&1
  local rc=$?
  echo "===== [${name}] rc=${rc}  $(date -u +%T)  (124/137 = timeout/hang)"
  tail -n 40 "${logs}/ttrun.log" | cut -c1-300
  local rb; rb=$(ls -t "${W}"/ttrun-cwd/generated/ttrun/*/rank_bindings*.yaml 2>/dev/null | head -1)
  [ -n "${rb}" ] && { echo "--- rank binding tt-run solved: ${rb}"; grep -E 'rank:|mesh_id|mesh_host_rank|CLUSTER_ID|VISIBLE' "${rb}" | head -60; }
  for f in $(find "${logs}" -type f \( -name stdout -o -name stderr \) | sort); do
    if grep -qE 'Error|FATAL|Traceback|FAILED|PASSED|passed|failed|_OK' "${f}"; then
      echo "--- ${f#${logs}/}"; grep -E 'Error|FATAL|Traceback|FAILED|PASSED|passed|failed|_OK|assert' "${f}" | tail -n 12 | cut -c1-300
    fi
  done
  return ${rc}
}

if [[ " ${STEPS} " == *" 1 "* ]]; then
  ttrun step1_sc4_carve_allgather 900 "${HOSTS}" "${TOPO}/ci/sc4_4x4_rr_mgd.textproto" \
    "python3 -m pytest models/demos/deepseek_v3_d_p/tests/test_rr_prefill_mesh_allgather.py -sv --timeout=600"
fi

# Steps 2/3 need two cabled neighbours; tt-run discovery rejects a pair that is not, so try the ring pairs in order.
pair_run() {  # name timeout mgd -- command
  local name=$1 tmo=$2 mgd=$3; shift 3
  local n=${#HOST_LIST[@]}
  for i in $(seq 0 $((n - 1))); do
    local a=${HOST_LIST[$i]} b=${HOST_LIST[$(((i + 1) % n))]}
    ttrun "${name}_${i}" "${tmo}" "${a},${b}" "${mgd}" "$@" && return 0
    grep -qiE 'no valid|could not|failed to map|mapping|topology' "${W}/${name}_${i}/ttrun.log" || return 1
  done
  return 1
}

if [[ " ${STEPS} " == *" 2 "* ]]; then
  pair_run step2_seam_d2d 900 "${TOPO}/ci/sc2_4x4_rr_mgd.textproto" \
    "python3 -m pytest models/demos/deepseek_v3_d_p/tests/test_rr_prefill_d2d_seam.py -sv --timeout=600"
fi

if [[ " ${STEPS} " == *" 3 "* ]]; then
  pair_run step3_glm_seam_stage 1800 "${TOPO}/ci/sc2_4x4_rr_seam_mgd.textproto" \
    "export GLM53_HF_MODEL=/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4; \
     export TT_GLM53_PREFILL_TTNN_CACHE=/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/GLM-5.3-Cache-prefill; \
     export DS_PERF_IGNORE_POWER=1 SKIP_PERF_GATES=1 LOGURU_LEVEL=INFO; \
     python3 -m pytest 'models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked_no_pcc[blackhole-glm53-torus-xy-4x4-sp1-L5-preload0-chunks1-iters1-notrace]' -sv --timeout=1500"
fi
echo "===== rr_diag done $(date -u +%T)"
