#!/usr/bin/env bash
# 2-galaxy all_gather + D2D + all_gather probe, launched as a pytest under tt-run.
#
#   rank 0 = RANK0_HOST (mesh 0): high_bw_all_gather -> D2D send
#   rank 1 = RANK1_HOST (mesh 1): D2D recv          -> high_bw_all_gather
#
# Reuses the connected-2-galaxy MGD + FABRIC_2D via the existing D2D rank binding (topology only; the
# test sets fabric itself from device_params). ttrun wraps mpirun around `pytest` — the same launch
# shape the dual-galaxy CCL suite uses. Requires Pavle's high_bw_all_gather (PR #51134) in the build.
#
#   RANK0_HOST=bh-glx-110-c10u02 RANK1_HOST=bh-glx-110-c10u08 ./run_allgather_d2d_probe.sh
set -euo pipefail

RANK0_HOST="${RANK0_HOST:-bh-glx-110-c10u02}"
RANK1_HOST="${RANK1_HOST:-bh-glx-110-c10u08}"
TCP_IFACE="${TCP_IFACE:-ens5f0np0}"

# TT_METAL_HOME = the tt-metal tree this script lives in (runners dir -> 5 levels up).
TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME"
# Per-host local JIT cache (/tmp), same rationale as run_pipeline_prefill.sh.
export TT_METAL_CACHE="${PP_TT_METAL_CACHE:-/tmp/tt-metal-cache-probe-${USER}}"
cd "$TT_METAL_HOME"
source python_env/bin/activate

BINDING="models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_rank_binding_2rank_d2d.yaml"
TEST="tests/ttnn/unit_tests/operations/experimental/deepseek_prefill/test_allgather_d2d_probe.py"

exec python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface "$TCP_IFACE" \
  --rank-binding "$BINDING" \
  --mpi-args "--host ${RANK0_HOST}:1,${RANK1_HOST}:1 --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH" \
  pytest -svv "$TEST"
