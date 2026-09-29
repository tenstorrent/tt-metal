#!/usr/bin/env bash
set -euo pipefail

RANK_BINDING="${1:?usage: run_pipeline_prefill.sh <rank_binding.yaml> [host_list] [tcp_iface]}"
HOST_LIST="${2:-bh-glx-d03u02:1,bh-glx-d03u08:1}"
TCP_IFACE="${3:-ens5f0np0}"

TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "$TT_METAL_HOME/python_env/bin/activate" ] && source "$TT_METAL_HOME/python_env/bin/activate"
export TT_METAL_CACHE="${PP_TT_METAL_CACHE:-/tmp/tt-metal-cache-pp}"
cd "$TT_METAL_HOME"

FWD_ENV=""
[ -n "${PREFILL_MANIFEST:-}" ] && FWD_ENV="${FWD_ENV} -x PREFILL_MANIFEST"
[ -n "${PREFILL_MODEL:-}" ] && FWD_ENV="${FWD_ENV} -x PREFILL_MODEL"
[ -n "${PREFILL_MOCK_MIGRATION:-}" ] && FWD_ENV="${FWD_ENV} -x PREFILL_MOCK_MIGRATION"

exec python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface "$TCP_IFACE" \
  --rank-binding "$RANK_BINDING" \
  --mpi-args "--host ${HOST_LIST} --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH${FWD_ENV}" \
  -- python3 -m models.demos.common.prefill.runners.prefill_runner
