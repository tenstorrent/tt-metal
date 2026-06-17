#!/usr/bin/env bash
# Thin launcher for the pipeline-parallel prefill runner under tt-run.
#
# All real config (mesh, layer split, chunk count, transport, PCC, PREFILL_* env) lives in the
# rank-binding YAML's global_env — ttrun does NOT auto-propagate PREFILL_* from the shell, so
# per-run knobs belong there, not here. This script only captures the launch boilerplate: env
# exports, the host list / TCP interface, the activation-handoff dir cleanup, and the ttrun call.
#
# Usage:
#   run_pipeline_prefill.sh <rank_binding.yaml> [host_list] [tcp_iface]
#
#   <rank_binding.yaml>  path (relative to TT_METAL_HOME or absolute) to the tt-run rank binding.
#   [host_list]          mpirun --host value. Default: bh-glx-d03u02:1,bh-glx-d03u08:1 (2 galaxies).
#                        For a single-rank/one-galaxy binding pass e.g. bh-glx-d03u02:1.
#   [tcp_iface]          NIC for MPI TCP. Default: ens5f0np0 (the 10.32.24.x cluster net here).
#
# PREFILL_PP_DIR (the host->host activation handoff dir) is read from the YAML; this script clears
# it before launch so a previous run's files can't be mistaken for this one's. Override by exporting
# PP_DIR before invoking.
#
# Examples:
#   # 2-galaxy half-and-half split (rank0 u02 layers 0-30, rank1 u08 layers 31-60):
#   ./run_pipeline_prefill.sh models/demos/deepseek_v3_d_p/tt/runners/pipeline_prefill_rank_binding_2rank.yaml
#
#   # single-galaxy 1-rank full-model de-risk:
#   ./run_pipeline_prefill.sh models/demos/deepseek_v3_d_p/tt/runners/pipeline_prefill_real_1galaxy_1rank.yaml bh-glx-d03u02:1
set -euo pipefail

RANK_BINDING="${1:?usage: run_pipeline_prefill.sh <rank_binding.yaml> [host_list] [tcp_iface]}"
HOST_LIST="${2:-bh-glx-d03u02:1,bh-glx-d03u08:1}"
TCP_IFACE="${3:-ens5f0np0}"
PP_DIR="${PP_DIR:-/data/jjovicic/pp_acts}"

# TT_METAL_HOME = the tt-metal tree this script lives in
# (models/demos/deepseek_v3_d_p/tt/runners -> 5 levels up).
TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME"
# Per-host LOCAL JIT cache. A shared (NFS) TT_METAL_CACHE makes both hosts write the same generated
# kernel files (defines_generated.h, ...) concurrently on a cold cache -> "Stale file handle" compile
# failures. /tmp is per-host, so each rank compiles into its own dir. ttrun auto-propagates TT_* vars.
export TT_METAL_CACHE="${PP_TT_METAL_CACHE:-/tmp/tt-metal-cache-pp}"
cd "$TT_METAL_HOME"

# Fresh handoff dir so a stale activation file can never be read as this run's.
mkdir -p "$PP_DIR"
rm -f "$PP_DIR"/pp_act_* 2>/dev/null || true

exec python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface "$TCP_IFACE" \
  --rank-binding "$RANK_BINDING" \
  --mpi-args "--host ${HOST_LIST} --map-by slot --bind-to none --tag-output --allow-run-as-root" \
  python3 -m models.demos.deepseek_v3_d_p.tt.runners.pipeline_prefill_runner
