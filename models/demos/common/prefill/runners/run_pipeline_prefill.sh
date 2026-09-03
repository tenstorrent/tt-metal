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
# This launcher is model-agnostic (it only execs the common prefill_runner entry point). The
# rank-binding YAMLs + mesh-graph descriptors are topology config (model-agnostic; the model is
# selected by the binding's PREFILL_MANIFEST) and live at
# models/demos/common/prefill/runners/topology_configuration/. Pass your binding as $1.
#
# Examples:
#   # 2-galaxy D2D pipeline (connected MGD, FABRIC_2D):
#   ./run_pipeline_prefill.sh models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_rank_binding_2rank_d2d.yaml bh-glx-d07u02:1,bh-glx-d07u08:1
#
#   # 4-galaxy D2D pipeline (ring-chain host order — see the 4-galaxy connected MGD):
#   ./run_pipeline_prefill.sh models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_rank_binding_4rank_d2d.yaml bh-glx-d07u02:1,bh-glx-d07u08:1,bh-glx-d08u08:1,bh-glx-d08u02:1
#
#   # single-galaxy 1-rank full-model de-risk:
#   ./run_pipeline_prefill.sh models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_real_1galaxy_1rank.yaml bh-glx-d07u02:1
set -euo pipefail

RANK_BINDING="${1:?usage: run_pipeline_prefill.sh <rank_binding.yaml> [host_list] [tcp_iface]}"
HOST_LIST="${2:-bh-glx-d03u02:1,bh-glx-d03u08:1}"
TCP_IFACE="${3:-ens5f0np0}"

# TT_METAL_HOME = the tt-metal tree this script lives in
# (models/demos/common/prefill/runners -> 5 levels up).
TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME"
# Per-host LOCAL JIT cache. A shared (NFS) TT_METAL_CACHE makes both hosts write the same generated
# kernel files (defines_generated.h, ...) concurrently on a cold cache -> "Stale file handle" compile
# failures. /tmp is per-host, so each rank compiles into its own dir. ttrun auto-propagates TT_* vars.
export TT_METAL_CACHE="${PP_TT_METAL_CACHE:-/tmp/tt-metal-cache-pp}"
cd "$TT_METAL_HOME"

# -x PATH/LD_LIBRARY_PATH: ttrun only forwards TT_*/ARCH_*/... prefixed vars, not PATH, so peer ranks
# would otherwise resolve a bare `python3` to the system interpreter (no ttnn). Forwarding the launch
# host's PATH works only because every host's venv sits at the identical clone path.
exec python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface "$TCP_IFACE" \
  --rank-binding "$RANK_BINDING" \
  --mpi-args "--host ${HOST_LIST} --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH" \
  python3 -m models.demos.common.prefill.runners.prefill_runner
