#!/usr/bin/env bash
# Launch the single-galaxy PP4 probe under tt-run: 4 ranks on THIS host, 8 chips each.
#
# Mirrors run_pipeline_prefill.sh, with the one difference that matters: the host list is
# "<this host>:4" (four slots on one machine) instead of one slot per galaxy, because all four
# pipeline stages live on the same 32-chip galaxy.
#
# Usage: run_probe_pp4.sh [mesh|d2d|model] [tcp_iface]
#   mesh  — just open the four 8-chip meshes (fast; isolates the carve + fabric)
#   d2d   — additionally build the D2D endpoints and walk a tensor 0->1->2->3
#   model — build the real Mistral transformer per stage with random weights and run a chunk
set -euo pipefail

STAGE="${1:-mesh}"
TCP_IFACE="${2:-ens5f0np0}"

case "$STAGE" in
  mesh|d2d) TARGET="mistral4_bringup/pp4/probe_pp4.py" ;;
  model)    TARGET="mistral4_bringup/pp4/probe_pp4_model.py" ;;
  *) echo "unknown stage '$STAGE' (want mesh|d2d|model)" >&2; exit 2 ;;
esac

TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "$TT_METAL_HOME/python_env/bin/activate" ] && source "$TT_METAL_HOME/python_env/bin/activate"
# Per-host local JIT cache, as in the prefill launcher: four ranks compiling concurrently into a
# shared NFS cache race on the generated kernel headers.
export TT_METAL_CACHE="${PP_TT_METAL_CACHE:-/tmp/tt-metal-cache-pp4}"
cd "$TT_METAL_HOME"

export PROBE_STAGE="$STAGE"
HOSTS="$(hostname):4"

echo "[run_probe_pp4] stage=$STAGE hosts=$HOSTS iface=$TCP_IFACE"

# PROBE_* is not one of ttrun's auto-forwarded prefixes (TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_), so it
# needs an explicit -x. PREFILL_* likewise is not forwarded — those live in the binding's global_env.
exec python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface "$TCP_IFACE" \
  --rank-binding mistral4_bringup/pp4/pp4_single_galaxy_rank_bindings.yaml \
  --mpi-args "--host ${HOSTS} --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PROBE_STAGE -x PROBE_NUM_LAYERS -x PROBE_N_EXPERTS" \
  -- python3 "$TARGET"
