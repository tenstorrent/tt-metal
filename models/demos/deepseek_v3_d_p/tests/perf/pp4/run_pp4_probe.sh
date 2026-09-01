#!/usr/bin/env bash
# Launch the 4-rank [8,1] topology/D2D probe under tt-run with the new rank binding.
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/env.sh"
cd "$TT_METAL_HOME"
source "$TT_METAL_HOME/python_env/bin/activate"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/tmp/tt-metal-cache-pp}"
export PREFILL_MANIFEST=$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tt/runners/manifests/mistral4.json
# Prefer the host-specific binding: the [8,1] column->device map is per-galaxy and a wrong map does
# not error, it just builds stages that are not columns.
B0=models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.yaml
BINDING="${B0%.yaml}.$(hostname).yaml"; [ -f "$BINDING" ] || BINDING="$B0"
echo "[probe] binding=$BINDING"
exec python3 ttnn/ttnn/distributed/ttrun.py \
  --rank-binding "$BINDING" \
  --mpi-args "--host $(hostname):4 --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x MISTRAL4_HF_MODEL -x PREFILL_HF_MODEL -x PREFILL_MANIFEST -x PREFILL_MODEL -x PREFILL_TTNN_CACHE" \
  -- python3 "$S/probe_pp4_d2d.py"
