#!/bin/bash
# tt-metal: MiniMax-M3 media splice vs HF reference on one galaxy (--topo galaxy --mode single)
#
# test_media_splice_vs_ref.py's post-splice comparison at 8x4. The p100a run covers (1,1), which
# cannot exercise either the TP-sharded residual across 4 TP ranks or the 8-chip SP row rotation;
# parametrize_mesh_with_fabric auto-filters to the shapes that fit, so -k 8x4 selects only the
# galaxy case here.
#
# Accuracy only: no latency is read off this run, so the quad's 8 kW cap does not matter.
set -euo pipefail

export TT_METAL_HOME="$WORKER_METAL_HOME"
export TT_METAL_RUNTIME_ROOT="$WORKER_METAL_HOME"
export LD_LIBRARY_PATH="$WORKER_METAL_HOME/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$WORKER_METAL_HOME"

# The HF comparison imports transformers.models.minimax_m3_vl and skips without it. 5.12.1 is the
# CI pin; the tree's python_env has 5.10, so the run needs --pip transformers==5.12.1.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MGD="$WORKER_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
TARGET="$WORKER_METAL_HOME/models/demos/minimax_m3/tests/unit/test_media_splice_vs_ref.py"

cd "$TTRUN_CWD"
python3 "$TTRUN_PY" \
  --skip-executable-check \
  --tcp-interface ens5f0np0 \
  --mesh-graph-descriptor "$MGD" \
  --hosts "$RESOLVED_HOSTS" \
  --force-rediscovery \
  --mpi-args "--bind-to none --tag-output --wdir $WORKER_METAL_HOME --output-filename $RANKLOG_DIR -x PYTHONPATH -x TT_METAL_HOME -x TT_METAL_RUNTIME_ROOT -x TT_METAL_CACHE -x LD_LIBRARY_PATH -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE" \
  pytest -svv -p no:benchmark "$TARGET" -k "8x4"
