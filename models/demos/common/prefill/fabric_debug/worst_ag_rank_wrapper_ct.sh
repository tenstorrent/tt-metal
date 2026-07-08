#!/usr/bin/env bash
RANK="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}"
PY=/home/ppopovic/tt-metal/python_env/bin/python
export PYTHONPATH="/home/ppopovic/tt-metal:/home/ppopovic/tt-metal/models/demos/common/prefill/fabric_debug"
CTDIR=/data/ppopovic/prof_out/ct_connected
if [ "$WORST_AG_CT_MODE" = "capture" ]; then
  # capture must NOT run under tracy (router instrumentation conflicts); per-rank logs dir -> separate yaml
  export TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE=1 TT_METAL_FABRIC_TRIMMING_PRESERVE_VC0_FORWARDING=1
  export TT_METAL_LOGS_PATH="$CTDIR/rank$RANK"
  mkdir -p "$TT_METAL_LOGS_PATH"
  echo "[wrapper] rank$RANK CAPTURE -> $TT_METAL_LOGS_PATH/generated/reports/channel_trimming_capture.yaml"
  exec "$PY" -m worst_allgather_test
else
  # apply: each rank loads ITS OWN captured profile; rank0 under tracy for the ops CSV
  export TT_METAL_FABRIC_TRIMMING_PROFILE="$CTDIR/rank$RANK/generated/reports/channel_trimming_capture.yaml"
  if [ "$RANK" = "0" ]; then
    OUT="${WORST_AG_OUT:-/data/ppopovic/prof_out/worst_ag_pipe_trim}"; mkdir -p "$OUT"
    echo "[wrapper] rank0 APPLY profile=$TT_METAL_FABRIC_TRIMMING_PROFILE -> tracy"
    exec "$PY" -m tracy -r -o "$OUT" -m worst_allgather_test
  else
    echo "[wrapper] rank$RANK APPLY profile=$TT_METAL_FABRIC_TRIMMING_PROFILE plain"
    exec "$PY" -m worst_allgather_test
  fi
fi
