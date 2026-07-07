#!/usr/bin/env bash
# Per-rank launch for the connected-mesh leg: rank0 under tracy -r (ops CSV), others plain (idle).
RANK="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${TTRUN_RANK:-0}}}"
PY=/home/ppopovic/tt-metal/python_env/bin/python
export PYTHONPATH="/home/ppopovic/tt-metal:/home/ppopovic/tt-metal/models/demos/common/prefill/fabric_debug"
if [ "$RANK" = "0" ]; then
  OUT="${WORST_AG_OUT:-/data/ppopovic/prof_out/worst_ag_pipe}"; mkdir -p "$OUT"
  echo "[wrapper] rank0 -> tracy -r -o $OUT -m worst_allgather_test"
  exec "$PY" -m tracy -r -o "$OUT" -m worst_allgather_test
else
  echo "[wrapper] rank$RANK -> plain (idle, holds fabric)"
  exec "$PY" -m worst_allgather_test
fi
