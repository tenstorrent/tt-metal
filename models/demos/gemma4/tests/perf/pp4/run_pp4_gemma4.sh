#!/usr/bin/env bash
# Gemma 4 31B, 60 layers, real PP=4 over device-to-device fabric sockets: four ranks on [8,1]
# column sub-meshes of one Blackhole galaxy, each its own process, driven by prefill_producer.
#
# Working material, not a merge proposal: it hardcodes one site's paths. The topology config, the
# adapter/runtime changes and run_stage.py are the parts meant to survive.
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-/data/kmabee/tt-metal-2}"
export PYTHONPATH="$TT_METAL_HOME"
export HF_HOME="${HF_HOME:-/data/kmabee/hf_cache}"
export HF_MODEL="${HF_MODEL:-google/gemma-4-31B-it}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-/data/kmabee/hf_cache/tt_cache/google--gemma-4-31B-it}"
# Per-USER as well as per-host: this is the JIT kernel cache and the harness WRITES to it, so a
# fixed shared /tmp path belongs to whoever ran first and everyone after gets EACCES -- from a path
# that appears in no env file.
export TT_METAL_CACHE="${TT_METAL_CACHE:-/tmp/tt-metal-cache-pp-$(id -un)}"
cd "$TT_METAL_HOME"

RUN_TAG="${RUN_TAG:-pp4}"
OUT="${OUT:-$S/logs/$RUN_TAG}"
mkdir -p "$OUT"

SERVICE_ID=gemma4_prefill
DESCRIPTOR=/dev/shm/tt_h2d_stream_service_${SERVICE_ID}.bin
HOST="$(hostname)"
BINDING="${PP_BINDING:-models/demos/common/prefill/runners/topology_configuration/gemma4_pipeline_prefill_4rank_8x1.$HOST.yaml}"
[ -f "$BINDING" ] || { echo "[driver] FAIL: no per-host binding $BINDING; run gen_gemma4_pp4_binding.py"; exit 2; }

CHUNK="${PP_CHUNK_SIZE:-8192}"
MAX_SEQ="${PP_MAX_SEQ_LEN:-32768}"
CHUNKS="${PP_CHUNKS:-$((MAX_SEQ / CHUNK))}"
REQUESTS="${PP_REQUESTS:-1}"
LAYER_COUNTS="${PP_LAYER_COUNTS:-}"
USERS="${PP_USERS:-2}"

# PREFILL_MAX_SEQ_LEN and friends live in the binding's global_env, which ttrun applies to the
# child env and which therefore BEATS -x. Vary them by rewriting a copy of the binding, not by
# exporting -- an exported value would be silently ignored while the driver reported it.
cp "$BINDING" "$OUT/binding.yaml"
sed -i "s|PREFILL_MAX_SEQ_LEN: \"[0-9]*\"|PREFILL_MAX_SEQ_LEN: \"$MAX_SEQ\"|" "$OUT/binding.yaml"
sed -i "s|PREFILL_CHUNK_SIZE: \"[0-9]*\"|PREFILL_CHUNK_SIZE: \"$CHUNK\"|" "$OUT/binding.yaml"
[ -n "$LAYER_COUNTS" ] && sed -i "s|PREFILL_PP_LAYER_COUNTS: \"[0-9,]*\"|PREFILL_PP_LAYER_COUNTS: \"$LAYER_COUNTS\"|" "$OUT/binding.yaml"
# Runner AND producer must agree on the slot count: the runner asserts 0 <= slot_id < num_users, so
# a producer driving more slots than the runner allocated dies mid-run on slot_id out of range. And
# the count changes the WORKLOAD, not just the capacity -- with 2 slots the producer interleaves two
# independent 256k requests, which is a different thing from one request end to end.
sed -i "s|PREFILL_NUM_USERS: \"[0-9]*\"|PREFILL_NUM_USERS: \"$USERS\"|" "$OUT/binding.yaml"
BINDING="$OUT/binding.yaml"
echo "[driver] binding=$BINDING max_seq=$MAX_SEQ chunk=$CHUNK chunks=$CHUNKS requests=$REQUESTS users=$USERS counts=$(grep -o 'PREFILL_PP_LAYER_COUNTS: "[0-9,]*"' "$BINDING")"

# Per-rank chunk timings for analyze_pp.py.
export PREFILL_TIMING_DIR="$OUT/timing"
rm -rf "$PREFILL_TIMING_DIR"; mkdir -p "$PREFILL_TIMING_DIR"

# A stale descriptor from a prior run makes the readiness poll pass before THIS runner is up.
rm -f "$DESCRIPTOR"

echo "[driver] launching 4-rank runner under tt-run ($(date -Is))"
# ttrun forwards only TT_/ARCH_/TTNN_ automatically. HF_HOME and PYTHONPATH are neither, and a rank
# that cannot find the checkpoint fails 60 s in with a confusing config error, so list them.
setsid python3 ttnn/ttnn/distributed/ttrun.py \
  --rank-binding "$BINDING" \
  --mpi-args "--host $HOST:${PP_RANKS:-4} --map-by slot --bind-to none --tag-output --allow-run-as-root \
              -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x HF_HOME -x HF_MODEL -x TT_CACHE_PATH \
              -x TT_METAL_HOME -x TT_METAL_CACHE -x PREFILL_TIMING_DIR" \
  -- python3 -m models.demos.common.prefill.runners.prefill_runner \
  > "$OUT/runner.log" 2>&1 &
RUNNER_PGID=$!
echo "$RUNNER_PGID" > "$OUT/runner.pgid"
echo "[driver] runner pgid=$RUNNER_PGID; waiting for H2D descriptor $DESCRIPTOR"

# Readiness: rank 0 publishes the H2D descriptor once it is serving. Migration is off, so there is
# no KV table or device map to wait on and the descriptor is the only gate.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-3000}"
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
while [ ! -e "$DESCRIPTOR" ]; do
  if ! kill -0 "$RUNNER_PGID" 2>/dev/null; then
    echo "[driver] FAIL: runner exited during startup; tail:"; tail -60 "$OUT/runner.log"; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[driver] FAIL: runner not ready within ${READY_TIMEOUT_S}s; tail:"; tail -60 "$OUT/runner.log"; exit 1
  fi
  sleep 5
done
echo "[driver] runner ready ($(date -Is)); starting producer"

PREFILL_MODEL=gemma4_31b \
PREFILL_SP=8 PREFILL_TP=1 \
PREFILL_NUM_LAYERS=60 \
PREFILL_MAX_SEQ_LEN="$MAX_SEQ" \
PREFILL_CHUNK_SIZE="$CHUNK" \
PREFILL_NUM_USERS="$USERS" \
PREFILL_H2D_SERVICE_ID=$SERVICE_ID \
PREFILL_PRODUCER_SYNTHETIC_TOKENS=1 \
PREFILL_PRODUCER_CHECK_PCC=0 \
PREFILL_PRODUCER_CHUNKS="$CHUNKS" \
PREFILL_PRODUCER_MAX_REQUESTS="$REQUESTS" \
PREFILL_PRODUCER_P_GAP=0 \
PREFILL_PRODUCER_P_BURST=0 \
PREFILL_SEND_SHUTDOWN=1 \
  timeout "${PRODUCER_TIMEOUT_S:-2400}" python3 -m models.demos.common.prefill.runners.prefill_producer \
  > "$OUT/producer.log" 2>&1
PROD_RC=$?
echo "[driver] producer rc=$PROD_RC ($(date -Is))"

# The shutdown sentinel drains the pipeline and every rank exits on its own.
for i in $(seq 1 72); do kill -0 "$RUNNER_PGID" 2>/dev/null || break; sleep 5; done
if kill -0 "$RUNNER_PGID" 2>/dev/null; then
  echo "[driver] runner still up after the sentinel; terminating"
  kill -INT -"$RUNNER_PGID" 2>/dev/null || kill -INT "$RUNNER_PGID" 2>/dev/null || true
  sleep 20
  kill -9 -"$RUNNER_PGID" 2>/dev/null || kill -9 "$RUNNER_PGID" 2>/dev/null || true
fi

# The producer pushes into a socket and exits 0 whether or not the ranks survived, so its rc alone
# CANNOT see a rank-side crash -- a dead run would otherwise be recorded as a complete cell. Gate on
# the runner log too. Python-level failures only: rank teardown legitimately logs TT_FATAL from the
# D2D stream-service destructors after the device is closed, so grepping TT_FATAL fails every
# healthy run instead.
RUNNER_RC=0
if grep -qE 'Traceback \(most recent call last\)|AssertionError|Segmentation fault' "$OUT/runner.log" 2>/dev/null; then
  RUNNER_RC=3
  echo "[driver] FAIL: runner.log contains a rank-level failure; this cell's numbers are not usable:"
  grep -hoE '(AssertionError|RuntimeError|KeyError|ValueError|TypeError|Segmentation fault)[^\n]*' "$OUT/runner.log" \
    | sed 's/\x1b\[[0-9;]*m//g' | sort -u | head -6 | sed 's/^/[driver]   /'
fi

echo "[driver] done; logs in $OUT (producer rc=$PROD_RC runner rc=$RUNNER_RC)"
[ "$PROD_RC" != "0" ] && exit "$PROD_RC"
exit $RUNNER_RC
