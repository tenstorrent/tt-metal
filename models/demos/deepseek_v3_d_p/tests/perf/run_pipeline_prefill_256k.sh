#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Mistral Small 4 PP=4 x (8,1) prefill at 261,120 tokens -- the 256K throughput headline in
# docs/MISTRAL4_PP4_VS_SINGLE_RANK.md, as one runnable cell.
#
# One 8x4 Blackhole galaxy carved into 4 Z-connected column sub-meshes, one process per stage,
# traced, driven by prefill_producer over a real MeshSocket at every hop. 51 chunks x 5,120.
#
# There is no pytest for PP=4: the runner is driven by shell under ttrun with a 4-rank binding.
# This is deliberately the minimum that reproduces the number. The full campaign -- 16 cells, both
# latency modes, Tracy captures, board recovery -- is on the PP=4 reference branch.
set -euo pipefail

T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$T/../../../../.." && pwd)}"
export PYTHONPATH="$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/lib:${LD_LIBRARY_PATH:-}"
cd "$TT_METAL_HOME"

# The harness got this from its env.sh, which is not in-tree. Without it a bare python3 has no ttnn.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$TT_METAL_HOME/python_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$TT_METAL_HOME/python_env/bin/activate"
fi
python3 -c 'import ttnn' 2>/dev/null || { echo "[repro] FAIL: python3 cannot import ttnn -- build the venv (./create_venv.sh) or activate it"; exit 1; }

# --- artifacts: multi-GB, not in the repo, so both must be pointed at your copies ----------------
export MISTRAL4_HF_MODEL="${MISTRAL4_HF_MODEL:?set to the Mistral-Small-4-119B checkpoint}"
export PREFILL_HF_MODEL="${PREFILL_HF_MODEL:-$MISTRAL4_HF_MODEL}"
# Resolved as {name}_{arch}_{num_devices}dev/{sp}x{tp}. Each rank sees 8 devices -> 8dev/8x1. Cache
# keys are GLOBAL layer indices, so all four ranks share one directory and it must hold layers 0..35.
export PREFILL_TTNN_CACHE="${PREFILL_TTNN_CACHE:?set to the 8x1 TTNN weight cache}"
# JIT kernel cache, and this run WRITES to it. Per-user: a fixed shared path belongs to whoever ran
# first, and the EACCES surfaces as signal 6 with the real cause buried in runner.log.
export TT_METAL_CACHE="${TT_METAL_CACHE:-/tmp/tt-metal-cache-pp-$(id -un)}"

export PREFILL_MANIFEST="$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tt/runners/manifests/mistral4.json"
export PREFILL_CHUNK_SIZE=5120        # analyze_prefill_throughput.py assumes this
export PREFILL_NUM_LAYERS=36          # not auto-propagated by ttrun; must be in the -x list below
export PREFILL_NUM_USERS=1            # one long request: 2 slots would double the KV budget here
export PREFILL_KV_ONLY_LAST_LAYER=1   # throughput, not TTFT -- no final norm/LM head, no token
export PREFILL_USE_TRACE=1
export LOGURU_LEVEL=INFO

# The headline is 51 chunks x 5,120 = 261,120. Both are overridable ONLY to smoke-test the plumbing
# on a few chunks before spending an hour of galaxy on the real thing -- override either and the
# result is no longer the published number.
CHUNKS="${CHUNKS:-51}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-261120}"
REQUESTS="${REQUESTS:-2}"
OUT="${OUT:-$TT_METAL_HOME/mistral4_pp4_256k_$(hostname)}"
mkdir -p "$OUT"

TOPO=models/demos/common/prefill/runners/topology_configuration
BASE="$TOPO/pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.yaml"
# The column -> device map is PER-GALAXY and a wrong one does NOT error: it builds stages that are
# not columns and reports plausible wrong numbers. gen_pipeline_binding.py writes <name>.<host>.yaml.
if [ -f "${BASE%.yaml}.$(hostname).yaml" ]; then
  BASE="${BASE%.yaml}.$(hostname).yaml"
else
  echo "[repro] WARNING: no binding for $(hostname); using the checked-in map, read off bh-glx-b03u02."
  echo "[repro]          Generate yours first: python3 $T/gen_pipeline_binding.py"
fi
# PREFILL_MAX_SEQ_LEN lives in the binding's global_env, which ttrun applies over -x, so the window
# has to be widened by rewriting a copy rather than exporting.
BINDING="$OUT/binding.yaml"
sed "s|PREFILL_MAX_SEQ_LEN: \"[0-9]*\"|PREFILL_MAX_SEQ_LEN: \"$MAX_SEQ_LEN\"|" "$BASE" > "$BINDING"

DESCRIPTOR=/dev/shm/tt_h2d_stream_service_ds_prefill.bin
rm -f "$DESCRIPTOR"   # a stale one makes the readiness poll pass before this runner is up

echo "[repro] launching 4 ranks traced ($(date -Is)); logs in $OUT"
setsid python3 ttnn/ttnn/distributed/ttrun.py \
  --rank-binding "$BINDING" \
  --mpi-args "--host $(hostname):4 --map-by slot --bind-to none --tag-output --allow-run-as-root \
              -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x MISTRAL4_HF_MODEL -x PREFILL_HF_MODEL \
              -x PREFILL_MANIFEST -x PREFILL_TTNN_CACHE -x PREFILL_CHUNK_SIZE -x PREFILL_NUM_USERS \
              -x PREFILL_USE_TRACE -x PREFILL_KV_ONLY_LAST_LAYER -x PREFILL_NUM_LAYERS" \
  -- python3 -m models.demos.common.prefill.runners.prefill_runner \
  > "$OUT/runner.log" 2>&1 &
PGID=$!

# Rank 0 publishes the H2D descriptor once it is serving. No migration here, so that is the only gate.
deadline=$(( $(date +%s) + ${READY_TIMEOUT_S:-3000} ))
while [ ! -e "$DESCRIPTOR" ]; do
  if ! kill -0 "$PGID" 2>/dev/null; then
    echo "[repro] FAIL: runner exited during startup; tail:"; tail -40 "$OUT/runner.log"; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[repro] FAIL: runner not ready within ${READY_TIMEOUT_S:-3000}s; tail:"; tail -40 "$OUT/runner.log"; exit 1
  fi
  sleep 5
done
echo "[repro] runner ready ($(date -Is)); starting producer"

PROD_RC=0
PREFILL_MAX_SEQ_LEN=$MAX_SEQ_LEN \
PREFILL_H2D_SERVICE_ID=ds_prefill \
PREFILL_PRODUCER_CHECK_PCC=0 \
PREFILL_PRODUCER_CHUNKS=$CHUNKS \
PREFILL_PRODUCER_MAX_REQUESTS=$REQUESTS \
PREFILL_PRODUCER_INTERLEAVE=round_robin \
PREFILL_SEND_SHUTDOWN=1 \
  timeout "${PRODUCER_TIMEOUT_S:-5400}" python3 -m models.demos.common.prefill.runners.prefill_producer \
  > "$OUT/producer.log" 2>&1 || PROD_RC=$?

# The shutdown sentinel drains the pipeline and every rank exits on its own.
for _ in $(seq 1 60); do kill -0 "$PGID" 2>/dev/null || break; sleep 5; done
if kill -0 "$PGID" 2>/dev/null; then
  echo "[repro] runner still up after the sentinel; terminating"
  kill -INT -"$PGID" 2>/dev/null || true; sleep 20; kill -9 -"$PGID" 2>/dev/null || true
fi

# The producer pushes chunks into a socket and exits 0 whether or not the ranks survived, so its rc
# alone CANNOT see a runner-side crash -- a dead run would otherwise look like a passing one. Gate on
# the log too, but on Python-level failures only: rank teardown legitimately logs TT_FATAL from the
# D2D stream-service destructors after the device is closed.
RUNNER_RC=0
if grep -qE 'Traceback \(most recent call last\)|AssertionError' "$OUT/runner.log" 2>/dev/null; then
  RUNNER_RC=3
  echo "[repro] FAIL: rank-level Python failure; this run's numbers are not usable:"
  grep -hoE '(AssertionError|RuntimeError|KeyError|ValueError|TypeError)[^\n]*' "$OUT/runner.log" \
    | sed 's/\x1b\[[0-9;]*m//g' | sort -u | head -5 | sed 's/^/[repro]   /'
fi

echo "[repro] producer rc=$PROD_RC runner rc=$RUNNER_RC ($(date -Is))"
if [ "$PROD_RC" != 0 ] || [ "$RUNNER_RC" != 0 ]; then exit 1; fi

# Warm-up 8, not the analyzer's default of 4: a multi-chunk interval grows as the KV cache deepens,
# so every published table in the docs uses 8 and a 4 does not compare.
echo "[repro] --- steady-state throughput ---"
python3 "$T/analyze_prefill_throughput.py" "$OUT/runner.log" 8
