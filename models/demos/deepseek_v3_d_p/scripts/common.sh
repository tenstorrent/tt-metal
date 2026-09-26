#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Shared config + helpers for the DeepSeek-V3 prefill stress scripts.
# Sourced (not executed) by stress.sh / watch.sh / tail.sh / watch_multiple_dirs.sh.
#
# Common positional args: <log_name> [loop_count]

TT_METAL_HOME="${TT_METAL_HOME:-/data/$USER/tt-metal}"

# Make sure TT_METAL_HOME is on PYTHONPATH (only add it if not already present).
case ":${PYTHONPATH:-}:" in
  *":$TT_METAL_HOME:"*) ;;
  *) export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

LOG_NAME="${1:-deepseek_v3_d_p_log}"
# Loop count: prefer an explicit LOOP env var (used by watch_multiple_dirs.sh,
# whose positional args are all log names), else the positional [loop_count].
LOOP="${LOOP:-${2:-20}}"

# Per-run logs: one log_NN under here per outer iteration.
LOG_DIR="/data/$USER/$LOG_NAME"

# Test selection — single source of truth. All models run the same chunked no-PCC file; MODEL picks
# the test function, the parametrize ids, and the model's env vars (each adapter uses its own env var
# names — see tt/runners/adapters/).
TEST_FILE="$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py"

# Shared parametrize ids — same for every model, override per run if you want a different point.
MESH_ID="${MESH_ID:-torus-xy-8x4}"
PRELOAD_ID="${PRELOAD_ID:-preload0}"
CHUNKS_ID="${CHUNKS_ID:-chunks20}"
ITERS_ID="${ITERS_ID:-iters20}"
# The use_trace axis, on both models' tests: "notrace" replays nothing, "traced" captures the chunk
# forward once and replays it per chunk.
TRACE_ID="${TRACE_ID:-notrace}"
# Kimi only: the perf test's perf_margin axis. The single param is None — "take the band from the
# mode" (TRACED_PERF_MARGIN / UNTRACED_PERF_MARGIN, resolved by kimi_chunked_perf_gate) — so its id
# is "margin_auto". It was "margin5pct" while the param was the DEFAULT_PERF_MARGIN literal; keep
# this in step with the ids= on the test's perf_margin parametrize, or the node id stops collecting.
MARGIN_ID="${MARGIN_ID:-margin_auto}"

# Per-model: test function, node-id variant token(s), num_layers id, node-id suffix, env vars.
# ENV_VARS must set the model's TTNN cache var: the weight_cache_path fixture (tests/conftest.py)
# reads variant.ttnn_cache_env and, when it is unset, silently builds a fresh cache under the HF model
# dir instead of failing — so it must be set here per MODEL rather than exported once in your shell.
case "${MODEL:-}" in
  KIMI_K2_7)
    TEST_FUNC="test_kimi_prefill_transformer_chunked_perf"
    VARIANT_ID="kimi_k2_7"; LAYERS_ID="L61"; NODE_SUFFIX="-$MARGIN_ID-$TRACE_ID"
    ENV_VARS='KIMI_K2_7_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K2.7-Code-dequantized TT_KIMI_PREFILL_TTNN_CACHE=/mnt/weka/model-cache/scratch/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill PREFILL_TRACE_DIR=/mnt/weka/model-cache/scratch/deepseek-ai/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320'
    ;;
  GLM5_2)
    TEST_FUNC="test_glm_prefill_transformer_chunked_no_pcc"
    # The GLM test has no perf_margin param, so its node id has no margin token.
    VARIANT_ID="glm52"; LAYERS_ID="L78"; NODE_SUFFIX="-$TRACE_ID"
    ENV_VARS='GLM52_HF_MODEL=/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8 TT_GLM52_PREFILL_TTNN_CACHE=/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k'
    ;;
  *)
    echo "ERROR: set MODEL to one of: KIMI_K2_7 | GLM5_2  (got '${MODEL:-<unset>}')" >&2
    echo "  e.g.  export MODEL=KIMI_K2_7" >&2
    # Sourced, so return; the || exit covers the case where this file is executed directly.
    return 1 2>/dev/null || exit 1
    ;;
esac

# Shared env, prepended to every model's vars. LOGURU_LEVEL=INFO pins the log level the same way the
# CI jobs do: DEBUG makes the run very chatty (every per-tensor "Loaded cache for ..." line) and slows
# the weight load, so it is not the default here. Override with LOGURU_LEVEL=DEBUG for a debug run.
# DS_PERF_IGNORE_POWER=1 drops the test's is_high_power() skipif; passed through only when set, so a
# high-power host keeps the guard.
ENV_VARS="LOGURU_LEVEL=${LOGURU_LEVEL:-INFO} ${DS_PERF_IGNORE_POWER:+DS_PERF_IGNORE_POWER=$DS_PERF_IGNORE_POWER }$ENV_VARS"

# Exact pytest node id. Preferred over -k, which cannot pin one point of a sweep this wide: a -k
# expression loose enough to type selects several rows and runs them all per iteration.
# "blackhole-" comes from the root conftest's silicon_arch_name param.
PYTEST_TARGET="$TEST_FILE::$TEST_FUNC[blackhole-$VARIANT_ID-$MESH_ID-$LAYERS_ID-$PRELOAD_ID-$CHUNKS_ID-$ITERS_ID$NODE_SUFFIX]"

# Inner-iteration count, derived from the num_iters id above (two of the ids spell the count out).
case "$ITERS_ID" in
  two_iters) INNER_ITERS=2 ;;
  ten_iters) INNER_ITERS=10 ;;
  *) INNER_ITERS=$(grep -oE '[0-9]+' <<<"$ITERS_ID" | head -1) ;;
esac

# Seconds without log growth before a still-running iteration is flagged STALE.
STALE_SECS="${STALE_SECS:-240}"

# Path of the Nth outer-iteration log (zero-padded): log_for 3 -> <dir>/log_03
log_for() { printf "%s/log_%02d" "$1" "$2"; }

# Epoch seconds of the FIRST line of <log> matching <pattern>, read off its loguru timestamp.
# Empty when the pattern has not been logged yet, which is how callers detect "phase not reached".
log_ts() {
  local line
  line=$(grep -m1 "$2" "$1" 2>/dev/null)
  [[ $line =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2}\ [0-9:]{8}) ]] && date -d "${BASH_REMATCH[1]}" +%s 2>/dev/null
}

# Seconds -> 1h02m03s / 49m24s / 13s
dur() {
  if [ "$1" -ge 3600 ]; then printf '%dh%02dm%02ds' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60))
  elif [ "$1" -ge 60 ]; then printf '%dm%02ds' $(($1 / 60)) $(($1 % 60))
  else printf '%ds' "$1"; fi
}

# "load 49m24s  fwd 9m13s" for <log> up to <end epoch>. Worth reading apart because the weight load
# dominates an outer iteration (measured 49m24s load vs 9m13s forward on a 61-layer traced Kimi run),
# so "this run is slow" is almost always "this load is slow" and the forward number is the one that
# actually tracks the thing under test. Everything before forward_layer_0_start is still load.
phase_split() {
  local start fwd
  start=$(log_ts "$1" 'Building TtPrefillTransformer')
  [ -z "$start" ] && return
  fwd=$(log_ts "$1" 'forward_layer_0_start')
  if [ -z "$fwd" ]; then
    printf 'load %s' "$(dur $(($2 - start)))"
  else
    printf 'load %s  fwd %s' "$(dur $((fwd - start)))" "$(dur $(($2 - fwd)))"
  fi
}

# Decode a shell exit status into a signal name. 128+N means "killed by signal N",
# and the distinction matters: 135 (SIGBUS) is a failed host mapping — the tt-kmd
# pin_user_pages failure mode — while 139 (SIGSEGV) or 134 (SIGABRT) point at the
# process itself. Anything under 128 is pytest's own status, not a signal.
sig_name() {
  local rc="$1"
  if [ "$rc" -lt 128 ]; then echo "exit $rc"; return; fi
  case $((rc - 128)) in
    6) echo "SIGABRT" ;;
    7) echo "SIGBUS" ;;
    9) echo "SIGKILL" ;;
    11) echo "SIGSEGV" ;;
    15) echo "SIGTERM" ;;
    *) echo "signal $((rc - 128))" ;;
  esac
}

# Host-state post-mortem for a failed iteration: crash_snapshot <dir> <n> <rc> <log>
# Writes <dir>/crash_NN.txt. Everything in here is readable without root, on purpose
# — the kernel ring buffer is off limits on these nodes, so the substitute is the
# state that precedes the driver's complaint: how many 1 GB hugepages are free (one
# per PCIe device is needed; a crashed run that has not released its pages leaves
# free < devices, and the NEXT run is the one that dies), the process's pinning
# limits, and anything still holding the devices.
crash_snapshot() {
  local dir="$1" n="$2" rc="$3" log="$4"
  local out ndev hp1g hp2m
  out=$(printf "%s/crash_%02d.txt" "$dir" "$n")
  hp1g=/sys/kernel/mm/hugepages/hugepages-1048576kB
  hp2m=/sys/kernel/mm/hugepages/hugepages-2048kB
  ndev=$(ls /sys/bus/pci/drivers/tenstorrent/ 2>/dev/null | grep -c '^0000:')

  {
    echo "=== crash snapshot: iteration $n on $(hostname -s) at $(date '+%F %T %Z')"
    echo "exit=$rc ($(sig_name "$rc"))   log=$log"
    echo
    echo "--- hugepages (1 GB pool: one page per PCIe device, $ndev bound)"
    printf "1G  nr=%s free=%s resv=%s surplus=%s\n" \
      "$(cat $hp1g/nr_hugepages 2>/dev/null)" "$(cat $hp1g/free_hugepages 2>/dev/null)" \
      "$(cat $hp1g/resv_hugepages 2>/dev/null)" "$(cat $hp1g/surplus_hugepages 2>/dev/null)"
    printf "2M  nr=%s free=%s\n" \
      "$(cat $hp2m/nr_hugepages 2>/dev/null)" "$(cat $hp2m/free_hugepages 2>/dev/null)"
    grep -E '^(MemTotal|MemFree|MemAvailable|Hugetlb):' /proc/meminfo
    echo "hugetlbfs files still present:"; ls -l /dev/hugepages-1G/ 2>&1 | tail -n +2
    echo
    echo "--- limits (shell)"
    # ulimit -l is kB and bash cannot print anything else, so convert: a raw
    # 74202924 reads like bytes and hides that this is ~71 GiB, i.e. not the limit.
    echo "memlock=$(awk -v k="$(ulimit -l)" 'BEGIN{printf "%.1f", k/1048576}') GiB  nofile=$(ulimit -n)  nproc=$(ulimit -u)"
    echo
    echo "--- processes still holding the devices"
    pgrep -af 'pytest|test_prefill' 2>/dev/null || echo "(none)"
    for p in $(pgrep -f 'pytest.*test_prefill_transformer_chunked' 2>/dev/null); do
      echo "pid $p:"
      grep -E '^(VmRSS|VmLck|VmPin):' "/proc/$p/status" 2>/dev/null
      grep -E 'Max locked memory|Max open files' "/proc/$p/limits" 2>/dev/null
    done
    echo
    echo "--- fatal error block from the log (if any)"
    grep -m1 -A6 'Fatal Python error' "$log" 2>/dev/null || echo "(no faulthandler dump)"
    echo
    echo "--- last 30 log lines"
    tail -30 "$log" 2>/dev/null
  } > "$out" 2>&1
}

# Scan one log dir over outer iterations 1..LOOP.
# Sets globals: pass fail crash hang running pending, and the `details` array.
scan_log_dir() {
  local dir="$1"
  pass=0; fail=0; crash=0; hang=0; running=0; pending=0
  details=()

  local i f next N iter layer mtime now idle elapsed progress split rc
  for i in $(seq 1 "$LOOP"); do
    f=$(log_for "$dir" "$i")
    next=$(log_for "$dir" $((i + 1)))
    N=$(printf "%02d" "$i")
    if [ ! -f "$f" ]; then
      ((pending++))
      continue
    fi
    mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    split=$(phase_split "$f" "$mtime")
    if grep -qE 'smoke test passed|Chunked prefill no-PCC run done|^=+.*1 passed' "$f" 2>/dev/null; then
      elapsed=$(grep -oE '[0-9]+\.[0-9]+s \([0-9:]+\)' "$f" | tail -1)
      details+=("  $N: PASS  $elapsed  $split")
      ((pass++))
    elif grep -qE '^=+.*(1 failed|1 error)' "$f" 2>/dev/null; then
      details+=("  $N: FAIL")
      ((fail++))
    elif rc=$(grep -oE 'TEST_DONE_EXIT=[0-9]+' "$f" 2>/dev/null | tail -1 | cut -d= -f2) &&
      [ -n "$rc" ] && [ "$rc" -ge 128 ]; then
      # Killed by a signal: pytest never printed a summary, so the PASS/FAIL greps
      # above both miss it and the mtime logic below would call it HANG?. See
      # crash_NN.txt in this dir for the host state at the time.
      details+=("  $N: CRASH  $(sig_name "$rc")")
      ((crash++))
    elif grep -q 'Fatal Python error' "$f" 2>/dev/null; then
      # Same thing caught mid-flight: the faulthandler dump is in the log but the
      # exit line has not been appended yet (or predates that change).
      details+=("  $N: CRASH  $(grep -m1 -oE 'Fatal Python error: .*' "$f")")
      ((crash++))
    else
      # Single-shot test logs "Starting iteration:"; the chunked no-PCC test logs
      # "iter N done (C chunks) in ...s" once per completed outer iteration.
      iter=$(grep -cE 'Starting iteration:|iter [0-9]+ done \([0-9]+ chunks\)' "$f" 2>/dev/null)
      layer=$(grep -oE 'forward_layer_[0-9]+_(start|end)' "$f" 2>/dev/null | tail -1)
      now=$(date +%s)
      idle=$((now - mtime))

      # Before the forward loop starts there are no forward_layer markers, and the run sits in the
      # weight load for most of its wall clock. tt_prefill_transformer logs "Building layer N/M" per
      # layer, which is the load-phase counterpart of forward_layer_N_end, so show it verbatim.
      if [ -z "$layer" ]; then
        progress=$(grep -oE 'Building layer [0-9]+/[0-9]+' "$f" 2>/dev/null | tail -1)
        [ -n "$progress" ] && progress="loading ${progress#Building }"
      else
        progress="$layer"
      fi

      if [ -f "$next" ]; then
        details+=("  $N: HANG?  iter=$iter/$INNER_ITERS  $progress  $split")
        ((hang++))
      elif [ "$idle" -gt "$STALE_SECS" ]; then
        details+=("  $N: STALE ${idle}s  iter=$iter/$INNER_ITERS  $progress  $split")
        ((running++))
      else
        details+=("  $N: RUN    iter=$iter/$INNER_ITERS  $progress  $split  (idle ${idle}s)")
        ((running++))
      fi
    fi
  done
}
