#!/usr/bin/env bash
# Shared driver functions for the Kimi prefill perf overnight orchestrator.
# Sourced by orchestrator.sh. Do not run directly.

REPO=/home/ppopovic/tt-metal
INVDIR=/home/ppopovic/kimi_perf_overnight
QUEUE="$INVDIR/queue"
DONE="$INVDIR/done"
LOGS="$INVDIR/logs"
RESULTS="$INVDIR/RESULTS.md"
MASTER="$INVDIR/orchestrator.log"

TRACE=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok

# Kimi test env (the runner does NOT need these; the pytest path does).
KIMI_ENV=(
  KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized
  TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
  TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden
  TT_DS_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
  TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

# Wait until $1 (file) contains regex $2, up to $3 seconds. Returns 1 on timeout.
wait_for() {
  local f="$1" pat="$2" to="$3" waited=0
  while ! grep -Eq -- "$pat" "$f" 2>/dev/null; do
    sleep 3; waited=$((waited + 3))
    [ -f "$INVDIR/STOP" ] && return 2
    [ "$waited" -ge "$to" ] && return 1
  done
  return 0
}

count_for() { grep -Ec -- "$2" "$1" 2>/dev/null || echo 0; }

# Kill any lingering device-holding processes and wait for the mesh to free.
# Matches the module path so it never hits the orchestrator/grep itself.
cleanup_procs() {
  local pat pids
  for pat in "runners.prefill_runner" "runners.prefill_h2d_producer"; do
    pids=$(pgrep -f "$pat" || true)
    [ -n "$pids" ] && { log "cleanup: SIGTERM $pat -> $pids"; kill -TERM $pids 2>/dev/null; }
  done
  sleep 6
  for pat in "runners.prefill_runner" "runners.prefill_h2d_producer"; do
    pids=$(pgrep -f "$pat" || true)
    [ -n "$pids" ] && { log "cleanup: SIGKILL $pat -> $pids"; kill -KILL $pids 2>/dev/null; }
  done
  # Wait until the processes are FULLY GONE (not just zombie). A multithreaded zombie (Zl) keeps the
  # sysmem NOC mapping + chip lock until init reaps ALL its threads — observed to take up to ~6 min.
  # Relaunching while one lingers crashes the new runner with "sysmem mapped at ..." in start_driver.
  local waited=0
  while :; do
    local remaining
    remaining=$(pgrep -f "runners.prefill_runner|runners.prefill_h2d_producer" || true)
    [ -z "$remaining" ] && break
    log "cleanup: waiting for full reap of: $remaining"
    sleep 5; waited=$((waited + 5))
    [ "$waited" -ge 420 ] && { log "cleanup: WARNING procs not reaped after 420s: $remaining"; break; }
  done
  sleep 8  # settle: let the kernel release the sysmem NOC address space before the next device open
  # Remove stale POSIX shm the runner creates with O_CREAT|O_EXCL and can't re-create after a SIGKILL
  # (the layer-ack channel + h2d socket). Scoped to our service id so we never touch a co-user's shm.
  rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_h2d_stream_service_ds_prefill* 2>/dev/null
}

record_fail() {
  { echo; echo "### $1 — FAILED ($(date '+%F %T'))"; echo "reason: $2"; } >> "$RESULTS"
}

record_runner_result() {
  local name="$1" rlog="$2"
  {
    echo; echo "### $name (runner) — $(date '+%F %T')"
    echo 'per-iter pipeline.prefill() ms:'
    echo '```'
    grep -E 'pipeline.prefill\(\) =' "$rlog" | sed -E 's/.*pipeline.prefill\(\) = //' || echo '(none)'
    echo '```'
    if grep -q 'section-timing' "$rlog"; then
      echo 'section-timing (last 12 chunks):'; echo '```'; grep 'section-timing' "$rlog" | tail -12; echo '```'
    fi
    if grep -q 'construction-dump' "$rlog"; then
      echo 'construction-dump:'; echo '```'; grep -A7 'construction-dump' "$rlog" | head -12; echo '```'
    fi
    if grep -q 'input-dump' "$rlog"; then
      echo 'input-dump:'; echo '```'; grep -A4 'input-dump' "$rlog" | head -8; echo '```'
    fi
  } >> "$RESULTS"
}

record_standalone_result() {
  local name="$1" rlog="$2"
  {
    echo; echo "### $name (standalone-chunked) — $(date '+%F %T')"
    echo 'total + per-chunk timing:'
    echo '```'
    grep -E 'standalone-chunked\].*(prefilled in|prefilled chunk)' "$rlog" || echo '(none)'
    echo '```'
    if grep -q 'section-timing' "$rlog"; then
      echo 'section-timing (last 12 chunks):'; echo '```'; grep 'section-timing' "$rlog" | tail -12; echo '```'
    fi
    if grep -q 'construction-dump' "$rlog"; then
      echo 'construction-dump:'; echo '```'; grep -A7 'construction-dump' "$rlog" | head -12; echo '```'
    fi
    if grep -qE 'KV cache PCC|kv.*pcc|PCC' "$rlog"; then
      echo 'PCC (record-only):'; echo '```'; grep -E 'PCC' "$rlog" | tail -8; echo '```'
    fi
  } >> "$RESULTS"
}

# run_standalone NAME "EXTRA_ENV"
# Runs the runner in PREFILL_STANDALONE_CHUNKED mode: NO producer, NO socket, NO ack channel,
# and (critically) NO request-mode clear_loaded_sub_device_manager at prefill_runner.py:578.
# Reads tokens from the trace metadata.json itself, prefills 11 chunks, logs total ms, then exits.
run_standalone() {
  local name="$1" extra="$2"
  local rlog="$LOGS/$name.runner.log"
  log "=== [$name] STANDALONE-CHUNKED start (extra='$extra') ==="
  cleanup_procs
  ( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
      -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE \
      PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=61 \
      PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
      PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill \
      PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=11 \
      PREFILL_STANDALONE_CHUNKED_SLOT=0 PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 \
      PREFILL_PREFILL_SYNC=1 \
      DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
      $extra \
      python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner ) > "$rlog" 2>&1 &
  local pid=$!
  log "[$name] standalone pid=$pid; waiting for 'chunks prefilled in' (<=1500s)"
  wait_for "$rlog" "standalone-chunked\] [0-9]+ chunks prefilled in" 1500
  case $? in
    1) log "[$name] FAIL standalone timeout"; tail -8 "$rlog" | tee -a "$MASTER"; cleanup_procs; record_fail "$name" "standalone timeout"; return 1 ;;
    2) log "[$name] STOP during standalone"; cleanup_procs; return 1 ;;
  esac
  log "[$name] standalone done; giving PCC ~60s then tearing down"
  # The per-layer PCC check runs after the timing line; we only need the timing, so don't block long.
  local waited=0
  while pgrep -f "runners.prefill_runner" >/dev/null 2>&1 && [ "$waited" -lt 60 ]; do sleep 5; waited=$((waited+5)); done
  cleanup_procs
  record_standalone_result "$name" "$rlog"
  log "=== [$name] STANDALONE-CHUNKED done ==="
}

record_test_result() {
  local name="$1" tlog="$2"
  {
    echo; echo "### $name (test) — $(date '+%F %T')"
    echo 'pytest outcome:'; echo '```'; grep -E 'PASSED|FAILED|ERROR|passed|failed|error' "$tlog" | tail -5 || echo '(none)'; echo '```'
    if grep -q 'section-timing' "$tlog"; then
      echo 'section-timing (last 12 chunks):'; echo '```'; grep 'section-timing' "$tlog" | tail -12; echo '```'
    fi
    if grep -q 'construction-dump' "$tlog"; then
      echo 'construction-dump:'; echo '```'; grep -A7 'construction-dump' "$tlog" | head -12; echo '```'
    fi
    if grep -q 'input-dump' "$tlog"; then
      echo 'input-dump:'; echo '```'; grep -A4 'input-dump' "$tlog" | head -8; echo '```'
    fi
    # Any per-chunk timing the test itself prints.
    if grep -qE 'forward_chunk.*ms|chunk.*= .* ms' "$tlog"; then
      echo 'test per-chunk timing (tail):'; echo '```'; grep -E 'forward_chunk.*ms|chunk.*= .* ms' "$tlog" | tail -15; echo '```'
    fi
  } >> "$RESULTS"
}

# run_runner NAME "RUNNER_EXTRA_ENV" "PRODUCER_EXTRA_ENV"
run_runner() {
  local name="$1" rextra="$2" pextra="$3"
  local rlog="$LOGS/$name.runner.log" plog="$LOGS/$name.producer.log"
  log "=== [$name] RUNNER start (rextra='$rextra' pextra='$pextra') ==="
  cleanup_procs

  ( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
      -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE -u PREFILL_STANDALONE_CHUNKED \
      PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=61 \
      PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
      PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_STANDALONE_CHUNKED_SLOT=0 \
      $rextra \
      python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner ) > "$rlog" 2>&1 &
  local rpid=$!
  log "[$name] runner pid=$rpid; waiting for 'exported descriptor' (<=900s)"
  wait_for "$rlog" "exported descriptor" 900
  case $? in
    1) log "[$name] FAIL runner descriptor timeout"; tail -8 "$rlog" | tee -a "$MASTER"; cleanup_procs; record_fail "$name" "runner descriptor timeout"; return 1 ;;
    2) log "[$name] STOP during runner init"; cleanup_procs; return 1 ;;
  esac
  log "[$name] runner ready; launching producer"

  ( cd "$REPO" && env PREFILL_REQUEST_LOOP_PCC=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=11 PREFILL_STANDALONE_CHUNKED_SLOT=0 \
      DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
      PREFILL_SP=8 PREFILL_TP=4 PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
      PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_H2D_CONNECT_TIMEOUT=180 \
      PREFILL_STANDALONE_ITERS=1 \
      $pextra \
      python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer ) > "$plog" 2>&1 &
  local ppid=$!
  log "[$name] producer pid=$ppid; waiting for '[producer] done.' (<=600s)"
  wait_for "$plog" "\[producer\] done\." 600
  if [ $? -ne 0 ]; then
    log "[$name] WARN producer did not log done; tail:"; tail -8 "$plog" | tee -a "$MASTER"
  fi

  # Wait for the 11 per-iter prefill lines the runner logs as it consumes the pushed chunks.
  local waited=0
  while [ "$(count_for "$rlog" 'pipeline.prefill\(\) =')" -lt 11 ]; do
    sleep 3; waited=$((waited + 3)); [ "$waited" -ge 240 ] && break
  done
  log "[$name] runner logged $(count_for "$rlog" 'pipeline.prefill\(\) =') prefill lines; tearing down"
  cleanup_procs
  record_runner_result "$name" "$rlog"
  log "=== [$name] RUNNER done ==="
}

# run_test NAME "TEST_EXTRA_ENV"
run_test() {
  local name="$1" textra="$2"
  local tlog="$LOGS/$name.test.log"
  log "=== [$name] TEST start (textra='$textra') ==="
  cleanup_procs
  ( cd "$REPO" && env "${KIMI_ENV[@]}" $textra \
      python -m pytest \
        "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc" \
        -k "L61 and chunks11 and iters20 and kimi and mesh-8x4" -s --timeout=0 ) > "$tlog" 2>&1
  log "[$name] pytest exit=$?"
  cleanup_procs
  record_test_result "$name" "$tlog"
  log "=== [$name] TEST done ==="
}
