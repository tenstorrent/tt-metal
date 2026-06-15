#!/usr/bin/env bash
# Profile the 3-layer pytest (test_prefill_3l_profile.py) WITH or WITHOUT the H2D stream service,
# capturing only the WARM chunk (the compile pass is excluded by the parser, which slices to the LAST
# forward_chunk_layer_* signpost occurrence). Produces:
#   - 3L_test_<mode>_log.new   per-op / per-layer breakdown (same format as 9L_with_service_sdpafix_log.new)
#   - prints total_op_kernel_time + total_op2op_gap (raw AND negatives-clamped-to-0)
#   - tracy_3Ltest_<mode>.tracy   raw tracy capture for manual inspection (open in the Tracy profiler GUI)
#
# Usage: profile_3L_test.sh <noservice|service> [tag]
# Env overrides: PREFILL_PROFILE_KV (default 51200), PREFILL_PROFILE_ITERS (2), PREFILL_PROFILE_NUM_LAYERS (3),
#                PSC (profiler program-support count, default 3000), PROFDIR (default /dev/shm/ttprof3l).
#
# Direct (no-script) equivalent of the run, for reference:
#   TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=3000 TT_METAL_PROFILER_DIR=/dev/shm/ttprof3l \
#   KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized \
#   TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
#   TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden \
#   TT_DS_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
#   TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden \
#   PREFILL_PROFILE_NUM_LAYERS=3 PREFILL_PROFILE_KV=51200 PREFILL_PROFILE_ITERS=2 \
#   python -m tracy -r -p --disable-device-data-push-to-tracy -m pytest -svv \
#     'models/demos/deepseek_v3_d_p/tests/test_prefill_3l_profile.py::test_kimi_prefill_3l_profile[blackhole-kimi-service-mesh-8x4]'
set -u
MODE="${1:?usage: profile_3L_test.sh noservice|service [tag]}"
case "$MODE" in noservice|service) :;; *) echo "MODE must be noservice|service"; exit 2;; esac
# Select by EXACT node id (single token, no spaces) — robust through the tracy->pytest arg forwarding
# (a multi-word `-k "service and not noservice"` gets word-split there). "service" is a substring of
# "noservice", so an exact id is also the only unambiguous selector for the service case.
NODEID="blackhole-kimi-${MODE}-mesh-8x4"
TAG="${2:-3Ltest_$MODE}"
REPO=/home/ppopovic/tt-metal
INV=$REPO/kimi_perf_overnight
PROFDIR="${PROFDIR:-/dev/shm/ttprof3l}"; mkdir -p "$PROFDIR"
LOGD="$PROFDIR/.logs"; REPORTS="$PROFDIR/reports"
LOG="$INV/profile_${TAG}.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Reap only python test procs (NOT this bash script) to avoid self-kill; wait for full reap (sysmem hold).
reap(){
  me=$$
  for p in $(pgrep -x python; pgrep -x python3); do
    [ "$p" = "$me" ] && continue
    if tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q "test_prefill_3l_profile"; then kill -KILL "$p" 2>/dev/null; fi
  done
  while pgrep -f "test_prefill_3l_profile" | grep -qv "^$$\$" >/dev/null 2>&1; do
    alive=0
    for p in $(pgrep -x python; pgrep -x python3); do
      tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q "test_prefill_3l_profile" && alive=1
    done
    [ "$alive" = 0 ] && break; sleep 3
  done
  rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_d2h_* /dev/shm/tt_h2d_stream_service_ds_prefill* 2>/dev/null; sleep 2
}
purge_raw(){ rm -f "$LOGD/profile_log_device.csv" "$LOGD/tracy_ops_times.csv" "$LOGD/tracy_profile_log_host.tracy" 2>/dev/null; }

PSC="${PSC:-3000}"
log "######## profile 3L pytest MODE=$MODE tag=$TAG START; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
reap; purge_raw

# disk guard: kill the run if free space drops under 0.8G
( while true; do f=$(df --output=avail "$REPO"|tail -1);
    if [ "$f" -lt 838860 ]; then echo "[disk-guard] <0.8G ($((f/1024))M) KILL"|tee -a "$LOG";
      for p in $(pgrep -x python; pgrep -x python3); do tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q test_prefill_3l_profile && kill -KILL "$p" 2>/dev/null; done; break; fi
    echo "[disk-watch] $((f/1024))M free" >> "$LOG"; sleep 4; done ) & GUARD=$!

( cd "$REPO" && \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=$PSC TT_METAL_PROFILER_DIR=$PROFDIR \
  KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized \
  TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
  TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden \
  TT_DS_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
  TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden \
  PREFILL_PROFILE_NUM_LAYERS="${PREFILL_PROFILE_NUM_LAYERS:-3}" \
  PREFILL_PROFILE_KV="${PREFILL_PROFILE_KV:-51200}" \
  PREFILL_PROFILE_ITERS="${PREFILL_PROFILE_ITERS:-2}" \
  python -m tracy -r -p --disable-device-data-push-to-tracy -m pytest -svv \
    "models/demos/deepseek_v3_d_p/tests/test_prefill_3l_profile.py::test_kimi_prefill_3l_profile[${NODEID}]" \
) > "$INV/profile_${TAG}.runner.log" 2>&1
RC=$?
log "$TAG exit=$RC ; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free"
kill "$GUARD" 2>/dev/null

echo "  marker-drop warnings: $(grep -c 'markers were dropped' "$INV/profile_${TAG}.runner.log")" | tee -a "$LOG"
# raw tracy for manual inspection (Tracy GUI)
if [ -f "$LOGD/tracy_profile_log_host.tracy" ]; then
  cp -f "$LOGD/tracy_profile_log_host.tracy" "$INV/tracy_${TAG}.tracy"; log "saved raw tracy -> tracy_${TAG}.tracy"
else
  log "WARN no tracy_profile_log_host.tracy in $LOGD"
fi
# ops report -> per-op log + totals
latest=$(ls -dt "$REPORTS"/*/ 2>/dev/null | head -1)
rep=$(ls "$latest"ops_perf_results_*.csv 2>/dev/null | head -1)
if [ -n "$rep" ] && [ -f "$rep" ]; then
  cp -f "$rep" "$INV/ops_${TAG}.csv"; log "saved ops_${TAG}.csv ($(wc -l < "$INV/ops_${TAG}.csv") rows)"
  python3 "$INV/parse_NL_detail.py" "$INV/ops_${TAG}.csv" "3L $MODE (warm, kv=${PREFILL_PROFILE_KV:-51200})" "$INV/3L_test_${MODE}_log.new" 2>&1 | tee -a "$LOG"
  echo "------------------------------------------------------------" | tee -a "$LOG"
  python3 "$INV/op_totals.py" "$INV/ops_${TAG}.csv" "3L $MODE (warm, kv=${PREFILL_PROFILE_KV:-51200})" 2>&1 | tee -a "$LOG"
  echo "------------------------------------------------------------" | tee -a "$LOG"
else
  log "WARN no ops_perf_results CSV for '$TAG' (rc=$RC) — check profile_${TAG}.runner.log"
fi
[ -n "$latest" ] && rm -rf "$latest" 2>/dev/null
purge_raw; reap
log "######## $TAG DONE; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
