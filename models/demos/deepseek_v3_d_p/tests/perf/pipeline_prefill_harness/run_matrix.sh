#!/usr/bin/env bash
# E2E prefill matrix: single-rank (SP8xTP4) vs PP=4x(8,1), TTFT and throughput, four ISLs.
#
# Both configs go through the SAME runner + producer driver, so TOPOLOGY IS THE ONLY VARIABLE.
# That is deliberate: comparing this harness's numbers against a different harness's is exactly how
# the earlier PP4 comparison went wrong.
#
# CHUNK_SIZE is fixed at the production 5,120 and ISL is varied by CHUNK COUNT, because that is what a
# serving system actually does with a longer prompt. So ISL 25,600 = 5 chunks, not one 25,600 push.
#
# Two modes per cell:
#   ttft  - ONE request: latency = rank0 first chunk start -> last rank last chunk end.
#           MEASURED WITH KV_ONLY_LAST_LAYER=1, i.e. the final norm + LM head do NOT run and no first
#           token is emitted, so this is PREFILL-COMPLETION LATENCY, not literally TTFT. That is not a
#           choice, it is forced: emitting a token makes the last rank do an event sync, and
#           `TT_FATAL: Event Synchronization is not supported during trace capture`
#           (fd_mesh_command_queue.cpp:932) -- so traced + token is impossible. Running it eager
#           instead would cost ~1.35x (eager vs traced measured at window 5,120) and would no longer
#           be comparable to the traced throughput cells, which is the whole point of this matrix.
#           MODES=tokentail measures the missing LM-head tail separately, eager, so it can be added on.
#   thru  - many requests back to back, last rank's steady chunk-to-chunk interval.
# They are separate runs because they need different settings: one request cannot show steady state,
# and a multi-request run's tokens/wall is a stream throughput, not one prefill's latency.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/env.sh"
cd ${TT_METAL_HOME} || exit 1
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-${TT_METAL_HOME}/mistral4_perf_$(hostname)}"
mkdir -p "$OUT"
TOPO=models/demos/common/prefill/runners/topology_configuration
# The [8,1] column->device map is PER-GALAXY, so prefer a binding generated for this host by
# gen_pipeline_binding.py. Falling back to the checked-in template on a different galaxy would not error --
# it would build stages that are not columns and quietly report wrong numbers.
pick_binding(){ local b="$1"; local h="${b%.yaml}.$(hostname).yaml"; [ -f "$h" ] && echo "$h" || echo "$b"; }

# config -> ranks | binding | weight-cache root
declare -A BIND=(
  [1rank]="$TOPO/pipeline_prefill_request_1rank.yaml"
  [pp4]="$(pick_binding "$TOPO/pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.yaml")"
)
declare -A RANKS=( [1rank]=1 [pp4]=4 )
declare -A CACHE=(
  [1rank]=${M4_CACHE_8x4}
  [pp4]=${M4_CACHE_8x1}
)
# ISL -> chunks (of 5,120) | cache width | throughput request count (keeps total chunks ~50-100)
# 40,960 (8 chunks) is the 2026-09-15 campaign's break-even bracket cell; the rest are unchanged.
declare -A CHUNKS=(  [5120]=1     [25600]=5     [40960]=8     [102400]=20    [261120]=51    )
declare -A WIDTH=(   [5120]=10240 [25600]=30720 [40960]=46080 [102400]=122880 [261120]=261120 )
declare -A NREQ=(    [5120]=48    [25600]=10    [40960]=6     [102400]=3     [261120]=2     )

CONFIGS=${CONFIGS:-"1rank pp4"}
ISLS=${ISLS:-"5120 25600 102400 261120"}
MODES=${MODES:-"ttft thru"}

for cfg in $CONFIGS; do
 for isl in $ISLS; do
  for mode in $MODES; do
    tag="${cfg}_${isl}_${mode}"
    if [ -s "$OUT/$tag/runner.log" ] && [ -z "${FORCE:-}" ]; then
      echo "[matrix] skip $tag (log exists; FORCE=1 to redo)"; continue
    fi
    trace=1
    case "$mode" in
      ttft)      req=1; users=1; kvonly=1 ;;
      # PROCESS-WARM latency: two requests in ONE process, report the second. Trace capture and
      # first-chunk compile then fall outside the measured request, which `ttft` cannot achieve --
      # running that cell twice does not help, because each run is a fresh process and re-captures.
      # Requires the E2E_CLOCK_V2 metric fix on the branch; read with analyze_prefill_latency_v2.py.
      lat)       req=2; users=1; kvonly=1 ;;
      # The LM-head/norm tail, eager (the only way it can run at all), with and without the token.
      # The DIFFERENCE of the two is the tail cost; neither number is meaningful on its own.
      tokentail) req=1; users=1; kvonly=0; trace=0 ;;
      notail)    req=1; users=1; kvonly=1; trace=0 ;;
      *)         req=${NREQ[$isl]}; users=2; kvonly=1 ;;
    esac
    # A single long request needs only one slot; two slots at 261,120 would double the KV budget.
    [ "$isl" -ge 102400 ] && users=1
    echo "[matrix] === $tag  (ranks=${RANKS[$cfg]} chunks=${CHUNKS[$isl]} width=${WIDTH[$isl]} req=$req users=$users kv_only=$kvonly trace=$trace) $(date -Is)"
    RUN_TAG="$tag" \
    PP_BINDING="${BIND[$cfg]}" PP_RANKS="${RANKS[$cfg]}" PP_TTNN_CACHE="${CACHE[$cfg]}" \
    PP_REQUESTS="$req" PP_USERS="$users" PP_CHUNKS="${CHUNKS[$isl]}" \
    PP_MAX_SEQ_LEN="${WIDTH[$isl]}" PP_KV_ONLY_LAST_LAYER="$kvonly" \
    PREFILL_USE_TRACE="$trace" \
      "$S/run_pp4_model.sh" > "$OUT/${tag}.driver.log" 2>&1
    rc=$?
    # The driver writes logs under its own scratch; collect them next to the driver log.
    mkdir -p "$OUT/$tag"
    cp "$S/logs/$tag/runner.log" "$S/logs/$tag/producer.log" "$OUT/$tag/" 2>/dev/null
    echo "[matrix] $tag rc=$rc"
    if [ "$rc" != "0" ]; then
      echo "[matrix] --- tail of driver log ---"; tail -15 "$OUT/${tag}.driver.log"
    fi
    # A hard failure can leave the fabric un-mappable, and on SOME galaxies `tt-smi -r` cannot fix it
    # (CPLD FW < 1.16 -> the tool itself says use -glx_reset, which needs sudo). If the board is left
    # broken, every remaining cell fails with a confusing topology-mapper error and the whole matrix is
    # wasted. So verify health after a failure and ABORT rather than cascade.
    if [ "$rc" != "0" ]; then
      # `/proc` is mounted hidepid=invisible on these boxes, so `ps` can only ever show your OWN
      # processes. The previous guard tested `$1 != $USER`, which zero rows can satisfy -- OTHER was
      # always 0, this branch was unreachable, and the script reset UNCONDITIONALLY on any cell
      # failure. Measured 2026-09-15: it fired twice into another user's live 4-rank run.
      #
      # Attribution is exactly what hidepid denies, so check the SHAPE of the driver's pid files
      # instead -- you do not need to know whose run it is to know not to reset. Idle baseline is
      # ONE pid listed on all 32 chips (the root tt_telemetry_collector); a live 4-rank galaxy run
      # is four distinct pids at 8 chips each. Also refuse if a per-run prefill descriptor belongs
      # to someone else: those are mode 600 and created/removed per run.
      #
      # Do NOT test TT_UMD_LOCK.* -- they are mode 666, shared, and persist across runs, so they
      # sit there owned by whoever opened a chip first and would block every legitimate reset.
      # (Verified: locks dated 19:19 owned by another user, while our own cell ran fine at 20:35.)
      # Count, never branch on a pipeline's exit status. `grep -qv` on EMPTY input was observed
      # returning 0 in one shell and 1 in another, which makes an exit-code guard non-deterministic
      # -- and an exit-code guard that could never be true is what the original defect WAS.
      OTHER=0
      DISTINCT=$(for d in /proc/driver/tenstorrent/[0-9]*; do cat "$d/pids" 2>/dev/null; done \
        | sort -u | awk 'NF{c++} END{print c+0}')
      if [ "${DISTINCT:-0}" -gt 1 ]; then OTHER=1; fi
      FOREIGN=$(ls -l /dev/shm 2>/dev/null \
        | awk -v me="$(id -un)" '$3!=me && $9 ~ /tt_prefill_|tt_h2d_stream_service/ {c++} END{print c+0}')
      if [ "${FOREIGN:-0}" -gt 0 ]; then OTHER=1; fi
      echo "[matrix] reset guard: distinct chip holders=$DISTINCT foreign prefill descriptors=$FOREIGN -> OTHER=$OTHER"
      if [ "${OTHER:-0}" = "0" ]; then
        # -glx_reset is the one that actually recovers this box: its CPLD is below v1.16, so tt-smi -r
        # is effectively a no-op here (the tool says so itself and points at -glx_reset). It takes
        # ~90s. -r is kept only as a fallback for boxes where -glx_reset is unavailable.
        echo "[matrix] recovering with tt-smi -glx_reset (~90s)"
        (cd /tmp && timeout 300 tt-smi -glx_reset >/dev/null 2>&1) || \
          { echo "[matrix] -glx_reset failed; falling back to -r"; (cd /tmp && tt-smi -r >/dev/null 2>&1); }
      else
        echo "[matrix] NOT resetting: another user's job is live"
      fi
      if ! "$S/check_board.sh"; then
        echo "[matrix] ABORT: the fabric can no longer map an 8x4 mesh and tt-smi -r did not fix it."
        echo "[matrix] Run:  sudo tt-smi -glx_reset    then re-run this script (completed cells are skipped)."
        exit 2
      fi
      echo "[matrix] board still healthy; continuing"
    fi
  done
 done
done
echo "[matrix] done -> $OUT"
