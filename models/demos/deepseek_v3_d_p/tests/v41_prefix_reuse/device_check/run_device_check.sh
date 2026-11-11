#!/bin/bash
# Device check of V4.1 prefill-side prefix reuse H1 (V41_PREFILL_PREFIX_REUSE=1), run FROM HOST 29 against the prefill galaxy
# H (default 30) in a MAINTENANCE WINDOW: the serving stack's prefill runner on H must be DOWN (this refuses if any process
# holds H's chips or the serving runner's pid is alive). One model build of the production 512k runner (chunk 8192, MAXSEQ
# 524288, A3 + islands + warm-all + device embed + Engram RM + CP2 export), then the driver's phases:
#   R  cold references after an unrelated prompt; prompt A cold TWICE = determinism (export + model rows byte-identical)
#   S  reuse: A, then B (shares A's chunk 0) ... every digest == the cold reference of the same prompt
#   (U unmarked headers, T timing: add with PHASES=R,S,U,T)
# Verdicts: "[driver] VERDICT determinism ..." and "[driver] VERDICT reuse ..." (also in $OUT/check.log, results.json).
#
#   H=30 bash run_device_check.sh            (~25-40 min: build ~15-25 min incl. warm-all + islands, then ~10 min of prompts)
#   env: H, SRC (this worktree on 29), DST (its copy on H; NEVER the serving tree tt-metal-1017-serve), RUN (scratch dir on
#        H), OUT (results on 29), CHUNK, MAXSEQ, MAXPOS (the ring's CP2 positions), CKPTS, PHASES, TT_CACHE
#        H2 (the host chunk store; unset = H1 only, env.sh as before): STORE_GB, STORE_TIP_GB, RESERVE_GB, SPILL_TIPS --
#        see run_device_check_h2.sh
# It does not restart the serving stack: bring it back afterwards (v41ax_stack_up.sh / stack_up.sh).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=${H:-30}; P=10.82.97.$H
SRC=${SRC:-$(cd "$HERE/../../../../../.." && pwd)}
DST=${DST:-/home/ttuser/sdawle/tt-metal-1017-h1}
SERVING_TREE=/home/ttuser/sdawle/tt-metal-1017-serve
TM=/home/ttuser/sdawle/tt-metal-v41-main; POD=/mnt/tt-data/sdawle/pod_v41; U=$POD/ut
RUN=${RUN:-/home/ttuser/sdawle/prefill_runs/v41_prefix_check}
OUT=${OUT:-/home/ttuser/sdawle/v41ax_logs/prefix_check/$(date +%m%d_%H%M)}
CHUNK=${CHUNK:-8192}; MAXSEQ=${MAXSEQ:-524288}; MAXPOS=${MAXPOS:-520192}; CKPTS=${CKPTS:-16}; PHASES=${PHASES:-R,S}
TT_CACHE=${TT_CACHE:-/dev/shm/tt-metal-cache-v41p}   # the serving runner's JIT cache (same TT_METAL_HOME: same kernels)
SERVICE=dsv41_prefix_check; ETD=/dev/shm/dsv41_1017_engram; ROWS_SRV=$U/engram_row_server.py; ROWS_PORT=47011
SERVING_PID=/home/ttuser/sdawle/prefill_runs/k4/runner.pid
mkdir -p "$OUT"
log() { echo "[v41-prefix-check] $(date +%T) $*" | tee -a "$OUT/check.log"; }
R() { ssh -o BatchMode=yes -o ServerAliveInterval=30 -o LogLevel=ERROR "$1" "$2" < /dev/null; }

[ "$DST" != "$SERVING_TREE" ] || { log "DST must not be the serving tree $SERVING_TREE"; exit 1; }
[ -f "$SRC/models/demos/deepseek_v3_d_p/tt/v41/prefix_reuse.py" ] || { log "SRC $SRC has no tt/v41/prefix_reuse.py"; exit 1; }
holders=$(R $P "sudo -n fuser /dev/tenstorrent/* 2>/dev/null | tr ' ' '\n' | grep -o '[0-9]\+' | sort -u | tr '\n' ' '")
[ -z "$holders" ] || { log "host $H chips are held by pid(s) $holders -- not running (stop the serving stack first)"; exit 1; }
alive=$(R $P "[ -f $SERVING_PID ] && kill -0 \$(cat $SERVING_PID) 2>/dev/null && cat $SERVING_PID")
[ -z "$alive" ] || { log "the serving prefill runner (pid $alive) is alive on $H -- not running"; exit 1; }
free_gb=$(R $P "df -BG --output=avail / | tail -1 | tr -dc 0-9")
[ "${free_gb:-0}" -ge 10 ] || { log "host $H root disk has ${free_gb:-?} GB free (< 10) -- not running"; exit 1; }
log "host $H free (disk ${free_gb} GB); SRC $SRC @ $(git -C "$SRC" log --oneline -1 | cut -c1-60) ($(git -C "$SRC" status --short | wc -l) local change(s)); OUT $OUT"

# a session kept open for the whole check: a host with RemoveIPC wipes this user's /dev/shm (H2D service, ack channel) when
# the last ssh session ends (pod memory note, host 31 10-09)
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o LogLevel=ERROR $P "sleep 14400" < /dev/null > /dev/null 2>&1 &
KEEPER=$!

cleanup() {
  R $P "[ -f $RUN/runner.pid ] && kill -0 \$(cat $RUN/runner.pid) 2>/dev/null && { kill -KILL \$(cat $RUN/runner.pid); echo 'runner killed (pid file)'; }; rm -f /dev/shm/*${SERVICE}* /dev/shm/*v41_prefix_check_lcr* 2>/dev/null; true" | sed 's/^/  /' | tee -a "$OUT/check.log"
  rsync -a $P:$RUN/runner.log $P:$RUN/driver.log $P:$RUN/results.json $P:$RUN/stacks.txt $P:$RUN/env.sh "$OUT/" 2>/dev/null
  R $P "sudo -n fuser /dev/tenstorrent/* 2>/dev/null | wc -w" | sed 's/^/  chip holders on H after the check: /' | tee -a "$OUT/check.log"
  kill $KEEPER 2>/dev/null
  log "done; results in $OUT. The serving stack is NOT restarted by this script."
}
trap cleanup EXIT

log "sync $SRC -> $H:$DST (the serving tree $SERVING_TREE is not touched)"
rsync -a --delete --exclude .git --exclude '__pycache__' --exclude generated "$SRC/" $P:$DST/ || { log "sync failed"; exit 1; }
R $P "rm -rf $RUN && mkdir -p $RUN" || exit 1

# the Engram row servers (layer 1 on 31, layer 14 on 29), as prefill_up.sh starts them: reused when already up
for hl in 10.82.97.31:1 10.82.97.29:14; do
  h=${hl%%:*}; l=${hl#*:}; pf=/tmp/engram_rows_L$l.pid; df=/tmp/engram_rows_L$l.dir
  R $h "[ -f $ETD/engram_L$l.ok ] || { echo 'NO TABLE'; exit 0; }; if [ -f $pf ] && kill -0 \$(cat $pf) 2>/dev/null && [ \"\$(cat $df 2>/dev/null)\" = $ETD ]; then echo up; else (setsid nohup $TM/python_env/bin/python $ROWS_SRV --table-dir $ETD --layers $l --port $ROWS_PORT > /tmp/engram_rows_L$l.log 2>&1 < /dev/null & echo \$! > $pf); echo $ETD > $df; sleep 6; grep -q serving /tmp/engram_rows_L$l.log && echo started || { echo FAILED; tail -3 /tmp/engram_rows_L$l.log; }; fi" | sed "s/^/  rows L$l @ $h: /" | tee -a "$OUT/check.log"
done
grep -q "rows L.*\(FAILED\|NO TABLE\)" <(tail -2 "$OUT/check.log") && { log "Engram row servers not available (tables: engram_tables.py, tt-d-gen guide 3c)"; exit 1; }

log "reset $H only"
R $P "bash $U/reset_local.sh" 2>&1 | tail -2 | sed 's/^/  /' | tee -a "$OUT/check.log"

cat > "$OUT/env.sh" <<EOF
cd $DST; source $TM/python_env/bin/activate
export TT_METAL_HOME=$TM TT_METAL_RUNTIME_ROOT=$TM PYTHONPATH=$DST:$POD/e62_site:$POD/cpu_stub_m1017 TT_METAL_CACHE=$TT_CACHE LOGURU_LEVEL=INFO
export PREFILL_MODEL=deepseek_v4_1_flash PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=40 PREFILL_CHUNK_SIZE=$CHUNK PREFILL_MAX_SEQ_LEN=$MAXSEQ PREFILL_NUM_USERS=1 \\
  PREFILL_H2D_SERVICE_ID=$SERVICE PREFILL_HF_MODEL=/mnt/tt-data/sdawle/models/DeepSeek-V4.1-Flash-1017 PREFILL_TTNN_CACHE=/mnt/tt-data/sdawle/v41p_1017_weight_cache \\
  PREFILL_ENABLE_MIGRATION=0 PREFILL_MIGRATION_EXPORT_TO_FILE=1 PREFILL_MIGRATION_TABLE_PATH=$RUN/v41_table.pb PREFILL_MIGRATION_DEVICE_MAP_PATH=$RUN/v41_devmap.txt \\
  PREFILL_V41_TABLE_HOST=hostname PREFILL_FABRIC_MODE=2d PREFILL_ENABLE_LAYER_ACK=1 PREFILL_V41_PATH_A=1 PREFILL_V41_A3=1 \\
  V41_PREFILL_PROFILE=0 V41_PREFILL_WARM_ALL=1 V41_PREFILL_ISLANDS=1 DSV41_ENGRAM_TABLE_DIR=$ETD DSV41_ENGRAM_REMOTE=1=10.82.97.31:$ROWS_PORT,14=10.82.97.29:$ROWS_PORT,
unset PREFILL_MOCK_MIGRATION PREFILL_FIRST_TOKEN_DIR PREFILL_REQUEST_SIDECAR_DIR
export V41_PREFILL_DEVICE_EMBED=1 V41_PREFILL_ENGRAM_RM=1 V41_EXPORT_CP2=1 V41_EXPORT_IDX_SHARD=64 V41_EXPORT_CP2_POSITIONS=$MAXPOS
export V41_PREFILL_PREFIX_REUSE=1 V41_PREFILL_PREFIX_CKPTS=$CKPTS
export PREFIX_CHECK_CTL=$RUN/ctl PREFIX_CHECK_RUNNER_LOG=$RUN/runner.log PREFIX_CHECK_OUT=$RUN PREFIX_CHECK_PHASES=$PHASES PREFIX_CHECK_STACKS=$RUN/stacks.txt
EOF
if [ -n "${STORE_GB:-}" ]; then  # H2 (run_device_check_h2.sh): the host chunk store on top of H1
  echo "export V41_PREFILL_PREFIX_STORE_GB=$STORE_GB V41_PREFILL_PREFIX_HOST_RESERVE_GB=${RESERVE_GB:-32} V41_PREFILL_PREFIX_SPILL_TIPS=${SPILL_TIPS:-2}${STORE_TIP_GB:+ V41_PREFILL_PREFIX_STORE_TIP_GB=$STORE_TIP_GB}" >> "$OUT/env.sh"
fi
rsync -a "$OUT/env.sh" $P:$RUN/env.sh || exit 1
log "runner harness on $H (chunk $CHUNK, MAXSEQ $MAXSEQ, CP2 positions $MAXPOS, $CKPTS checkpoints, phases $PHASES)"
R $P "rm -f /dev/shm/*${SERVICE}*; source $RUN/env.sh && echo 'reuse -' > $RUN/ctl && (setsid nohup timeout 10800 python $DST/models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse/device_check/runner_harness.py > $RUN/runner.log 2>&1 < /dev/null & echo \$! > $RUN/runner.pid); sleep 2; cat $RUN/runner.pid" | sed 's/^/  runner pid /' | tee -a "$OUT/check.log"
for i in $(seq 1 240); do
  R $P "grep -q 'entering request loop' $RUN/runner.log" && { log "runner ready after ~$((i * 15)) s"; break; }
  R $P "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null" || { log "runner DIED"; R $P "grep -h 'Error\|TT_FATAL\|TT_THROW\|Traceback' $RUN/runner.log | head -8 | cut -c1-240" | tee -a "$OUT/check.log"; exit 1; }
  sleep 15
done
R $P "grep -q 'entering request loop' $RUN/runner.log" || { log "runner not ready after 60 min"; exit 1; }
R $P "grep -h 'v41-prefix\|compiled\|warm-all\|islands' $RUN/runner.log | cut -c1-240 | tail -8" | sed 's/^/  /' | tee -a "$OUT/check.log"

log "driver start"
R $P "source $RUN/env.sh && timeout 7200 systemd-run --user --scope -q -p MemoryMax=40G nice -n 19 python $DST/models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse/device_check/driver.py > $RUN/driver.log 2>&1; echo driver rc \$?" | tee -a "$OUT/check.log"
R $P "grep '\[driver\]' $RUN/driver.log | cut -c1-400" | tee -a "$OUT/check.log"
for i in $(seq 1 18); do R $P "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null" || break; sleep 10; done
R $P "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null && echo 'runner still up after the sentinel (killed at exit)' || echo 'runner exited'" | tee -a "$OUT/check.log"
