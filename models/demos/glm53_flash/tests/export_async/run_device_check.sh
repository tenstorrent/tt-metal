#!/bin/bash
# Device check of GLM53F_EXPORT_ASYNC=pack + the socket guard (tt-blaze G53F-0309), run from host 29 against a FREE
# 8x4 galaxy H (default 41): the real prefill runner (runner_harness.py) + a tt-d-gen-shaped producer (driver.py).
# One model build (~29 min), then: sync vs async timing, crc32 of the exported bytes, the pre-fix hang reproduced with
# the guard bypassed (+ Python and gdb stacks of the stuck runner), recovery, shutdown. ~35-40 min in all.
#   H, SRC (this worktree), WT (H's built tree, the one prefill_up.sh syncs into), PYBIN, RUN (scratch dir on H),
#   OUT (results on 29). Refuses if any process holds H's chips. Restores WT's runner files on exit.
set -u
H=${H:-41}; P=10.82.97.$H
SRC=${SRC:-/home/ttuser/sdawle/tt-metal-glm53fp-ttdgen-async}
WT=${WT:-/home/ttuser/sdawle/tt-metal-glm53fp-wave}; PYBIN=${PYBIN:-/home/ttuser/sdawle/tt-metal-glm53fp/python_env/bin/python}
RUN=${RUN:-/home/ttuser/sdawle/prefill_runs/export_check}
OUT=${OUT:-/home/ttuser/sdawle/glm53f_logs/trackG_export_async/$(date +%m%d_%H%M)}
CHUNK=${CHUNK:-12288}; MAXSEQ=${MAXSEQ:-110592}; SERVICE=glm53f_export_check
GP=/mnt/tt-data/sdawle/glm53f/prefill
FILES="models/demos/glm53_flash/tt/runners models/demos/glm53_flash/tests/test_runner_kv_export.py models/demos/common/prefill/adapter.py"
mkdir -p $OUT
log() { echo "[export-check] $(date +%T) $*" | tee -a $OUT/check.log; }
R() { ssh -o BatchMode=yes -o ServerAliveInterval=30 -o LogLevel=ERROR $P "$1" < /dev/null; }

busy=$(R "sudo -n lsof /dev/tenstorrent/* 2>/dev/null | awk 'NR>1{print \$1\":\"\$2}' | sort -u | tr '\n' ' '")
[ -z "$busy" ] || { log "host $H chips are held ($busy) -- not running"; exit 1; }
log "host $H free; SRC $(git -C $SRC log --oneline -1 | cut -c1-12) ($(git -C $SRC status --short | wc -l) local change(s)), OUT $OUT"

R "rm -rf $RUN && mkdir -p $RUN/scripts && cd $WT && tar czf $RUN/wt_runner_files_backup.tgz $FILES" || { log "backup failed"; exit 1; }
restore() {
  R "[ -f $RUN/runner.pid ] && kill -0 \$(cat $RUN/runner.pid) 2>/dev/null && { kill -TERM \$(cat $RUN/runner.pid); sleep 20; kill -KILL \$(cat $RUN/runner.pid) 2>/dev/null; }; cd $WT && rm -f models/demos/glm53_flash/tt/runners/export_guard.py && tar xzf $RUN/wt_runner_files_backup.tgz && md5sum models/demos/glm53_flash/tt/runners/runtime.py" | sed 's/^/  restored /' | tee -a $OUT/check.log
  rsync -a $P:$RUN/runner.log $P:$RUN/driver.log $P:$RUN/results.json $P:$RUN/stacks_U.txt $P:$RUN/stacks.txt $OUT/ 2>/dev/null
}
trap restore EXIT
for f in $FILES; do rsync -a --exclude __pycache__ $SRC/$f $P:$WT/$(dirname $f)/ || { log "sync of $f failed"; exit 1; }; done
rsync -a $SRC/models/demos/glm53_flash/tests/export_async/ $P:$RUN/scripts/

cat > $OUT/env.sh <<EOF
cd $WT
export TT_METAL_HOME=$WT TT_METAL_RUNTIME_ROOT=$WT PYTHONPATH=$WT:$WT/ttnn ARCH_NAME=blackhole HF_HOME=$GP/hf_home \\
  TT_METAL_CACHE=/dev/shm/tt-metal-cache-glm53fp-wave LOGURU_LEVEL=INFO
export PREFILL_MANIFEST=models/demos/glm53_flash/tt/runners/manifests/glm53_flash.json PREFILL_MODEL=glm53_flash \\
  PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=46 PREFILL_CHUNK_SIZE=$CHUNK PREFILL_MAX_SEQ_LEN=$MAXSEQ PREFILL_NUM_USERS=1 \\
  PREFILL_H2D_SERVICE_ID=$SERVICE PREFILL_TTNN_CACHE=$GP/mesh_cache PREFILL_ENABLE_LAYER_ACK=1 \\
  PREFILL_ENABLE_MIGRATION=0 PREFILL_MIGRATION_EXPORT_TO_FILE=1 PREFILL_MIGRATION_TABLE_PATH=$RUN/glm_table.pb \\
  PREFILL_MIGRATION_DEVICE_MAP_PATH=$RUN/glm_devmap.txt GLM53F_KDA_VERSIONS=2 PREFILL_LAYER_COMPLETION_RING=/tt_export_check_lcr \\
  GLM53F_RUNNER_WARMUP=c$((8 * CHUNK)),c$((3 * CHUNK + 5000)),4092:0 GLM53F_RUNNER_BUCKETS=512:4096,1024:8192,2048:12288
export GLM53_MESH_CACHE=$GP/mesh_cache GLM53_PARTITION_USERS=0 GLM53_PREFILL_EXPERTS=regions GLM53_REGIONS_T=16384 GLM53_PREFILL_QBLOCK=512 GLM53_MODEL_TRACE_MB=0 \\
  GLM53_REGIONS_BUF=full GLM53_REGIONS_COMBINE=fused GLM53_PREFILL_ABLOCK=2048 GLM53_CCL_GATHER_TOPOLOGY=ring GLM53_INDEXER_NO_CROP=1 \\
  GLM53_REGIONS_RANK=mm GLM53_REGIONS_UNTILIZE=used GLM53_REGIONS_SPANS=1 GLM53_INDEXER_EXPAND=custom GLM53_INDEXER_COLSPLIT=1 GLM53_KDA_CHRONO_ANCHOR=1
export GLM53F_EXPORT_ASYNC=pack GLM53F_EXPORT_PROFILE=1 EXPORT_CHECK_MODE_FILE=$RUN/mode EXPORT_CHECK_STACKS=$RUN/stacks.txt \\
  EXPORT_CHECK_RUNNER_LOG=$RUN/runner.log EXPORT_CHECK_OUT=$RUN
EOF
rsync -a $OUT/env.sh $P:$RUN/env.sh
R "rm -f /dev/shm/*${SERVICE}* /dev/shm/*export_check_lcr*; source $RUN/env.sh && echo sync > $RUN/mode && (setsid nohup timeout 10800 $PYBIN $RUN/scripts/runner_harness.py > $RUN/runner.log 2>&1 < /dev/null & echo \$! > $RUN/runner.pid); sleep 2; cat $RUN/runner.pid" | sed 's/^/  runner pid /' | tee -a $OUT/check.log
for i in $(seq 1 240); do
  R "grep -q 'entering request loop' $RUN/runner.log" && { log "runner ready after ~$((i * 15)) s"; break; }
  R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null" || { log "runner DIED"; R "grep -h 'Error\|TT_FATAL\|TT_THROW\|Traceback' $RUN/runner.log | head -8 | cut -c1-240" | tee -a $OUT/check.log; exit 1; }
  sleep 15
done
R "grep -q 'entering request loop' $RUN/runner.log" || { log "runner not ready after 60 min"; exit 1; }
R "grep -h 'model built\|warm-up\|socket guard\|slabs:' $RUN/runner.log | cut -c1-240" | tee -a $OUT/check.log
log "driver start"
R "source $RUN/env.sh && export EXPORT_CHECK_RUNNER_PID=\$(cat $RUN/runner.pid) && timeout 3600 systemd-run --user --scope -q -p MemoryMax=40G nice -n 19 $PYBIN $RUN/scripts/driver.py > $RUN/driver.log 2>&1; echo driver rc \$?" | tee -a $OUT/check.log
R "grep '\[driver\]' $RUN/driver.log" | tee -a $OUT/check.log
for i in $(seq 1 12); do R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null" || break; sleep 10; done
R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null && echo 'runner still up after the sentinel' || echo 'runner exited'; tail -3 $RUN/runner.log | cut -c1-200" | tee -a $OUT/check.log
log "done"
