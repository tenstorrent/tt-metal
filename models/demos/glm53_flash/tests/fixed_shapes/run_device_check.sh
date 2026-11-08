#!/bin/bash
# Device check of GLM53_PREFILL_FIXED_SHAPES (tt-blaze G53F-0313), run from host 29 against a FREE 8x4 galaxy H (default
# 41): (A) unit_device.py on one chip -- the index maintenance per-length vs fixed on stale caches, bit identity + programs;
# (B) the real prefill runner (runner_harness.py: per-prompt arm, export digests, program attribution) in the 1M serving
# recipe (chunk 12288, cache 1056768, buckets 512:4096,1024:8192,2048:12288, the serving warm-up incl. c1040000) driven by
# driver.py (growth probe pattern, ON vs OFF digests, warmed timing). One model build (~29 min); ~70 min in all.
#   H, SRC (this worktree), WT (H's built tree), PYBIN, RUN (scratch dir on H), OUT (results on 29), PHASES (A,B),
#   DRIVER_PHASES (G,D,T). Refuses if any process holds H's chips; restores WT's files on exit; the caller owns the lock.
set -u
H=${H:-41}; P=10.82.97.$H
SRC=${SRC:-/home/ttuser/sdawle/tt-metal-glm53fp-kvstep}
WT=${WT:-/home/ttuser/sdawle/tt-metal-glm53fp-wave}; PYBIN=${PYBIN:-/home/ttuser/sdawle/tt-metal-glm53fp/python_env/bin/python}
RUN=${RUN:-/home/ttuser/sdawle/prefill_runs/fs_check}
OUT=${OUT:-/home/ttuser/sdawle/glm53f_logs/k1_fixed_shapes/$(date +%m%d_%H%M)}
CHUNK=${CHUNK:-12288}; MAXSEQ=${MAXSEQ:-1056768}; SERVICE=glm53f_fs_check
BUCKETS=${BUCKETS:-512:4096,1024:8192,2048:12288}
WARMUP=${WARMUP:-508:0,1020:0,1532:0,2044:0,2556:0,3068:0,3580:0,4092:0,5116:0,6140:0,7164:0,8188:0,10236:0,12284:0,c1040000}
PHASES=${PHASES:-A,B}; DRIVER_PHASES=${DRIVER_PHASES:-G,D,T}
GP=/mnt/tt-data/sdawle/glm53f/prefill
FILES="models/demos/glm53_flash/tt/indexer.py models/demos/glm53_flash/tt/dsa.py models/demos/glm53_flash/tt/runners"
mkdir -p $OUT
log() { echo "[fs-check] $(date +%T) $*" | tee -a $OUT/check.log; }
R() { ssh -o BatchMode=yes -o ServerAliveInterval=30 -o LogLevel=ERROR $P "$1" < /dev/null; }

busy=$(R "sudo -n lsof /dev/tenstorrent/* 2>/dev/null | awk 'NR>1{print \$1\":\"\$2}' | sort -u | tr '\n' ' '")
[ -z "$busy" ] || { log "host $H chips are held ($busy) -- not running"; exit 1; }
log "host $H free; SHM_OFF=${SHM_OFF:-0}; SRC $(git -C $SRC log --oneline -1 | cut -c1-12) ($(git -C $SRC status --short | wc -l) local change(s)), OUT $OUT"
R "md5sum $WT/models/demos/glm53_flash/tt/indexer.py $WT/models/demos/glm53_flash/tt/dsa.py $WT/models/demos/glm53_flash/tt/runners/*.py" > $OUT/wt_md5_before.txt
R "rm -rf $RUN && mkdir -p $RUN/scripts && cd $WT && tar czf $RUN/wt_backup.tgz $FILES" || { log "backup failed"; exit 1; }
restore() {
  R "[ -f $RUN/runner.pid ] && kill -0 \$(cat $RUN/runner.pid) 2>/dev/null && { kill -TERM \$(cat $RUN/runner.pid); sleep 20; kill -KILL \$(cat $RUN/runner.pid) 2>/dev/null; }; cd $WT && tar xzf $RUN/wt_backup.tgz && rm -rf models/demos/glm53_flash/tt/__pycache__/indexer* models/demos/glm53_flash/tt/__pycache__/dsa* models/demos/glm53_flash/tt/runners/__pycache__" > /dev/null
  R "md5sum $WT/models/demos/glm53_flash/tt/indexer.py $WT/models/demos/glm53_flash/tt/dsa.py $WT/models/demos/glm53_flash/tt/runners/*.py" > $OUT/wt_md5_after.txt
  cmp -s $OUT/wt_md5_before.txt $OUT/wt_md5_after.txt && log "WT files restored (md5s = before)" || log "WARNING: WT md5s differ from before"
  rsync -a $P:$RUN/runner.log $P:$RUN/driver.log $P:$RUN/unit.log $P:$RUN/results.json $OUT/ 2>/dev/null
  R "rm -f /dev/shm/*${SERVICE}* /dev/shm/*fs_check_lcr*; rm -rf $RUN"
  R "sudo -n lsof /dev/tenstorrent/* 2>/dev/null | awk 'NR>1' | wc -l" | sed 's/^/  chip holders after the run: /' | tee -a $OUT/check.log
}
trap restore EXIT
for f in $FILES; do rsync -a --exclude __pycache__ $SRC/$f $P:$WT/$(dirname $f)/ || { log "sync of $f failed"; exit 1; }; done
rsync -a $SRC/models/demos/glm53_flash/tests/fixed_shapes/ $P:$RUN/scripts/
R "md5sum $WT/models/demos/glm53_flash/tt/indexer.py $WT/models/demos/glm53_flash/tt/dsa.py" | tee -a $OUT/check.log

BASE_ENV="cd $WT && export TT_METAL_HOME=$WT TT_METAL_RUNTIME_ROOT=$WT PYTHONPATH=$WT:$WT/ttnn ARCH_NAME=blackhole HF_HOME=$GP/hf_home TT_METAL_CACHE=/dev/shm/tt-metal-cache-glm53fp-wave LOGURU_LEVEL=INFO"
# SHM_OFF=1: the serving config since 11-08 (tt-smi SHM memory stats off: every program create / destroy otherwise rescans
# every live program's CBs on all 32 chips, G53F-0313 12:20)
[ "${SHM_OFF:-0}" = 1 ] && BASE_ENV="$BASE_ENV TT_METAL_SHM_TRACKING_DISABLED=1"
if [[ ",$PHASES," == *",A,"* ]]; then
  log "phase A: unit_device.py on one chip"
  R "$BASE_ENV TT_VISIBLE_DEVICES=0 && timeout 1200 systemd-run --user --scope -q -p MemoryMax=40G $PYBIN $RUN/scripts/unit_device.py > $RUN/unit.log 2>&1; echo unit rc \$?" | tee -a $OUT/check.log
  R "grep '\[unit\]' $RUN/unit.log; grep -h 'Error\|TT_FATAL\|TT_THROW\|Traceback' $RUN/unit.log | head -5 | cut -c1-300" | tee -a $OUT/check.log
  rsync -a $P:$RUN/unit.log $OUT/ 2>/dev/null
  R "grep -q 'RESULT PASS' $RUN/unit.log" || { log "phase A did not pass -- not starting the runner"; exit 1; }
fi
[[ ",$PHASES," == *",B,"* ]] || { log "done (no phase B)"; exit 0; }

cat > $OUT/env.sh <<EOF
$BASE_ENV
export PREFILL_MANIFEST=models/demos/glm53_flash/tt/runners/manifests/glm53_flash.json PREFILL_MODEL=glm53_flash \\
  PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=46 PREFILL_CHUNK_SIZE=$CHUNK PREFILL_MAX_SEQ_LEN=$MAXSEQ PREFILL_NUM_USERS=1 \\
  PREFILL_H2D_SERVICE_ID=$SERVICE PREFILL_TTNN_CACHE=$GP/mesh_cache PREFILL_ENABLE_LAYER_ACK=1 \\
  PREFILL_ENABLE_MIGRATION=0 PREFILL_MIGRATION_EXPORT_TO_FILE=1 PREFILL_MIGRATION_TABLE_PATH=$RUN/glm_table.pb \\
  PREFILL_MIGRATION_DEVICE_MAP_PATH=$RUN/glm_devmap.txt GLM53F_KDA_VERSIONS=2 PREFILL_LAYER_COMPLETION_RING=/tt_fs_check_lcr \\
  GLM53F_RUNNER_WARMUP=$WARMUP GLM53F_RUNNER_BUCKETS=$BUCKETS
export GLM53_MESH_CACHE=$GP/mesh_cache GLM53_PARTITION_USERS=0 GLM53_PREFILL_EXPERTS=regions GLM53_REGIONS_T=16384 GLM53_PREFILL_QBLOCK=512 GLM53_MODEL_TRACE_MB=0 \\
  GLM53_REGIONS_BUF=full GLM53_REGIONS_COMBINE=fused GLM53_PREFILL_ABLOCK=2048 GLM53_CCL_GATHER_TOPOLOGY=ring GLM53_INDEXER_NO_CROP=1 \\
  GLM53_REGIONS_RANK=mm GLM53_REGIONS_UNTILIZE=used GLM53_REGIONS_SPANS=1 GLM53_INDEXER_EXPAND=custom GLM53_INDEXER_COLSPLIT=1 GLM53_KDA_CHRONO_ANCHOR=1
export GLM53_PREFILL_FIXED_SHAPES=0 FS_CHECK_CTL=$RUN/ctl FS_CHECK_RUNNER_LOG=$RUN/runner.log FS_CHECK_OUT=$RUN FS_CHECK_PHASES=$DRIVER_PHASES
EOF
rsync -a $OUT/env.sh $P:$RUN/env.sh
R "rm -f /dev/shm/*${SERVICE}* /dev/shm/*fs_check_lcr*; source $RUN/env.sh && (setsid nohup timeout 10800 $PYBIN $RUN/scripts/runner_harness.py > $RUN/runner.log 2>&1 < /dev/null & echo \$! > $RUN/runner.pid); sleep 2; cat $RUN/runner.pid" | sed 's/^/  runner pid /' | tee -a $OUT/check.log
for i in $(seq 1 280); do
  R "grep -q 'entering request loop' $RUN/runner.log" && { log "runner ready after ~$((i * 15)) s"; break; }
  R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null" || { log "runner DIED"; R "grep -h 'Error\|TT_FATAL\|TT_THROW\|Traceback' $RUN/runner.log | head -8 | cut -c1-240" | tee -a $OUT/check.log; exit 1; }
  sleep 15
done
R "grep -q 'entering request loop' $RUN/runner.log" || { log "runner not ready after 70 min"; exit 1; }
R "grep -h 'model built\|warm-up\|DRAM guard' $RUN/runner.log | cut -c1-260" | tee -a $OUT/check.log
log "driver start"
R "source $RUN/env.sh && timeout 5400 systemd-run --user --scope -q -p MemoryMax=40G nice -n 19 $PYBIN $RUN/scripts/driver.py > $RUN/driver.log 2>&1; echo driver rc \$?" | tee -a $OUT/check.log
R "grep '\[driver\]' $RUN/driver.log | tail -60" | tee -a $OUT/check.log
for i in $(seq 1 12); do R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null" || break; sleep 10; done
R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null && echo 'runner still up after the sentinel' || echo 'runner exited'; tail -3 $RUN/runner.log | cut -c1-200" | tee -a $OUT/check.log
log "done"
