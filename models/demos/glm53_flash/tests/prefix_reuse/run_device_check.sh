#!/bin/bash
# Device check of the prefill-side prefix reuse (tt-blaze G53F-0311), run from host 29 against a FREE 8x4 galaxy H
# (default 41): the real prefill runner with GLM53F_PREFIX_REUSE=1 (runner_harness.py) + a tt-d-gen-shaped producer
# (driver.py) that marks chunk headers as tt-d-gen device.prefill.mark_last_chunk does. One model build
# (~29 min), then: cold references, the reuse sequence (export-slab + KDA digests must equal the references), timing
# (arm off vs cold vs reuse), the reuse sequence with the mark off, shutdown. ~40 min in all.
#   H, SRC (this worktree), WT (H's built tree, the one prefill_up.sh syncs into), PYBIN, RUN (scratch dir on H),
#   OUT (results on 29), CKPTS (checkpoint slots, default 4 so the run exercises eviction + the silent rebuild).
#   Refuses if any process holds H's chips. Restores WT's runner files on exit.
set -u
H=${H:-41}; P=10.82.97.$H
SRC=${SRC:-/home/ttuser/sdawle/tt-metal-glm53fp-prefix}
WT=${WT:-/home/ttuser/sdawle/tt-metal-glm53fp-wave}; PYBIN=${PYBIN:-/home/ttuser/sdawle/tt-metal-glm53fp/python_env/bin/python}
RUN=${RUN:-/home/ttuser/sdawle/prefill_runs/prefix_check}
OUT=${OUT:-/home/ttuser/sdawle/glm53f_logs/trackH_prefix_reuse/$(date +%m%d_%H%M)}
CHUNK=${CHUNK:-12288}; MAXSEQ=${MAXSEQ:-110592}; CKPTS=${CKPTS:-4}; SERVICE=glm53f_prefix_check
# phase 2: HOST_GB > 0 turns the host chunk store on (GLM53F_PREFIX_HOST_GB); PHASES picks the driver's phases (R,S,T,U = H1;
# M = several retained prompts); WARMUP / BUCKETS override the runner's (the live 1M config: MAXSEQ=1056768 + its buckets)
HOST_GB=${HOST_GB:-0}; PHASES=${PHASES:-R,S,T,U}
WARMUP=${WARMUP:-c83864,c41864,c39288,4092:0}; BUCKETS=${BUCKETS:-512:4096,1024:8192,2048:12288}
GP=/mnt/tt-data/sdawle/glm53f/prefill
FILES="models/demos/glm53_flash/tt/runners models/demos/glm53_flash/tests/test_runner_kv_export.py models/demos/common/prefill/adapter.py"
mkdir -p $OUT
log() { echo "[prefix-check] $(date +%T) $*" | tee -a $OUT/check.log; }
R() { ssh -o BatchMode=yes -o ServerAliveInterval=30 -o LogLevel=ERROR $P "$1" < /dev/null; }

busy=$(R "sudo -n lsof /dev/tenstorrent/* 2>/dev/null | awk 'NR>1{print \$1\":\"\$2}' | sort -u | tr '\n' ' '")
[ -z "$busy" ] || { log "host $H chips are held ($busy) -- not running"; exit 1; }
others=$(R "who | awk '{print \$1}' | sort -u | grep -v '^ttuser$' | tr '\n' ' '")
[ -z "$others" ] || { log "other users logged in on $H ($others) -- not running"; exit 1; }
log "host $H free; SRC $(git -C $SRC log --oneline -1 | cut -c1-12) ($(git -C $SRC status --short | wc -l) local change(s)), OUT $OUT"
R "md5sum $WT/models/demos/glm53_flash/tt/runners/*.py" | tee $OUT/wt_md5_before.txt > /dev/null

R "rm -rf $RUN && mkdir -p $RUN/scripts && cd $WT && tar czf $RUN/wt_runner_files_backup.tgz $FILES" || { log "backup failed"; exit 1; }
restore() {
  R "[ -f $RUN/runner.pid ] && kill -0 \$(cat $RUN/runner.pid) 2>/dev/null && { kill -TERM \$(cat $RUN/runner.pid); sleep 20; kill -KILL \$(cat $RUN/runner.pid) 2>/dev/null; }; cd $WT && rm -f models/demos/glm53_flash/tt/runners/prefix_reuse.py && tar xzf $RUN/wt_runner_files_backup.tgz && rm -rf models/demos/glm53_flash/tt/runners/__pycache__/prefix_reuse*" > /dev/null
  R "md5sum $WT/models/demos/glm53_flash/tt/runners/*.py" > $OUT/wt_md5_after.txt
  cmp -s $OUT/wt_md5_before.txt $OUT/wt_md5_after.txt && log "WT runner files restored (md5s = before)" || log "WARNING: WT runner md5s differ from before"
  rsync -a $P:$RUN/runner.log $P:$RUN/driver.log $P:$RUN/results.json $P:$RUN/stacks.txt $OUT/ 2>/dev/null
  R "rm -f /dev/shm/*${SERVICE}* /dev/shm/*prefix_check_lcr*; rm -rf $RUN"
  R "sudo -n lsof /dev/tenstorrent/* 2>/dev/null | awk 'NR>1' | wc -l" | sed 's/^/  chip holders after the run: /' | tee -a $OUT/check.log
}
trap restore EXIT
for f in $FILES; do rsync -a --exclude __pycache__ $SRC/$f $P:$WT/$(dirname $f)/ || { log "sync of $f failed"; exit 1; }; done
rsync -a $SRC/models/demos/glm53_flash/tests/prefix_reuse/ $P:$RUN/scripts/

cat > $OUT/env.sh <<EOF
cd $WT
export TT_METAL_HOME=$WT TT_METAL_RUNTIME_ROOT=$WT PYTHONPATH=$WT:$WT/ttnn ARCH_NAME=blackhole HF_HOME=$GP/hf_home \\
  TT_METAL_CACHE=/dev/shm/tt-metal-cache-glm53fp-wave LOGURU_LEVEL=INFO
export PREFILL_MANIFEST=models/demos/glm53_flash/tt/runners/manifests/glm53_flash.json PREFILL_MODEL=glm53_flash \\
  PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=46 PREFILL_CHUNK_SIZE=$CHUNK PREFILL_MAX_SEQ_LEN=$MAXSEQ PREFILL_NUM_USERS=1 \\
  PREFILL_H2D_SERVICE_ID=$SERVICE PREFILL_TTNN_CACHE=$GP/mesh_cache PREFILL_ENABLE_LAYER_ACK=1 \\
  PREFILL_ENABLE_MIGRATION=0 PREFILL_MIGRATION_EXPORT_TO_FILE=1 PREFILL_MIGRATION_TABLE_PATH=$RUN/glm_table.pb \\
  PREFILL_MIGRATION_DEVICE_MAP_PATH=$RUN/glm_devmap.txt GLM53F_KDA_VERSIONS=2 PREFILL_LAYER_COMPLETION_RING=/tt_prefix_check_lcr \\
  GLM53F_RUNNER_WARMUP=$WARMUP GLM53F_RUNNER_BUCKETS=$BUCKETS
export GLM53_MESH_CACHE=$GP/mesh_cache GLM53_PARTITION_USERS=0 GLM53_PREFILL_EXPERTS=regions GLM53_REGIONS_T=16384 GLM53_PREFILL_QBLOCK=512 GLM53_MODEL_TRACE_MB=0 \\
  GLM53_REGIONS_BUF=full GLM53_REGIONS_COMBINE=fused GLM53_PREFILL_ABLOCK=2048 GLM53_CCL_GATHER_TOPOLOGY=ring GLM53_INDEXER_NO_CROP=1 \\
  GLM53_REGIONS_RANK=mm GLM53_REGIONS_UNTILIZE=used GLM53_REGIONS_SPANS=1 GLM53_INDEXER_EXPAND=custom GLM53_INDEXER_COLSPLIT=1 GLM53_KDA_CHRONO_ANCHOR=1
export GLM53F_PREFIX_REUSE=1 GLM53F_PREFIX_CKPTS=$CKPTS GLM53F_PREFIX_HOST_GB=$HOST_GB PREFIX_CHECK_PHASES=$PHASES PREFIX_CHECK_CTL=$RUN/ctl \\
  PREFIX_CHECK_STACKS=$RUN/stacks.txt PREFIX_CHECK_RUNNER_LOG=$RUN/runner.log PREFIX_CHECK_OUT=$RUN
EOF
rsync -a $OUT/env.sh $P:$RUN/env.sh
R "rm -f /dev/shm/*${SERVICE}* /dev/shm/*prefix_check_lcr*; source $RUN/env.sh && echo 'reuse -' > $RUN/ctl && (setsid nohup timeout 10800 $PYBIN $RUN/scripts/runner_harness.py > $RUN/runner.log 2>&1 < /dev/null & echo \$! > $RUN/runner.pid); sleep 2; cat $RUN/runner.pid" | sed 's/^/  runner pid /' | tee -a $OUT/check.log
for i in $(seq 1 240); do
  R "grep -q 'entering request loop' $RUN/runner.log" && { log "runner ready after ~$((i * 15)) s"; break; }
  R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null" || { log "runner DIED"; R "grep -h 'Error\|TT_FATAL\|TT_THROW\|Traceback' $RUN/runner.log | head -8 | cut -c1-240" | tee -a $OUT/check.log; exit 1; }
  sleep 15
done
R "grep -q 'entering request loop' $RUN/runner.log" || { log "runner not ready after 60 min"; exit 1; }
R "grep -h 'model built\|warm-up\|glm53f-prefix\|slabs:' $RUN/runner.log | cut -c1-260" | tee -a $OUT/check.log
R "free -g | head -2; ps -o rss= --ppid \$(cat $RUN/runner.pid) | awk '{print \"runner RSS after warm-up \" \$1/1048576 \" GiB\"}'" | tee -a $OUT/check.log
log "driver start"
R "source $RUN/env.sh && timeout 3600 systemd-run --user --scope -q -p MemoryMax=40G nice -n 19 $PYBIN $RUN/scripts/driver.py > $RUN/driver.log 2>&1; echo driver rc \$?" | tee -a $OUT/check.log
R "grep '\[driver\]' $RUN/driver.log" | tee -a $OUT/check.log
for i in $(seq 1 12); do R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null" || break; sleep 10; done
R "kill -0 \$(cat $RUN/runner.pid) 2>/dev/null && echo 'runner still up after the sentinel' || echo 'runner exited'; tail -3 $RUN/runner.log | cut -c1-200" | tee -a $OUT/check.log
log "done"
