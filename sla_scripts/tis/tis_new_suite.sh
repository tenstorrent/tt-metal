#!/bin/bash
# Full check of the self-built image: garbage probe, SLA load tests, MMLU; then frees the Galaxy.
R=~/teja/runs; S=$R/tis_new_suite.log
log() { echo "$(date -u +%H:%M) $*" >> $S; }
log "wait for server"
until curl -sf localhost:8000/v1/models > /dev/null; do sleep 30; done
M=$(curl -s localhost:8000/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
log "server up, model $M"
python3 ~/teja/garbage_probe.py http://localhost:8000/v1/completions "$M" > $R/garbage_probe_new.log 2>&1
log "probe exit $?"
OUT=$R/bench_tis_new HIT_TAG=tis_new_rpm60_40pct bash ~/teja/bench_tis.sh
log "bench done"
MODEL="$M" NC=8 TMO=600 bash ~/teja/run_mmlu.sh tis_new
log "mmlu done: $(grep -E 'MMLU EXIT' $R/mmlu_tis_new.log)"
docker stop $(docker ps -q --filter name=tt-inference-server) >> $S 2>&1
tt-smi -glx_reset_auto > $R/reset_handover.log 2>&1
log "SUITE DONE, galaxy reset exit $?"
