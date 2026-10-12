#!/bin/bash
# t417 build-only job (no device): builds /data/smarton/fasth3/t417/src. Submit only through submit417.sh.
#SBATCH -J smarton-fasth3-t417
#SBATCH -N 1
#SBATCH -o /data/smarton/fasth3/t417/slurm-%j.out
set -eo pipefail
F=/data/smarton/fasth3; T=$F/t417; W=$T/src
RC=$T/job-$SLURM_JOB_ID.rc
trap 'echo "JOB_RC=$? $(date -u +%FT%TZ)" > $RC' EXIT
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log "start node=$(hostname) job=$SLURM_JOB_ID head=$(git -C $W rev-parse HEAD)"
avail=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9); log "/data free ${avail}G"; [ "$avail" -ge 170 ]
export PATH=$F/python_env/bin:$PATH
cd $W
timeout --foreground -k 10 2600 bash build_metal.sh --release --cpm-source-cache $F/cpm-cache
[ -f ttnn/ttnn/_ttnn.so ]
log "build done"
