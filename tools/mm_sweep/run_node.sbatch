#!/bin/bash
# Per-node compute job: runs the block sweep on THIS node's 32 chips for a pre-staged shard.
# tt-metal + the shard json are rsync'd to node-local disk by the A1 controller BEFORE submit (nodes
# can't ssh back to A1), so this job is self-contained and only needs node-local files.
# Submit: sbatch --partition=<p> --nodelist=<node> run_node.sbatch <shard.json> <out.json> [nchips]
#SBATCH --time=240
#SBATCH --output=/home/cglagovich/mm_sweep_out/slurm_%j.log
set -x
SHARD="$1"; OUT="$2"; NCHIPS="${3:-32}"
IMG=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
echo "node=$(hostname) job=$SLURM_JOB_ID shard=$SHARD out=$OUT nchips=$NCHIPS start=$(date)"
docker run --rm --user 4015:4015 \
  --device /dev/tenstorrent \
  -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v /home/cglagovich:/home/cglagovich \
  "$IMG" bash -c "
cd /home/cglagovich/tt-metal &&
export ARCH_NAME=wormhole_b0 TT_METAL_HOME=/home/cglagovich/tt-metal &&
export TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-120} TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 &&
export TT_METAL_DEVICE_PROFILER=1 USE_PROFILER=1 BSWEEP_JOINT_SPK=${BSWEEP_JOINT_SPK:-} BSWEEP_BASELINE=${BSWEEP_BASELINE:-} BSWEEP_FORCE_PREFETCH=${BSWEEP_FORCE_PREFETCH:-} BSWEEP_NO_LARGE_LEVERS=${BSWEEP_NO_LARGE_LEVERS:-} &&
/home/cglagovich/tt-metal/python_env/bin/python tools/mm_sweep/block_sweep_mp.py '$SHARD' '$OUT' $NCHIPS
"
echo "node=$(hostname) job=$SLURM_JOB_ID done=$(date) rc=$?"
