#!/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=bh_sp_5x4x32_C1_C10
#SBATCH --nodelist=bh-glx-c03u02,bh-glx-c03u08,bh-glx-c04u02,bh-glx-c04u08
#SBATCH --job-name=hello_world_docker
#SBATCH --output=logs/hello_world_inference_%j.out
#SBATCH --error=logs/hello_world_inference_%j.err

export TT_METAL_HOME="/data/dmadic/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_IMAGE="${TT_METAL_IMAGE:-ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-models-amd64:latest-rc}"

cd "${TT_METAL_HOME}"
source python_env/bin/activate

# Public image: pull on all nodes (no auth).
if ! srun --ntasks-per-node=1 docker pull "${TT_METAL_IMAGE}"; then
  echo "ERROR: docker pull failed for ${TT_METAL_IMAGE}"
  exit 1
fi

# # Reset galaxies
# if ! srun --ntasks-per-node=1 tt-smi -glx_reset; then
#   echo "ERROR: tt-smi -glx_reset failed"
#   exit 1
# fi

# python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py

# Same nodes, rankfile, and test as multi-host.sbatch; each rank runs in Docker via the wrapper.
export TT_METAL_IMAGE="ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-models-amd64:latest-rc"

# Dry-run to see the full mpirun + docker run command
tools/scaleout/exabox/tt-run-docker \
  --rank-binding blitz_decode_pipeline_rank_binding_single_pod.yaml \
  --mpi-args "--map-by rankfile:file=blitz_decode_pipeline_rank_file_single_pod --bind-to hwt:overload-allowed --tag-output --oversubscribe" \
  pytest -svv models/demos/deepseek_v3_b1/tests/unit_tests/test_multi_host_pipeline.py -k test_pipeline_block
