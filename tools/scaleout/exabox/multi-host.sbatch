#!/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=bh_sp_5x4x32_C1_C10
#SBATCH --nodelist=bh-glx-c03u02,bh-glx-c03u08,bh-glx-c04u02,bh-glx-c04u08
#SBATCH --job-name=test_multi_host_pipeline
#SBATCH --output=logs/test_multi_host_pipeline_%j.out
#SBATCH --error=logs/test_multi_host_pipeline_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/dmadic/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"
export TT_METAL_SLOW_DISPATCH_MODE=1

cd ${TT_METAL_HOME}
source python_env/bin/activate

tt-run --mpi-args "--map-by rankfile:file=blitz_decode_pipeline_rank_file_single_pod --bind-to hwt:overload-allowed --tag-output --oversubscribe" --rank-binding blitz_decode_pipeline_rank_binding_single_pod.yaml pytest -svv models/demos/deepseek_v3_b1/tests/unit_tests/test_multi_host_pipeline.py -k test_pipeline_block