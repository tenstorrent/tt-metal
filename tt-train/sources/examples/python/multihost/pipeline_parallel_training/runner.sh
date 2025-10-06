# copy all files to all machines
${TT_METAL_HOME}/tt-train/sources/examples/nano_gpt/3tier/all_machines_copy.sh --run --sync

HOST_FILE=${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/configurations/5loudboxes/hosts.txt
RANK_BINDINGS_FILE=${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/configurations/5loudboxes/rank_bindings.yaml

# install requirements
mpirun --hostfile ${HOST_FILE} --tag-output pip install -r ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/requirements.txt

# use tt-run to run the training script across all machines
${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py --rank-binding ${RANK_BINDINGS_FILE} --mpi-args "--hostfile ${HOST_FILE} --tag-output" python3 ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/training.py
