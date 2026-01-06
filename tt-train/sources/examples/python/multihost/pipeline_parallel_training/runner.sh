# Default profile
PROFILE="loudboxes"
USER=""
CONFIG_FILE=""
HOST_FILE=""
RANK_BINDINGS_FILE=""
MPI_EXTRA_ARGS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --profile)
            shift
            PROFILE="$1"
            ;;
        --hostfile)
            shift
            HOST_FILE="$1"
            ;;
        --user)
            shift
            USER="$1"
            ;;
        --rank-bindings)
            shift
            RANK_BINDINGS_FILE="$1"
            ;;
        --config)
            shift
            CONFIG_FILE="$1"
            ;;
        --mpi-extra-args)
            shift
            MPI_EXTRA_ARGS="$1"
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--profile loudboxes|galaxies] [--user USER] [--hostfile PATH] [--rank-bindings PATH] [--config CONFIG_FILE] [--mpi-extra-args ARGS]"
            exit 1
            ;;
    esac
    shift
done

# Set defaults based on profile if not explicitly provided
if [[ "$PROFILE" == "loudboxes" ]]; then
    USER="${USER:-ttuser}"
    CONFIG_FILE="${CONFIG_FILE:-training_shakespeare_llama8b_pp_fabric.yaml}"
    HOST_FILE="${HOST_FILE:-${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/configurations/5loudboxes/hosts.txt}"
    RANK_BINDINGS_FILE="${RANK_BINDINGS_FILE:-${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/configurations/5loudboxes/rank_bindings.yaml}"
elif [[ "$PROFILE" == "galaxies" ]]; then
    USER="${USER:-local-rfurko}"
    CONFIG_FILE="${CONFIG_FILE:-training_shakespeare_llama8b_pp_fabric_galaxy.yaml}"
    HOST_FILE="${HOST_FILE:-${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/configurations/5galaxies/hosts.txt}"
    RANK_BINDINGS_FILE="${RANK_BINDINGS_FILE:-${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/configurations/5galaxies/rank_bindings.yaml}"
    MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:---allow-run-as-root --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0}"
else
    echo "Error: Unknown profile '$PROFILE'. Use 'loudboxes' or 'galaxies'."
    exit 1
fi

# copy all files to all machines (pass user and hostfile)
${TT_METAL_HOME}/tt-train/sources/examples/nano_gpt/3tier/all_machines_copy.sh --run --sync --user "$USER" --hostfile "$HOST_FILE"

# install requirements
mpirun-ulfm --hostfile ${HOST_FILE} --tag-output pip install -r ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/requirements.txt

CMD="python3 ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/training.py -c ${CONFIG_FILE}"
# use tt-run to run the training script across all machines
${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py --rank-binding ${RANK_BINDINGS_FILE} --mpi-args "--hostfile ${HOST_FILE} ${MPI_EXTRA_ARGS} --tag-output" ${CMD}
