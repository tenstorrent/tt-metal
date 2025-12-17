USER="ttuser"
CONFIG_FILE="training_shakespeare_tinyllama_tensor_parallel_3tier_fabric.yaml"
HOST_FILE=${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/fabric_minimal_example/configurations/2loudboxes/hosts.txt
RANK_BINDINGS_FILE=${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/fabric_minimal_example/configurations/2loudboxes/rank_bindings.yaml

# Allow overrides via environment or CLI args (with defaults above)
while [[ "$#" -gt 0 ]]; do
    case $1 in
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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# copy all files to all machines (pass user and hostfile)
${TT_METAL_HOME}/tt-train/sources/examples/nano_gpt/3tier/all_machines_copy.sh --run --sync --user "$USER" --hostfile "$HOST_FILE"

CMD="python3 ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/fabric_minimal_example/example.py -c ${CONFIG_FILE}"
# use tt-run to run the example script across all machines
${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py --rank-binding ${RANK_BINDINGS_FILE} --mpi-args "--hostfile ${HOST_FILE} --tag-output" ${CMD}
