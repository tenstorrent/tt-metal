#!/bin/bash
#SBATCH --job-name=kv_bench
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=bh_pod_8x16_2
#SBATCH --nodelist=bh-glx-c03u02,bh-glx-c03u08
#SBATCH --output=kv_bench_%j.out
#SBATCH --error=kv_bench_%j.err
#SBATCH --exclusive

# Galaxy-to-Galaxy KV Cache Transfer Benchmark
# Tests different mesh shape configurations for disaggregated P/D

echo "=========================================="
echo "Galaxy KV Cache Transfer Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo ""

export TT_METAL_HOME="/data/dmadic/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"

cd ${TT_METAL_HOME}
source bh_python_env/bin/activate

# Create hostfile
HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "${host} slots=1"
done > ${HOSTFILE}

echo "Hostfile: ${HOSTFILE}"
cat ${HOSTFILE}
echo ""

# MPI options for multi-host
MPI_OPTS="--hostfile ${HOSTFILE} --mca btl_tcp_if_exclude docker0,lo --tag-output"

# Function to run benchmark for a specific configuration
run_benchmark() {
    local sender_mesh=$1
    local receiver_mesh=$2
    local rank_binding=$3
    
    echo ""
    echo "============================================================"
    echo "Running: ${sender_mesh} → ${receiver_mesh}"
    echo "Rank binding: ${rank_binding}"
    echo "============================================================"
    
    tt-run --verbose \
        --rank-binding ${rank_binding} \
        --mpi-args "${MPI_OPTS}" \
        python -m tracy -r -p tests/ttnn/distributed/benchmark_galaxy_kv_transfer.py \
            --sender-mesh ${sender_mesh} \
            --receiver-mesh ${receiver_mesh}
    
    local exit_code=$?
    echo "Exit code: ${exit_code}"
    return ${exit_code}
}

# Parse command line arguments
USE_BLACKHOLE=false
CONFIG="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --blackhole|-bh)
            USE_BLACKHOLE=true
            shift
            ;;
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--blackhole|-bh] [--config|-c CONFIG] [1x8-1x8|1x8-4x2|4x2-1x8|all]"
            echo ""
            echo "Options:"
            echo "  --blackhole, -bh    Use Blackhole MGD instead of Wormhole"
            echo "  --config, -c CONFIG  Configuration to run (default: all)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Configurations:"
            echo "  1x8-1x8             Sender 1x8 → Receiver 1x8"
            echo "  1x8-4x2             Sender 1x8 → Receiver 4x2"
            echo "  4x2-1x8             Sender 4x2 → Receiver 1x8"
            echo "  all                 Run all configurations (default)"
            exit 0
            ;;
        *)
            if [[ "$CONFIG" == "all" && "$1" != "" ]]; then
                CONFIG="$1"
            fi
            shift
            ;;
    esac
done

# Set rank binding file based on blackhole option
if [[ "$USE_BLACKHOLE" == "true" ]]; then
    RANK_BINDING="tests/tt_metal/distributed/config/exabox_2_galaxy_bh_rank_binding.yaml"
    echo "Using Blackhole configuration"
else
    RANK_BINDING="tests/tt_metal/distributed/config/exabox_2_galaxy_rank_binding.yaml"
    echo "Using Wormhole configuration"
fi

case $CONFIG in
    "1x8-1x8")
        run_benchmark "1x8" "1x8" "${RANK_BINDING}"
        ;;
    "1x8-4x2")
        run_benchmark "1x8" "4x2" "${RANK_BINDING}"
        ;;
    "4x2-1x8")
        run_benchmark "4x2" "1x8" "${RANK_BINDING}"
        ;;
    "all")
        echo "Running all configurations..."
        run_benchmark "1x8" "1x8" "${RANK_BINDING}"
        run_benchmark "1x8" "4x2" "${RANK_BINDING}"
        run_benchmark "4x2" "1x8" "${RANK_BINDING}"
        ;;
    *)
        echo "Unknown configuration: $CONFIG"
        echo "Usage: $0 [--blackhole|-bh] [--config|-c CONFIG] [1x8-1x8|1x8-4x2|4x2-1x8|all]"
        echo ""
        echo "Options:"
        echo "  --blackhole, -bh    Use Blackhole MGD instead of Wormhole"
        echo "  --config, -c CONFIG  Configuration to run (default: all)"
        echo ""
        echo "Configurations:"
        echo "  1x8-1x8             Sender 1x8 → Receiver 1x8"
        echo "  1x8-4x2             Sender 1x8 → Receiver 4x2"
        echo "  4x2-1x8             Sender 4x2 → Receiver 1x8"
        echo "  all                 Run all configurations (default)"
        exit 1
        ;;
esac

rm -f ${HOSTFILE}

echo ""
echo "=========================================="
echo "Benchmark completed"
echo "End time: $(date)"
echo "=========================================="
