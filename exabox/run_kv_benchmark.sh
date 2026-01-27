#!/bin/bash
#SBATCH --job-name=kv_bench
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=wh_pod_8x16_1
#SBATCH --nodelist=wh-glx-a03u08,wh-glx-a04u08
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
source python_env/bin/activate

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

    echo ""
    echo "============================================================"
    echo "Running: ${sender_mesh} → ${receiver_mesh}"
    echo "============================================================"

    tt-run --verbose \
        --rank-binding tests/tt_metal/distributed/config/exabox_2_galaxy_rank_binding.yaml \
        --mpi-args "${MPI_OPTS}" \
        python tests/ttnn/distributed/benchmark_galaxy_kv_transfer.py \
            --sender-mesh ${sender_mesh} \
            --receiver-mesh ${receiver_mesh}

    local exit_code=$?
    echo "Exit code: ${exit_code}"
    return ${exit_code}
}

# Configuration to run (default: all three)
CONFIG=${1:-all}

case $CONFIG in
    "1x8-1x8")
        run_benchmark "1x8" "1x8"
        ;;
    "1x8-4x2")
        run_benchmark "1x8" "4x2"
        ;;
    "4x2-1x8")
        run_benchmark "4x2" "1x8"
        ;;
    "all")
        echo "Running all configurations..."
        run_benchmark "1x8" "1x8"
        run_benchmark "1x8" "4x2"
        run_benchmark "4x2" "1x8"
        ;;
    *)
        echo "Unknown configuration: $CONFIG"
        echo "Usage: $0 [1x8-1x8|1x8-4x2|4x2-1x8|all]"
        exit 1
        ;;
esac

rm -f ${HOSTFILE}

echo ""
echo "=========================================="
echo "Benchmark completed"
echo "End time: $(date)"
echo "=========================================="
