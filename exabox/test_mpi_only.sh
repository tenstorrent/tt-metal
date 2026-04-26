#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=wh_pod_8x16_1
#SBATCH --nodelist=wh-glx-a03u08,wh-glx-a04u08
#SBATCH --output=mpi_test_%j.out
#SBATCH --error=mpi_test_%j.err
#SBATCH --time=2:00

# Pure MPI connectivity test - no ttnn/tt-metal

echo "Testing MPI connectivity between: $SLURM_JOB_NODELIST"
echo "Start: $(date)"

# Create hostfile
HOSTFILE=/data/dmadic/hostfile_mpi_$$
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "$host slots=1"
done > $HOSTFILE

echo "Hostfile:"
cat $HOSTFILE
echo ""

# Simple MPI test using mpirun directly
mpirun --hostfile $HOSTFILE \
    --mca btl_tcp_if_exclude docker0,lo \
    --tag-output \
    bash -c 'echo "Hello from $(hostname) - rank $OMPI_COMM_WORLD_RANK of $OMPI_COMM_WORLD_SIZE"'

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: MPI SUCCESS - basic network connectivity OK"
else
    echo "RESULT: MPI FAILED"
fi

rm -f $HOSTFILE
