echo "Running demo_multihost"
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/demo_multihost/demo_multihost ttuser@11.228.0.11:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/demo_multihost/
echo "demo_multihost succesfully copied"
mpirun --hostfile ~/mpi_hosts -np 2 /bin/bash -c "cd /home/ttuser/git/tt-metal && TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_HOME=/home/ttuser/git/tt-metal /home/ttuser/git/tt-metal/tt-train/build/sources/examples/demo_multihost/demo_multihost"
