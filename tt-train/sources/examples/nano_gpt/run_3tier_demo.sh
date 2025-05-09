echo "Running nanogpt 3tier demo"

# copy to LBOX_MPI_1
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.11:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.11:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.11:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/

# copy to LBOX_MPI_2
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.14:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.14:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.14:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/

# copy to LBOX_MPI_3
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.15:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.15:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.15:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/

# copy config file to LBOX_MPI_xxx
scp  /home/ttuser/git/tt-metal/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.11:/home/ttuser/git/tt-metal/tt-train/configs/
scp  /home/ttuser/git/tt-metal/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.14:/home/ttuser/git/tt-metal/tt-train/configs/
scp  /home/ttuser/git/tt-metal/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.15:/home/ttuser/git/tt-metal/tt-train/configs/

echo "executables and config file succesfully copied"

echo "Running nano_gpt 3tier demo..."
mpirun --hostfile ~/mpi_hosts \
  -x TT_METAL_LOGGER_LEVEL=FATAL \
  -x TT_METAL_HOME=/home/ttuser/git/tt-metal \
  -np 1 /bin/bash -c 'cd /home/ttuser/git/tt-metal && ./tt-train/build/sources/examples/nano_gpt/nano_gpt' \
  : -np 1 /bin/bash -c 'cd /home/ttuser/git/tt-metal && ./tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator' \
  : -np 1 /bin/bash -c 'cd /home/ttuser/git/tt-metal && ./tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer'
