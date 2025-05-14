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
# scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.15:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
# scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.15:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
# scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.15:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/

# copy to LBOX_MPI_4
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.16:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.16:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/
scp  /home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.16:/home/ttuser/git/tt-metal/tt-train/build/sources/examples/nano_gpt/

# copy config file to LBOX_MPI_xxx
scp  /home/ttuser/git/tt-metal/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.11:/home/ttuser/git/tt-metal/tt-train/configs/
scp  /home/ttuser/git/tt-metal/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.14:/home/ttuser/git/tt-metal/tt-train/configs/
# scp  /home/ttuser/git/tt-metal/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.15:/home/ttuser/git/tt-metal/tt-train/configs/
scp  /home/ttuser/git/tt-metal/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.16:/home/ttuser/git/tt-metal/tt-train/configs/

echo "executables and config file succesfully copied"

echo "Running nano_gpt 3tier demo..."
mpirun --hostfile ~/mpi_hosts \
  -np 2 /bin/bash -c 'cd /home/ttuser/git/tt-metal && TT_METAL_HOME=/home/ttuser/git/tt-metal TT_METAL_LOGGER_LEVEL=FATAL ./tt-train/build/sources/examples/nano_gpt/nano_gpt -d 1 -c tt-train/configs/training_shakespear_nanogpt_3tier.yaml' \
  : -np 1 /bin/bash -c 'cd /home/ttuser/git/tt-metal && TT_METAL_HOME=/home/ttuser/git/tt-metal TT_METAL_LOGGER_LEVEL=FATAL ./tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator -d 1 -c tt-train/configs/training_shakespear_nanogpt_3tier.yaml' \
  : -np 1 /bin/bash -c 'cd /home/ttuser/git/tt-metal && TT_METAL_HOME=/home/ttuser/git/tt-metal TT_METAL_LOGGER_LEVEL=FATAL ./tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer -d 1 -c tt-train/configs/training_shakespear_nanogpt_3tier.yaml'
