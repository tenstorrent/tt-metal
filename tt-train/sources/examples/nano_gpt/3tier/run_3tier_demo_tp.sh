echo "Running nanogpt 3tier demo"

# copy to LBOX_MPI_1
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.11:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.11:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.11:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/

# copy to LBOX_MPI_2
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.14:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.14:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.14:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/

# copy to LBOX_MPI_3
# scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.15:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/
# scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.15:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/
# scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.15:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/

# copy to LBOX_MPI_4
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt ttuser@11.228.0.16:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer ttuser@11.228.0.16:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/
scp  $TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator ttuser@11.228.0.16:$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt/

# copy config file to LBOX_MPI_xxx
scp  $TT_METAL_HOME/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.11:$TT_METAL_HOME/tt-train/configs/
scp  $TT_METAL_HOME/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.14:$TT_METAL_HOME/tt-train/configs/
# scp  $TT_METAL_HOME/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.15:$TT_METAL_HOME/tt-train/configs/
scp  $TT_METAL_HOME/tt-train/configs/training_shakespear_nanogpt_3tier.yaml ttuser@11.228.0.16:$TT_METAL_HOME/tt-train/configs/

echo "executables and config file succesfully copied"

echo "Running nano_gpt 3tier demo..."
mpirun --hostfile ~/mpi_hosts \
  -np 2 /bin/bash -c 'export TT_METAL_HOME=/home/ttuser/git/tt-metal && cd $TT_METAL_HOME && TT_METAL_LOGGER_LEVEL=FATAL ./tt-train/build/sources/examples/nano_gpt/nano_gpt -c tt-train/configs/training_shakespear_nanogpt_3tier.yaml -p 1' \
  : -np 1 /bin/bash -c 'export TT_METAL_HOME=/home/ttuser/git/tt-metal && cd $TT_METAL_HOME && TT_METAL_LOGGER_LEVEL=FATAL ./tt-train/build/sources/examples/nano_gpt/nano_gpt_aggregator -c tt-train/configs/training_shakespear_nanogpt_3tier.yaml -p 1' \
  : -np 1 /bin/bash -c 'export TT_METAL_HOME=/home/ttuser/git/tt-metal && cd $TT_METAL_HOME && TT_METAL_LOGGER_LEVEL=FATAL ./tt-train/build/sources/examples/nano_gpt/nano_gpt_optimizer -c tt-train/configs/training_shakespear_nanogpt_3tier.yaml -p 1'
