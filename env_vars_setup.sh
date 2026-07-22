#!/bin/bash

export TT_METAL_HOME=$(pwd)
export TT_METAL_RUNTIME_ROOT=${TT_METAL_HOME}
source $(pwd)/python_env/bin/activate
export TT_METAL_INSTALL_DIR=$(pwd)/build/install
export PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/tools:$PYTHONPATH
export PYTHONPATH=/home/alex/mpi-shfs/tenstorrent/tt-blaze:$PYTHONPATH
export PYTHONPATH=/home/alex/mpi-shfs/tenstorrent/blaze-nn:$PYTHONPATH
# Note: tt-smi is now integrated into tt-ctl, no separate PYTHONPATH needed
export TT_METAL_LIB_PATH=$TT_METAL_INSTALL_DIR/lib
export LD_LIBRARY_PATH=$TT_METAL_LIB_PATH:$LD_LIBRARY_PATH
# MPI paths
export OMPI_ROOT=/opt/openmpi-v5.0.7-ulfm/
export PATH=${OMPI_ROOT}/bin/:$PATH
export LD_LIBRARY_PATH=${OMPI_ROOT}/lib/:$LD_LIBRARY_PATH
export VLLM_TARGET_DEVICE="tt"

# Enable tt-ctl tab completion if installed
if command -v tt-ctl &> /dev/null; then
    eval "$(_TT_CTL_COMPLETE=bash_source tt-ctl)"
fi
