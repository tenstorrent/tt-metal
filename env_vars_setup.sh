
#!/bin/bash
source /opt/venv/bin/activate
export TT_METAL_HOME=$(pwd)
export TT_METAL_INSTALL_DIR=/opt/tt-metal/
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export TT_METAL_LIB_PATH=$TT_METAL_INSTALL_DIR/lib
export LD_LIBRARY_PATH=$TT_METAL_LIB_PATH:$LD_LIBRARY_PATH
# MPI paths
export OMPI_ROOT=/opt/openmpi-v5.0.7-ulfm/
export PATH=${OMPI_ROOT}/bin/:$PATH
export LD_LIBRARY_PATH=${OMPI_ROOT}/lib/:$LD_LIBRARY_PATH
