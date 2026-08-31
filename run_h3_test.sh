#!/bin/bash
# Run an H3 test on the 4x8 BH galaxy with the right env. Usage: ./run_h3_test.sh <pytest args...>
cd "$(dirname "$0")"
export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD"
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
exec ./python_env/bin/python -m pytest "$@"
