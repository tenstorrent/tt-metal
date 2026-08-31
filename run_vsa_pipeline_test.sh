#!/bin/bash
cd "$(dirname "$0")"
export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD"
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
export MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/data/cglagovich/tt_dit_cache_claude
mkdir -p "$TT_DIT_CACHE_DIR"
exec ./python_env/bin/python -m pytest "$@"
