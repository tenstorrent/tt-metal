#!/usr/bin/env bash
# Environment for running tt-train python examples from this checkout.
export TT_METAL_HOME=/localdev/umales/tt-metal
export TT_METAL_RUNTIME_ROOT=/localdev/umales/tt-metal
export PYTHONPATH=/localdev/umales/tt-metal/build_Release/tt-train/sources/ttml:/localdev/umales/tt-metal/tt-train/sources/ttml:/localdev/umales/tt-metal
export ARCH_NAME=blackhole
# The 8 chips here enumerate as a 2x4 mesh; without this the Python-side MGD
# validation in ttml.open_device_mesh is skipped.
export TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto}
PY=/localdev/umales/tt-metal/python_env/bin/python
exec "$PY" "$@"
