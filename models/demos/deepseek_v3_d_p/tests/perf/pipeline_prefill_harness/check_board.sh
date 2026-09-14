#!/usr/bin/env bash
# Exit 0 if the fabric can still map an 8x4 torus_xy mesh, non-zero if the board needs a real reset.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/env.sh"
cd "$TT_METAL_HOME" || exit 2
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
timeout 150 "$PY" -c "
import ttnn
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config as C
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.FabricReliabilityMode.RELAXED_INIT, None,
    ttnn.FabricTensixConfig.DISABLED, ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT,
    create_fabric_router_config(max_payload_size=C.FABRIC_PAYLOAD_SIZE))
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8,4)); print('HEALTH_OK', md.shape); ttnn.close_mesh_device(md)
" 2>&1 | grep -q HEALTH_OK
