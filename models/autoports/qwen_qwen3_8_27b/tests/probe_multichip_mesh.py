# SPDX-License-Identifier: Apache-2.0
"""Bounded externally; preserve discovered topology before trying fabric mesh open."""
import argparse
from pathlib import Path
import ttnn

parser = argparse.ArgumentParser()
parser.add_argument('--shape', type=int, nargs=2, default=[2, 2])
parser.add_argument('--fabric', default='FABRIC_2D')
parser.add_argument('--descriptor', required=True)
args = parser.parse_args()
print('CLUSTER_TYPE', ttnn.cluster.get_cluster_type(), flush=True)
source = Path(ttnn.cluster.serialize_cluster_descriptor())
Path(args.descriptor).write_text(source.read_text())
print('DESCRIPTOR', args.descriptor, flush=True)
ttnn.set_fabric_config(getattr(ttnn.FabricConfig, args.fabric))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(*args.shape), trace_region_size=0)
try:
    print('MESH', mesh, flush=True)
    print('LINKS', ttnn.get_num_links(mesh), flush=True)
finally:
    ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK', flush=True)
