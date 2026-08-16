import json
import sys

sys.path.insert(
    0, "/home/raahem/tt-metal/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/optimized_full_model/probes"
)
from sdpa_depth_probe import MESH_SHAPE, bench

import ttnn

mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
out = []
try:
    for depth, pos in [
        (1024, 0),
        (1024, 31),
        (1024, 63),
        (1024, 127),
        (1024, 128),
        (1024, 131),
        (1024, 255),
        (1024, 511),
        (1024, 1023),
        (4096, 131),
        (4096, 255),
    ]:
        ms = bench(mesh, depth, pos)
        out.append((depth, pos, ms))
        print(f"depth {depth:6d} cur_pos {pos:6d}  {ms*1000:8.2f} us", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
json.dump(
    out,
    open(
        "/home/raahem/tt-metal/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/optimized_full_model/probes/sdpa_curpos_probe.json",
        "w",
    ),
    indent=2,
)
