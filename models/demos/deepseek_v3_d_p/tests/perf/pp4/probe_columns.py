"""Probe: which PHYSICAL devices make up each (8,1) column submesh of the 8x4 galaxy?

The multi-process prefill runner gives each rank its own mesh via TT_VISIBLE_DEVICES, so an
[8,1]-per-rank pipeline needs the physical device ids of each logical column. Physical discovery
only exposes 2x4 trays/slices, so read the mapping off the live 8x4 mesh instead: open it with the
torus_xy descriptor (the production profile) and ask each (8,1) submesh for its device ids.
"""
import ttnn
from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config

ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.RELAXED_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    create_fabric_router_config(max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE),
)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
print("PROBE full mesh shape:", md.shape)
print("PROBE full mesh device ids:", md.get_device_ids())

for shape in [(8, 1), (2, 4)]:
    subs = md.create_submeshes(ttnn.MeshShape(*shape))
    print(f"PROBE submeshes for {shape}: n={len(subs)}")
    for i, sm in enumerate(subs):
        print(f"PROBE   {shape} rank{i}: shape={sm.shape} device_ids={sm.get_device_ids()}")
    # submeshes are owned by the parent; re-carving a different shape needs them released first
    for sm in subs:
        try:
            md.remove_submesh(sm)
        except Exception as e:
            print(f"PROBE   (remove_submesh unsupported: {e})")
            break
    else:
        continue
    break

ttnn.close_mesh_device(md)
print("PROBE done")
