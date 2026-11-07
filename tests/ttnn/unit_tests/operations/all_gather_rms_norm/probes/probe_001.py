import ttnn, torch

cfg = ttnn._ttnn.fabric.FabricRouterConfig()
cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    cfg,
)
import ttnn, torch

cfg = ttnn._ttnn.fabric.FabricRouterConfig()
cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    cfg,
)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm_program_descriptor as pd

for L in (1, 2):
    chips, G = pd.plan(mesh, 1, True, L)
    for coord, ch in sorted(chips.items()):
        print("L", L, coord, {k: v for k, v in ch.ports.items()}, "links", ch.links)
print("payload", ttnn.get_tt_fabric_max_payload_size_bytes(), "budget", pd.cb_l1_budget(mesh))
chips, G = pd.plan(mesh, 1, True, 2)
allowed = pd.allowed_cores(mesh)
pu = {c for ch in chips.values() for c in ch.ports.values()}
nw = len(allowed - pu)
print("n_workers", nw, "ports_union", sorted(pu))
for shape in [(1, 1, 640, 384), (1, 1, 640, 512), (1, 1, 640, 1792)]:
    geo = pd.derive_geometry(shape, 4)
    T = ttnn.tile_size(ttnn.bfloat16)
    sz = pd.Sizes(T, 4096, 192, 10, 7, 4, True)
    sp = pd.select_split(geo, sz, nw, pd.cb_l1_budget(mesh), True, 2)
    print(
        shape,
        "k",
        sp.k,
        "w_pitch",
        sp.w_pitch,
        "n_groups",
        sp.n_groups,
        "rows_max",
        sp.rows_max,
        "cands",
        pd.nested_candidates(geo),
    )
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print(
        shape,
        "k",
        sp.k,
        "w_pitch",
        sp.w_pitch,
        "n_groups",
        sp.n_groups,
        "rows_max",
        sp.rows_max,
        "cands",
        pd.nested_candidates(geo),
    )
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
