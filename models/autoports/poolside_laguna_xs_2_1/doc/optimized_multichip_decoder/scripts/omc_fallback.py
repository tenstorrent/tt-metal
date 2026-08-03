"""Dynamic fallback audit for OptimizedMultichipDecoder on the 1x4 mesh."""
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_multichip_decoder import OptimizedMultichipDecoder

H = 2048
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=90_000_000)
mm = ttnn.ReplicateTensorToMesh(dev)
try:
    cfg = R.build_config()
    for L in (0, 1, 4):
        raw = W.load_layer_tensors(L)
        dec = OptimizedMultichipDecoder.from_state_dict(
            raw, hf_config=cfg, layer_idx=L, mesh_device=dev, max_seq_len=128
        )
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=128, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        x = ttnn.from_torch(
            torch.randn(1, 64, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mm
        )
        xd = ttnn.from_torch(
            torch.randn(1, 1, 1, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mm
        )
        cur = ttnn.from_torch(
            torch.tensor([64], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        ridx = ttnn.from_torch(
            torch.tensor([[64]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        ttnn.CONFIG.throw_exception_on_fallback = True
        try:
            dec.prefill_forward(x, kv, pt, user_id=0, start_pos=0)
            dec.decode_forward(xd, cur, ridx, pt, kv)
            ttnn.synchronize_device(dev)
            print(f"layer {L} FALLBACK_CLEAN")
        finally:
            ttnn.CONFIG.throw_exception_on_fallback = False
    print("FALLBACK_AUDIT_CLEAN")
finally:
    ttnn.close_mesh_device(dev)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
