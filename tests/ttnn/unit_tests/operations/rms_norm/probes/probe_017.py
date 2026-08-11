import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
try:
    for shape, ml, layout in [
        ((2047, 2047), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 224, 3072), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
        ((13, 777, 1023), ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ]:
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=ttnn.bfloat16, device=device)
        tx = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
            memory_config=mc,
        )
        tg = ttnn.from_torch(
            torch.randn(shape[-1], dtype=torch.bfloat16).reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
        )
        geo = pd._Geometry(tx, tg)
        sv = pd._ShardView(tx, geo, geo.is_rm_in)
        infos = pd._shard_core_infos(geo, sv)
        groups = pd._sharded_groups(sv, infos)
        G = len(groups[0]["members"])
        C = sv.shard_w_tiles
        maxrows = max(g["row_count"] for g in groups)
        tail = 1 if any(m["w_elems"] % 32 for g in groups for m in g["members"]) else 0
        budget = pd._l1_cb_budget() - 2 * sv.bank_bytes
        print(
            f"DIAG {shape} {str(ml).split('.')[-1]} {str(layout).split('.')[-1]} shard={sv.shard_h}x{sv.shard_w} bank={sv.bank_bytes} C={C} G={G} maxrows={maxrows} tail={tail} Rt={geo.tensor_row_tiles} budget={budget}"
        )
        for d in pd._DEPTH_LADDER:
            fx, pr = pd._cb_bytes(
                geo, C, G, geo.is_rm_in, tail, depths=d, pin_in=not geo.is_rm_in, pin_out=not geo.is_rm_in
            )
            print(f"DIAG    depths={d} fixed={fx} per_row={pr} sum={fx+pr} fits={fx+pr<=budget}")
        tx.deallocate()
        tg.deallocate()
finally:
    ttnn.close_device(device)
