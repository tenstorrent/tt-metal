import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config, block_shard_geometry
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

device = ttnn.open_device(device_id=0)
geos = [
    ((1, 1, 1024, 1280), (7, 8)),
    ((1, 1, 1024, 1280), (8, 7)),
    ((1, 1, 1024, 1920), (8, 8)),
    ((1, 1, 4096, 1152), (8, 8)),
    ((1, 1, 4096, 320), (8, 7)),
    ((1, 1, 4096, 640), (6, 8)),
    ((1, 1, 4096, 960), (8, 8)),
    ((2, 1, 256, 1280), (8, 3)),
    ((1, 1, 1024, 1280), (8, 8)),
]
for shape, grid in geos:
    for layout in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        try:
            mem = block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
        except Exception as e:
            continue
        x = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
            memory_config=mem,
        )
        try:
            p = pd._plan_sharded(x, 32, has_gamma=True, has_beta=True, in_place=True, tuning={})
            st = pd.staging_bytes(
                W=p.W, map_period=p.map_period, is_rm=p.is_rm, n_affine=2, sharded=True, pilot_cols=p.pilot_cols
            )
            print(
                f"GEO {shape} {grid} {'RM' if p.is_rm else 'TILE'} shard={block_shard_geometry(shape, grid, layout)} W={p.W} h={p.h} R={p.R} NU={p.NU} NSEG={p.NSEG} stage_pages={p.stage_pages} pilot_cols={p.pilot_cols} col_mask={p.col_mask} rowmaj={p.mcast_row_major} ops={p.ops_path} ring={p.ring_pages} staging={st} fp={p.footprint} budget={p.l1_budget}"
            )
        except Exception as e:
            print(f"GEO {shape} {grid} {layout} refused: {e}")
        x.deallocate()
ttnn.close_device(device)
