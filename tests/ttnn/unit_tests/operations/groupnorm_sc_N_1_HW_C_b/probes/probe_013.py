import torch, ttnn
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b import tag_shard_fit
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b import feature_spec as fs

device = ttnn.open_device(device_id=0)
try:
    seen = set()
    for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
        for inp in fs.INPUTS:
            if len(inp) < 3 or (inp, layout) in seen:
                continue
            seen.add((inp, layout))
            if tag_shard_fit(inp, {"layout": layout}) != "even":
                continue
            shape, G, grid = inp
            mem = block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
            x = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=layout,
                device=device,
                memory_config=mem,
            )
            try:
                p = pd.plan_or_raise(x, G, has_gamma=True, has_beta=True)
                print(
                    f"PLAN {'TILE' if layout==ttnn.TILE_LAYOUT else 'RM  '} {shape} G={G} grid={grid} W={p.W} h={p.h} T={p.T} rows_core={p.rows_core_max} B={p.B_max} K_max={p.K_max} K={p.K_alloc} cpi={p.cpi} fp={p.footprint} budget={p.l1_budget}"
                )
            except NotImplementedError as e:
                print(f"REFUSE {layout} {shape} {grid}: {e}")
            x.deallocate()
finally:
    ttnn.close_device(device)
