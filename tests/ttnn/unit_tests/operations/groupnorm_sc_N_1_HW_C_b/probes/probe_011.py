import torch, ttnn
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    shapes = [
        ((1, 1, 1048576, 128), 32),
        ((1, 1, 65536, 512), 32),
        ((1, 1, 262144, 256), 32),
        ((1, 1, 262144, 512), 32),
        ((1, 1, 1048576, 256), 32),
        ((1, 1, 16384, 1152), 32),
        ((1, 1, 16384, 256), 16),
        ((1, 1, 1048576, 64), 16),
        ((8, 1, 407040, 128), 32),
    ]
    print("BUDGET", pd.l1_cb_budget(device))
    orig = pd.footprint_bytes
    for extra in (0, 2048):

        def fp(**kw):
            return orig(**kw) + extra * kw["W"]

        pd.footprint_bytes = fp
        for shp, G in shapes:
            x = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(shp)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            p = pd.plan_or_raise(x, G, has_gamma=True, has_beta=True)
            nb = max(w.num_blocks for w in p.work)
            print(
                f"SHAPE extra={extra} {shp} W={p.W} h={p.h} T={p.T} B_max={p.B_max} K_max={p.K_max} K_alloc={p.K_alloc} reread_chunks/block={max(0,p.B_max-p.K_alloc)} blocks={nb}"
            )
            x.deallocate()
    pd.footprint_bytes = orig
finally:
    ttnn.close_device(device)
