import ttnn
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    cases = [
        ((1, 1, 1048576, 128), 32, ttnn.TILE_LAYOUT),
        ((1, 1, 65536, 512), 32, ttnn.TILE_LAYOUT),
        ((1, 1, 262144, 256), 32, ttnn.TILE_LAYOUT),
        ((1, 1, 262144, 512), 32, ttnn.TILE_LAYOUT),
        ((1, 1, 1048576, 256), 32, ttnn.TILE_LAYOUT),
        ((1, 1, 16384, 1152), 32, ttnn.TILE_LAYOUT),
        ((1, 1, 16384, 256), 16, ttnn.TILE_LAYOUT),
        ((1, 1, 1048576, 64), 16, ttnn.TILE_LAYOUT),
        ((8, 1, 407040, 128), 32, ttnn.TILE_LAYOUT),
    ]
    for shape, G, lay in cases:
        t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, lay, dev, ttnn.DRAM_MEMORY_CONFIG)
        p = pd.plan_or_raise(t, G, has_gamma=True, has_beta=True)
        print(
            "PLAN",
            shape,
            "W",
            p.W,
            "T",
            p.T,
            "B_max",
            p.B_max,
            "K_max",
            p.K_max,
            "K_alloc",
            p.K_alloc,
            "slack",
            p.l1_budget - p.footprint,
        )
        t.deallocate()
finally:
    ttnn.close_device(dev)
