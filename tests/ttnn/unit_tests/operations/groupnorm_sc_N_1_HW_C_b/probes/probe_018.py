import ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b as mod_fn
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b import validate

dev = ttnn.open_device(device_id=0)
cases = [
    ("vae1Mx128", (1, 1, 1048576, 128), 32),
    ("vae64kx512", (1, 1, 65536, 512), 32),
    ("vae256kx256", (1, 1, 262144, 256), 32),
    ("vae256kx512", (1, 1, 262144, 512), 32),
    ("vae1Mx256", (1, 1, 1048576, 256), 32),
    ("ref16kx1152", (1, 1, 16384, 1152), 32),
    ("flx16kx256", (1, 1, 16384, 256), 16),
    ("flx1Mx64", (1, 1, 1048576, 64), 16),
    ("mochi", (8, 1, 407040, 128), 32),
]
for name, shape, G in cases:
    x = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    g = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, shape[-1]]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    p = validate(x, G, gamma=g, beta=g)
    ncores = len(p.work)
    rows = [w.rows_core for w in p.work]
    nb = [w.num_blocks for w in p.work]
    I = shape[0] * shape[2] * shape[3] * 2
    B = p.B_max
    K = min(p.K_alloc, B)
    reread = I * (B - K) / B
    print(
        f"{name}: cores={ncores} gx={p.gx} W={p.W} cpi={p.cpi} h={p.h} T={p.T} B_max={B} K_alloc={p.K_alloc} K_max={p.K_max} stream={p.stream} rows_core_max={p.rows_core_max} blocks/core={max(nb)} U={p.U} I_MiB={I/2**20:.1f} reread_MiB={reread/2**20:.1f} total_MiB={(2*I+reread)/2**20:.1f}"
    )
    x.deallocate()
    g.deallocate()
ttnn.close_device(dev)
