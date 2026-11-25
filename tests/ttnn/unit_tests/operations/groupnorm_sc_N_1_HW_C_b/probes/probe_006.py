import ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor import (
    plan_or_raise,
    l1_cb_budget,
)

device = ttnn.open_device(device_id=0)
print("budget", l1_cb_budget(device))


class FakeT:
    def __init__(s, shape, layout):
        s.shape = shape
        s.layout = layout

    def device(s):
        return device


ns = {
    (1, 1, 1048576, 128): 1972728,
    (1, 1, 65536, 512): 436297,
    (1, 1, 262144, 256): 914726,
    (1, 1, 262144, 512): 2006876,
    (1, 1, 1048576, 256): 4213458,
    (1, 1, 16384, 1152): 242230,
    (1, 1, 16384, 256): 61871,
    (1, 1, 1048576, 64): 863864,
    (8, 1, 407040, 128): 6382888,
}
for shape, G in [
    ((1, 1, 1048576, 128), 32),
    ((1, 1, 65536, 512), 32),
    ((1, 1, 262144, 256), 32),
    ((1, 1, 262144, 512), 32),
    ((1, 1, 1048576, 256), 32),
    ((1, 1, 16384, 1152), 32),
    ((1, 1, 16384, 256), 16),
    ((1, 1, 1048576, 64), 16),
    ((8, 1, 407040, 128), 32),
]:
    p = plan_or_raise(FakeT(shape, ttnn.TILE_LAYOUT), G, has_gamma=True, has_beta=True)
    tot = 0
    res = 0
    for w in p.work:
        B = -(-w.rows_core // p.h)
        K = min(p.K_alloc, B)
        tot += B * w.num_blocks
        res += K * w.num_blocks
    rho = (tot - res) / tot
    X = shape[0] * shape[2] * shape[3] * 2
    bytes_ = X * (2 + rho)
    print(
        shape,
        f"gx={p.gx} W={p.W} h={p.h} T={p.T} cpi={p.cpi} cores={len(p.work)} B_max={p.B_max} K={p.K_alloc} rho={rho:.2f} fp={p.footprint} GBps={bytes_/ns[shape]:.0f}",
    )
ttnn.close_device(device)
