import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor import (
    plan_or_raise,
    footprint_bytes,
)

device = ttnn.open_device(device_id=0)
print("unreserved", ttnn.get_max_worker_l1_unreserved_size())
print([a for a in dir(ttnn) if "l1" in a.lower()])


class FakeT:
    def __init__(s, shape, layout):
        s.shape = shape
        s.layout = layout

    def device(s):
        return device


for shape, G in [
    ((1, 1, 1048576, 128), 32),
    ((1, 1, 262144, 256), 32),
    ((1, 1, 262144, 512), 32),
    ((1, 1, 1048576, 256), 32),
    ((1, 1, 1048576, 64), 16),
    ((8, 1, 407040, 128), 32),
    ((1, 1, 16384, 1152), 32),
    ((1, 1, 65536, 512), 32),
    ((1, 1, 16384, 256), 16),
]:
    p = plan_or_raise(FakeT(shape, ttnn.TILE_LAYOUT), G, has_gamma=True, has_beta=True)
    print(
        shape,
        dict(
            W=p.W,
            h=p.h,
            T=p.T,
            R=p.R,
            cpi=p.cpi,
            U=p.U,
            B_max=p.B_max,
            K_max=p.K_max,
            K=p.K_alloc,
            stream=p.stream,
            fp=p.footprint,
            budget=p.l1_budget,
            cores=len(p.work),
        ),
    )
ttnn.close_device(device)
