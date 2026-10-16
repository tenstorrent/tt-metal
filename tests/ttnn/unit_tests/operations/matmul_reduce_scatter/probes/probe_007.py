import ttnn

d = ttnn.open_device(device_id=0)
b = int(ttnn._ttnn.reports.get_device_info(d).l1_bank_size)
print("BANK", b, b - 65536, (b - 65536) / 1024)
from ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter_program_descriptor import footprint

for kb in (16, 8, 4):
    for dA in (3, 2):
        import ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter_program_descriptor as m

        m.DEPTH_IN0 = dA
        for dh in (2, 1):
            m.DEPTH_HOLD = dh
            f = footprint(
                kb, True, Kt=64, core_m=2, core_n=12, tA=2048, tW=1088, scatter_dim=-2, slots={"fwd": 2, "bwd": 2}
            )
            print("FP kb", kb, "dA", dA, "dh", dh, f / 1024, f <= b - 65536)
ttnn.close_device(d)
