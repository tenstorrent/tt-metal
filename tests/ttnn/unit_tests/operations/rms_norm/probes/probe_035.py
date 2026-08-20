import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as D, default_compute_kernel_config

device = ttnn.open_device(device_id=0)


def loose():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def show(shape, dtype, layout, glayout, cfg, levers=None, tag=""):
    x = ttnn.from_torch(torch.zeros(shape), dtype=dtype, layout=layout, device=device)
    g = ttnn.from_torch(torch.zeros((1, 1, 1, shape[-1])), dtype=dtype, layout=glayout, device=device)
    p = D.blocking_plan(x, g, x, device, cfg, levers)
    print(
        f"{tag:44s} G={p.group_size:4d} Wt_core={p.Wt_core:4d} reg={p.regime} bht={p.BLOCK_HT} nrb={p.num_row_blocks} used={p.groups_used}"
    )
    ttnn.deallocate(x)
    ttnn.deallocate(g)


T, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
# does the split reach the OTHER dtypes / configs?
show(
    (1, 1, 32, 7168), ttnn.float32, T, T, default_compute_kernel_config(), tag="fp32 + fp32_dest_acc=True (1,1,32,7168)"
)
show((1, 1, 32, 7168), ttnn.float32, T, T, loose(), tag="fp32 + fp32_dest_acc=False")
show((1, 1, 32, 7168), ttnn.bfloat8_b, T, T, default_compute_kernel_config(), tag="bfloat8_b default")
show((1, 1, 32, 7168), ttnn.bfloat8_b, T, T, loose(), tag="bfloat8_b loose")
c = default_compute_kernel_config()
c.math_fidelity = ttnn.MathFidelity.LoFi
show((1, 1, 32, 7168), ttnn.bfloat16, T, T, c, tag="bf16 LoFi")
c2 = default_compute_kernel_config()
c2.dst_full_sync_en = True
show((1, 1, 32, 7168), ttnn.bfloat16, T, T, c2, tag="bf16 dst_full_sync_en=True")
show((1, 32, 7168), ttnn.bfloat16, T, T, loose(), tag="rank 3 (1,32,7168)")
show((32, 7168), ttnn.bfloat16, T, T, loose(), tag="rank 2 (32,7168)")
show((1, 1, 100, 7168), ttnn.bfloat16, T, T, loose(), tag="h_non_aligned wide (1,1,100,7168)")
show((1, 1, 1024, 7168), ttnn.bfloat16, RM, RM, loose(), tag="RM input (1,1,1024,7168)")
# levers
show((1, 1, 32, 7168), ttnn.bfloat16, T, T, loose(), dict(w_split=0), tag="focus w_split=0 (off arm)")
show((1, 1, 32, 7168), ttnn.bfloat16, T, T, loose(), dict(w_group=56), tag="focus w_group=56")
show((1, 1, 32, 7168), ttnn.bfloat16, T, T, loose(), dict(active_cores=32), tag="focus active_cores=32 (A0 off arm)")
ttnn.close_device(device)
